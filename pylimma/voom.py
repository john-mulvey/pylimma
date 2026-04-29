# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   voom.R                    Copyright (C) 2011-2026 Gordon Smyth, Charity Law
#   vooma.R                   Copyright (C) 2012-2024 Gordon Smyth, Charity Law,
#                                                     Mengbo Li
#   voomaLmFit.R              Copyright (C) 2023-2024 Mengbo Li, Gordon Smyth
#   voomWithQualityWeights.R  Copyright (C) 2014-2025 Matt Ritchie, Cynthia Liu,
#                                                     Gordon Smyth
# Python port: Copyright (C) 2026 John Mulvey
"""
voom: Transform RNA-seq counts for linear modelling.

Implements voom and related functions from R limma for RNA-seq analysis:
- voom(): Transform counts to log-CPM with precision weights
- voom_with_quality_weights(): voom combined with sample quality weights
- vooma(): voom-like weights for non-count expression data
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy import interpolate, linalg
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

from .lmfit import lm_fit, _parse_design
from .utils import choose_lowess_span
from .classes import EList, get_eawp, put_eawp, _is_anndata


def _draw_voom_trend(
    sx: np.ndarray,
    sy: np.ndarray,
    x_line: np.ndarray,
    y_line: np.ndarray,
    *,
    xlab: str,
    ylab: str,
    title: str,
    ax=None,
) -> None:
    """Render a voom/vooma mean-variance trend plot via matplotlib.

    Lazy import matches the pattern used by ``qqt``/``qqf`` in
    ``pylimma/utils.py``: matplotlib is an optional dependency.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import warnings
        warnings.warn("matplotlib not available for plotting")
        return

    if ax is None:
        _fig, ax = plt.subplots()
    ax.scatter(sx, sy, s=2, c="black")
    ax.plot(x_line, y_line, color="red")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)


def _draw_array_weights_bar(aw: np.ndarray, ax=None, col=None) -> None:
    """Bar plot of sample-specific array weights (matches voomWithQualityWeights).

    ``col`` forwards R's ``col`` argument to matplotlib's bar ``color``
    keyword so ``voomWithQualityWeights(counts, design, col="red")`` in
    R translates directly.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import warnings
        warnings.warn("matplotlib not available for plotting")
        return

    if ax is None:
        _fig, ax = plt.subplots()
    bar_kwargs = {}
    if col is not None:
        bar_kwargs["color"] = col
    ax.bar(np.arange(1, len(aw) + 1), aw, **bar_kwargs)
    ax.axhline(1.0, color="red", linestyle="--")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Weight")
    ax.set_title("Sample-specific weights")


def _normalize_between_arrays(y: np.ndarray, method: str = "none") -> np.ndarray:
    """Thin dispatcher onto :func:`pylimma.normalize.normalize_between_arrays`."""
    from .normalize import normalize_between_arrays
    return normalize_between_arrays(y, method=method)


def voom(
    counts,
    design: np.ndarray | None = None,
    lib_size: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    offset_prior: np.ndarray | None = None,
    normalize_method: str = "none",
    block: np.ndarray | None = None,
    correlation: float | None = None,
    weights: np.ndarray | None = None,
    span: float = 0.5,
    adaptive_span: bool = True,
    plot: bool = False,
    save_plot: bool = False,
    *,
    out_layer: str = "voom_E",
    weights_layer: str = "voom_weights",
    key: str = "voom",
    layer: str | None = None,
):
    """
    Transform RNA-seq counts for linear modelling with mean-variance weighting.

    Transforms count data to log2-counts per million (log-CPM), estimates the
    mean-variance relationship, and computes observation weights that can be
    used in weighted linear models via lm_fit().

    Parameters
    ----------
    counts : ndarray
        Matrix of counts, shape (n_genes, n_samples). Must be non-negative
        with no NA values.
    design : ndarray, optional
        Design matrix, shape (n_samples, n_coefficients). If None, uses an
        intercept-only model.
    lib_size : ndarray, optional
        Library sizes for each sample. If None, computed as column sums of
        counts.
    offset : ndarray, optional
        Offset matrix, shape (n_genes, n_samples). If provided without
        offset_prior, offset_prior is computed as offset - rowMeans(offset).
    offset_prior : ndarray, optional
        Pre-centered offset matrix, shape (n_genes, n_samples). Applied as:
        lib_size_matrix = exp(log(lib_size_matrix) + offset_prior).
        Takes precedence over offset if both are provided.
    normalize_method : str, default "none"
        Normalization method. Currently only "none" is supported.
    block : ndarray, optional
        Factor indicating blocking structure for samples.
    correlation : float, optional
        Intra-block correlation (required if block is specified).
    weights : ndarray, optional
        Prior weights for samples or observations.
    span : float, default 0.5
        LOWESS span for trend fitting (used if adaptive_span=False).
    adaptive_span : bool, default True
        If True, choose span adaptively based on number of genes.
    save_plot : bool, default False
        If True, include trend data in output for plotting.

    Returns
    -------
    dict
        E : ndarray
            Log2-CPM expression matrix, shape (n_genes, n_samples).
        weights : ndarray
            Precision weights, shape (n_genes, n_samples).
        design : ndarray
            Design matrix used for fitting.
        lib_size : ndarray
            Library sizes.
        span : float, optional
            LOWESS span used (only if adaptive_span=True).
        voom_xy : dict, optional
            Trend data for plotting (only if save_plot=True).
        voom_line : dict, optional
            LOWESS fit for plotting (only if save_plot=True).
        offset_prior : ndarray, optional
            The offset_prior matrix used (only if offset or offset_prior provided).

    Notes
    -----
    The voom method [1]_ transforms count data to log2-CPM, then estimates
    the mean-variance trend from the residual standard deviations of a
    preliminary linear model fit. The trend is used to compute precision
    weights for each observation.

    References
    ----------
    .. [1] Law CW, Chen Y, Shi W, Smyth GK (2014). voom: precision weights
           unlock linear model analysis tools for RNA-seq read counts.
           Genome Biology 15:R29.
    """
    # Polymorphic input: ndarray / dict / EList / AnnData (R limma's voom.EList
    # and .default branches are collapsed here via get_eawp).
    original_input = counts
    eawp = get_eawp(counts, layer=layer)
    counts = np.asarray(eawp["exprs"], dtype=np.float64)
    # EList-specific warn-and-proceed: R's voom calls as.matrix(EList)
    # (voom.R:32) which drops every slot except E. pylimma keeps the more
    # useful behaviour of honouring EList['design'] and EList['weights']
    # but warns so users porting R code know the divergence is there.
    # See known_diff_voom_elist_warning.md for rationale.
    _input_is_elist = isinstance(original_input, EList)
    if design is None and eawp.get("design") is not None:
        if _input_is_elist:
            warnings.warn(
                "pylimma's voom is using 'design' from the EList's design "
                "slot. R's voom would silently discard it via "
                "as.matrix(EList). To match R behaviour, pass "
                "design=el['design'] explicitly (or clear the slot).",
                UserWarning,
            )
        design = eawp["design"]
    if weights is None and eawp.get("weights") is not None:
        if _input_is_elist:
            warnings.warn(
                "pylimma's voom is using prior weights from the EList's "
                "weights slot. R's voom would silently discard them via "
                "as.matrix(EList). To match R, pass "
                "weights=el['weights'] explicitly.",
                UserWarning,
            )
        weights = np.asarray(eawp["weights"], dtype=np.float64)

    # Check counts
    n_genes, n_samples = counts.shape
    if n_genes < 2:
        raise ValueError("Need at least two genes to fit a mean-variance trend")
    if np.any(np.isnan(counts)):
        raise ValueError("NA counts not allowed")
    if np.min(counts) < 0:
        raise ValueError("Negative counts not allowed")

    # Parse design. Handles formula strings (via patsy + adata.obs when
    # input is AnnData), ndarray / DataFrame / patsy DesignMatrix, or
    # None (intercept-only). Matches lm_fit's dispatch.
    sample_data = original_input.obs if _is_anndata(original_input) else None
    design, _ = _parse_design(design, data=sample_data, n_samples=n_samples)

    # Check lib_size
    if lib_size is None:
        lib_size = np.sum(counts, axis=0)
    lib_size = np.asarray(lib_size, dtype=np.float64)

    lib_size_matrix = np.broadcast_to(lib_size, (n_genes, n_samples)).copy()

    # Handle offset parameters
    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64)
        if offset.shape != counts.shape:
            raise ValueError("counts and offset must have equal dimensions.")
        if offset_prior is None:
            offset_prior = offset - np.mean(offset, axis=1, keepdims=True)
        else:
            warnings.warn("Ignoring offset in favor of offset_prior. Should not set both.")
            offset = None

    if offset_prior is not None:
        offset_prior = np.asarray(offset_prior, dtype=np.float64)
        if offset_prior.shape != counts.shape:
            raise ValueError("counts and offset_prior must have equal dimensions.")
        lib_size_matrix = np.exp(np.log(lib_size_matrix) + offset_prior)

    # Choose span based on number of genes
    if adaptive_span:
        span = choose_lowess_span(n_genes, small_n=50, min_span=0.3, power=1/3)

    # Compute log2-counts-per-million
    # y = log2((counts+0.5)/(lib.size.matrix+1)*1e6)
    y = np.log2((counts + 0.5) / (lib_size_matrix + 1) * 1e6)
    y = _normalize_between_arrays(y, method=normalize_method)

    # Fit linear model
    fit = lm_fit(y, design, block=block, correlation=correlation, weights=weights)

    # Compute Amean if not present
    if fit.get("Amean") is None:
        fit["Amean"] = np.nanmean(y, axis=1)

    # If no replication found, set all weights to 1
    n_with_reps = np.sum(fit["df_residual"] > 0)
    if n_with_reps < 2:
        if n_with_reps == 0:
            warnings.warn(
                "The experimental design has no replication. Setting weights to 1."
            )
        elif n_with_reps == 1:
            warnings.warn(
                "Only one gene with any replication. Setting weights to 1."
            )
        return put_eawp(
            {
                "E": y,
                "weights": np.ones_like(y),
                "design": design,
                "lib_size": lib_size,
                "targets": {"lib_size": lib_size},
            },
            original_input,
            out_layer=out_layer,
            weights_layer=weights_layer,
            uns_key=key,
        )

    # Fit lowess trend to sqrt-standard-deviations by log-count-size
    # sx = fit$Amean + mean(log2(lib.size+1)) - log2(1e6)
    sx = fit["Amean"] + np.mean(np.log2(lib_size + 1)) - np.log2(1e6)
    sy = np.sqrt(fit["sigma"])

    # Exclude all-zero genes
    all_zero = np.sum(counts, axis=1) == 0
    if np.any(all_zero):
        sx_fit = sx[~all_zero]
        sy_fit = sy[~all_zero]
    else:
        sx_fit = sx
        sy_fit = sy

    # Fit LOWESS trend using statsmodels (matches R's lowess closely)
    # it=3 matches R's default of 3 robustifying iterations
    # delta=0.01*range(x) matches R's default delta parameter
    x_range = sx_fit.max() - sx_fit.min()
    lowess_result = sm_lowess(
        sy_fit, sx_fit, frac=span, it=3, delta=0.01 * x_range, return_sorted=True
    )
    x_sorted = lowess_result[:, 0]
    y_sorted = lowess_result[:, 1]

    # approxfun with rule=2 extrapolates using boundary values
    f = interpolate.interp1d(
        x_sorted, y_sorted,
        kind="linear",
        bounds_error=False,
        fill_value=(y_sorted[0], y_sorted[-1])
    )

    # Compute fitted values from linear model
    # fitted.values = coefficients %*% t(design)
    coefficients = fit["coefficients"]
    rank = fit["rank"]

    if rank < design.shape[1]:
        pivot = fit["pivot"]
        j = pivot[:rank]
        fitted_values = coefficients[:, j] @ design[:, j].T
    else:
        fitted_values = coefficients @ design.T

    # Convert to fitted counts
    # fitted.cpm = 2^fitted.values
    # fitted.count = 1e-6 * fitted.cpm * (lib.size.matrix+1)
    # fitted.logcount = log2(fitted.count)
    fitted_cpm = 2 ** fitted_values
    fitted_count = 1e-6 * fitted_cpm * (lib_size_matrix + 1)
    fitted_logcount = np.log2(fitted_count)

    # Apply trend to individual observations
    # w = 1/f(fitted.logcount)^4
    trend_values = f(fitted_logcount)
    w = 1 / trend_values ** 4

    # Build output
    out = {
        "E": y,
        "weights": w,
        "design": design,
        "lib_size": lib_size,
    }

    if adaptive_span:
        out["span"] = span

    if plot:
        _draw_voom_trend(
            sx_fit, sy_fit, x_sorted, y_sorted,
            xlab="log2( count size + 0.5 )",
            ylab="Sqrt( standard deviation )",
            title="voom: Mean-variance trend",
        )

    if save_plot:
        out["voom_xy"] = {
            "x": sx_fit,
            "y": sy_fit,
            "xlab": "log2( count size + 0.5 )",
            "ylab": "Sqrt( standard deviation )",
        }
        out["voom_line"] = {
            "x": x_sorted,
            "y": y_sorted,
        }

    if offset_prior is not None:
        out["offset_prior"] = offset_prior

    return put_eawp(
        out,
        original_input,
        out_layer=out_layer,
        weights_layer=weights_layer,
        uns_key=key,
    )


def voom_with_quality_weights(
    counts,
    design: np.ndarray | None = None,
    lib_size: np.ndarray | None = None,
    normalize_method: str = "none",
    plot: bool = False,
    span: float = 0.5,
    adaptive_span: bool = True,
    var_design: np.ndarray | None = None,
    var_group: np.ndarray | None = None,
    method: str = "genebygene",
    maxiter: int = 50,
    tol: float = 1e-5,
    trace: bool = False,
    col=None,
    *,
    out_layer: str = "voom_E",
    weights_layer: str = "voom_weights",
    key: str = "voom",
    layer: str | None = None,
    **voom_kwargs,
):
    """
    voom transformation with sample-specific quality weights.

    Combines the voom mean-variance modelling with sample-specific quality
    weights estimated by array_weights(). This can improve power when some
    samples have higher technical variability than others.

    Parameters
    ----------
    counts : ndarray
        Matrix of counts, shape (n_genes, n_samples).
    design : ndarray, optional
        Design matrix. If None, uses an intercept-only model.
    lib_size : ndarray, optional
        Library sizes. If None, computed as column sums.
    normalize_method : str, default "none"
        Normalization method (passed to voom).
    var_design : ndarray, optional
        Design matrix for variance model.
    var_group : ndarray, optional
        Factor defining variance groups.
    method : str, default "genebygene"
        Method for array weights estimation.
    maxiter : int, default 50
        Maximum iterations for array weights.
    tol : float, default 1e-5
        Convergence tolerance for array weights.
    trace : bool, default False
        If True, print iteration progress to stdout during the second
        array_weights() estimation (matches R behaviour).
    span : float, default 0.5
        LOWESS span (used if adaptive_span=False).
    adaptive_span : bool, default True
        If True, choose span adaptively.

    Returns
    -------
    dict
        Same keys as :func:`voom`, with one addition:

        sample_weights : ndarray
            Per-sample quality weights.

    Notes
    -----
    The R `col` argument (bar colour for the array-weight plot) is not
    exposed; matplotlib defaults are used when ``plot=True``.

    See Also
    --------
    voom : Basic voom transformation.
    array_weights : Estimate sample quality weights.
    """
    # Import here to avoid circular import
    from .weights import array_weights

    # Extract raw counts once so internal voom calls work on plain ndarray and
    # the final put_eawp packages the result based on the ORIGINAL input class.
    original_input = counts
    eawp = get_eawp(counts, layer=layer)
    counts_arr = np.asarray(eawp["exprs"], dtype=np.float64)
    # EList design warning: R's voomWithQualityWeights passes the EList
    # into its internal voom (voomWithQualityWeights.R:13,19) which then
    # hits as.matrix(EList) and drops all slots. pylimma picks design up
    # here before the inner voom sees an ndarray, so we warn for parity
    # with the R behaviour. Weights are not handled here (neither side
    # uses y$weights in voomWithQualityWeights).
    if design is None and eawp.get("design") is not None:
        if isinstance(original_input, EList):
            warnings.warn(
                "pylimma's voom_with_quality_weights is using 'design' "
                "from the EList's design slot. R's "
                "voomWithQualityWeights would silently discard it via "
                "as.matrix(EList). To match R behaviour, pass "
                "design=el['design'] explicitly.",
                UserWarning,
            )
        design = eawp["design"]

    # Parse design once, up front. Inner voom calls are fed counts_arr
    # (plain ndarray), so formula strings must be resolved here -
    # otherwise the inner calls lose the adata.obs context needed by patsy.
    sample_data = original_input.obs if _is_anndata(original_input) else None
    n_samples = counts_arr.shape[1]
    design, _ = _parse_design(design, data=sample_data, n_samples=n_samples)

    # Initial voom without array weights
    v = voom(
        counts_arr,
        design=design,
        lib_size=lib_size,
        normalize_method=normalize_method,
        span=span,
        adaptive_span=adaptive_span,
        **voom_kwargs,
    )

    # Estimate array weights
    aw = array_weights(
        v,
        design=design,
        var_design=var_design,
        var_group=var_group,
        method=method,
        maxiter=maxiter,
        tol=tol,
    )

    # Re-run voom with array weights, drawing the trend if plot=True
    v = voom(
        counts_arr,
        design=design,
        weights=aw,
        lib_size=lib_size,
        normalize_method=normalize_method,
        span=span,
        adaptive_span=adaptive_span,
        plot=plot,
        **voom_kwargs,
    )

    # Update array weights with new voom output (matches R: trace only here)
    aw = array_weights(
        v,
        design=design,
        var_design=var_design,
        var_group=var_group,
        method=method,
        maxiter=maxiter,
        tol=tol,
        trace=trace,
    )

    # Incorporate array weights into voom weights
    # v$weights <- t(aw * t(v$weights))
    v["weights"] = v["weights"] * aw[np.newaxis, :]
    v["sample_weights"] = aw

    if plot:
        _draw_array_weights_bar(aw, col=col)

    return put_eawp(
        dict(v),
        original_input,
        out_layer=out_layer,
        weights_layer=weights_layer,
        uns_key=key,
    )


def vooma(
    y,
    design: np.ndarray | None = None,
    block: np.ndarray | None = None,
    correlation: float | None = None,
    predictor: np.ndarray | None = None,
    span: float | None = None,
    legacy_span: bool = False,
    plot: bool = False,
    save_plot: bool = False,
    *,
    out_layer: str = "vooma_E",
    weights_layer: str = "vooma_weights",
    key: str = "vooma",
    layer: str | None = None,
):
    """
    voom-like weights for non-count expression data.

    Similar to voom but for continuous log-expression data (e.g., microarray).
    Estimates the mean-variance relationship and computes observation weights.

    Parameters
    ----------
    y : ndarray
        Expression matrix (log-scale), shape (n_genes, n_samples).
    design : ndarray, optional
        Design matrix. If None, uses an intercept-only model.
    block : ndarray, optional
        Factor indicating blocking structure.
    correlation : float, optional
        Intra-block correlation (required if block is specified).
    predictor : ndarray, optional
        Precision predictor, shape (n_genes,) or (n_genes, n_samples). When
        given, the variance trend is fitted against a linear combination of
        average log-expression and the row-mean predictor, and sample-specific
        weights are derived from the predictor.
    span : float, optional
        LOWESS span. If None, chosen adaptively.
    legacy_span : bool, default False
        If True, use the legacy adaptive-span rule
        (small_n=10, power=0.5); otherwise use small_n=50, power=1/3.
        Ignored if `span` is given.
    save_plot : bool, default False
        If True, include trend data in output.

    Returns
    -------
    dict
        E : ndarray
            Expression matrix (same as input y).
        weights : ndarray
            Precision weights, shape (n_genes, n_samples).
        design : ndarray
            Design matrix.
        span : float
            LOWESS span used.
        voom_xy : dict, optional
            Trend data (only if save_plot=True).
        voom_line : dict, optional
            LOWESS fit (only if save_plot=True).
    """
    # Polymorphic input dispatch (R: vooma.EList and .default branches).
    original_input = y
    eawp = get_eawp(y, layer=layer)
    y = np.asarray(eawp["exprs"], dtype=np.float64)
    if design is None and eawp.get("design") is not None:
        design = eawp["design"]
    n_genes, n_samples = y.shape

    # Parse design (formula strings via patsy + adata.obs when AnnData).
    sample_data = original_input.obs if _is_anndata(original_input) else None
    design, _ = _parse_design(design, data=sample_data, n_samples=n_samples)

    # Compute row means
    A = np.nanmean(y, axis=1)
    if np.any(np.isnan(A)):
        raise ValueError("y contains entirely NA rows")

    # Validate predictor (matches R's shape + NA handling)
    if predictor is not None:
        predictor = np.asarray(predictor, dtype=np.float64)
        if predictor.ndim == 1:
            if predictor.shape[0] != n_genes:
                raise ValueError("predictor is of wrong dimension")
            predictor = np.broadcast_to(
                predictor[:, np.newaxis], (n_genes, n_samples)
            ).copy()
        elif predictor.ndim == 2:
            if predictor.shape[0] != n_genes:
                raise ValueError("predictor is of wrong dimension")
            if predictor.shape[1] == 1:
                predictor = np.broadcast_to(
                    predictor, (n_genes, n_samples)
                ).copy()
            elif predictor.shape[1] != n_samples:
                raise ValueError("predictor is of wrong dimension")
        else:
            raise ValueError("predictor is of wrong dimension")
        if np.any(np.isnan(predictor)):
            y_has_na = np.any(np.isnan(y))
            if y_has_na:
                if np.any(np.isnan(predictor[~np.isnan(y)])):
                    raise ValueError(
                        "All observed y values must have non-NA predictors"
                    )
            else:
                raise ValueError(
                    "All observed y values must have non-NA predictors"
                )

    # Fit linear model
    if block is None:
        # Simple OLS fit
        from scipy import linalg
        q, r, pivot = linalg.qr(design, pivoting=True)
        rank = np.sum(np.abs(np.diag(r)) > 1e-10)
        qty = q.T @ y.T
        fitted = (q[:, :rank] @ qty[:rank, :]).T

        # Residual variance
        residual_effects = qty[rank:, :]
        s2 = np.mean(residual_effects ** 2, axis=0)
    else:
        # With block correlation
        block = np.asarray(block)
        if len(block) != n_samples:
            raise ValueError("Length of block does not match number of arrays")
        if correlation is None:
            raise ValueError(
                "correlation must be specified when block is provided. "
                "Use duplicate_correlation() to estimate intra-block correlation."
            )

        unique_blocks = np.unique(block)
        n_blocks = len(unique_blocks)

        # Build block indicator matrix Z
        Z = np.zeros((n_samples, n_blocks))
        for i, ub in enumerate(unique_blocks):
            Z[:, i] = (block == ub).astype(float)

        # Correlation matrix
        cormatrix = Z @ (correlation * Z.T)
        np.fill_diagonal(cormatrix, 1.0)

        # Cholesky decomposition
        from scipy import linalg
        chol_v = linalg.cholesky(cormatrix, lower=False)

        # Transform data and design
        z = linalg.solve_triangular(chol_v, y.T, trans="T")
        X = linalg.solve_triangular(chol_v, design, trans="T")

        # Fit
        q, r, pivot = linalg.qr(X, pivoting=True)
        rank = np.sum(np.abs(np.diag(r)) > 1e-10)
        qtz = q.T @ z
        fitted_transformed = q[:, :rank] @ qtz[:rank, :]
        fitted = (chol_v.T @ fitted_transformed).T

        # Residual variance
        residual_effects = qtz[rank:, :]
        s2 = np.mean(residual_effects ** 2, axis=0)

    # Prepare for trend fitting
    sx = A
    sy = np.sqrt(np.sqrt(s2))  # Quarter-root variance = sqrt(sqrt(s2))
    mu = fitted

    # Optionally combine ave log intensity with precision predictor.
    # R: vartrend <- lm.fit(cbind(1, sx, sxc), sy); sx <- vartrend$fitted.values;
    #    mu <- beta[1] + beta[2]*mu + beta[3]*predictor
    if predictor is not None:
        sxc = np.nanmean(predictor, axis=1)
        vartrend_design = np.column_stack([np.ones(n_genes), sx, sxc])
        beta, *_ = linalg.lstsq(vartrend_design, sy)
        sx = vartrend_design @ beta
        mu = beta[0] + beta[1] * mu + beta[2] * predictor
        xlab = "Combined predictor"
    else:
        xlab = "Average log-expression"

    # Choose span (legacy vs default adaptive rule)
    if span is None:
        if legacy_span:
            span = choose_lowess_span(n_genes, small_n=10, min_span=0.3, power=0.5)
        else:
            span = choose_lowess_span(n_genes, small_n=50, min_span=0.3, power=1/3)

    # Fit LOWESS trend using statsmodels (matches R's lowess closely)
    # delta=0.01*range(x) matches R's default delta parameter
    x_range = sx.max() - sx.min()
    lowess_result = sm_lowess(sy, sx, frac=span, it=3, delta=0.01 * x_range, return_sorted=True)
    x_sorted = lowess_result[:, 0]
    y_sorted = lowess_result[:, 1]

    f = interpolate.interp1d(
        x_sorted, y_sorted,
        kind="linear",
        bounds_error=False,
        fill_value=(y_sorted[0], y_sorted[-1])
    )

    # Compute weights from fitted values (mu)
    w = 1 / f(mu) ** 4

    # Build output
    out = {
        "E": y,
        "weights": w,
        "design": design,
        "span": span,
    }

    if plot:
        _draw_voom_trend(
            sx, sy, x_sorted, y_sorted,
            xlab=xlab,
            ylab="Sqrt( standard deviation )",
            title=("vooma variance trend"
                   if predictor is not None
                   else "vooma mean-variance trend"),
        )

    if save_plot:
        out["voom_xy"] = {
            "x": sx,
            "y": sy,
            "xlab": xlab,
            "ylab": "Sqrt( standard deviation )",
        }
        out["voom_line"] = {
            "x": x_sorted,
            "y": y_sorted,
        }

    return put_eawp(
        out,
        original_input,
        out_layer=out_layer,
        weights_layer=weights_layer,
        uns_key=key,
    )


def vooma_lm_fit(
    y,
    design=None,
    prior_weights: np.ndarray | None = None,
    block: np.ndarray | None = None,
    sample_weights: bool = False,
    var_design: np.ndarray | None = None,
    var_group: np.ndarray | None = None,
    prior_n: float = 10,
    predictor: np.ndarray | None = None,
    span: float | None = None,
    legacy_span: bool = False,
    plot: bool = False,
    save_plot: bool = False,
    keep_elist: bool = True,
    *,
    key: str = "pylimma",
    voom_key: str = "vooma",
    weights_layer: str = "vooma_weights",
    layer: str | None = None,
) -> dict | None:
    """
    Combined vooma + lmFit with iterative refinement.

    Applies vooma-style mean-variance modelling to expression data, then fits
    a linear model. If block correlation or sample weights are requested,
    performs one iteration to refine the estimates.

    Parameters
    ----------
    y : ndarray
        Expression matrix (log-scale), shape (n_genes, n_samples).
    design : ndarray, optional
        Design matrix. If None, uses an intercept-only model.
    prior_weights : ndarray, optional
        Prior observation weights. Cannot be combined with sample_weights.
    block : ndarray, optional
        Block factor for correlated samples.
    sample_weights : bool, default False
        If True, estimate sample-specific quality weights.
    var_design : ndarray, optional
        Design matrix for variance model (used with sample_weights).
    var_group : ndarray, optional
        Factor defining variance groups (used with sample_weights).
    prior_n : float, default 10
        Prior sample size for array weights estimation.
    span : float, optional
        LOWESS span. If None, chosen adaptively.
    legacy_span : bool, default False
        If True, use legacy span selection algorithm.
    save_plot : bool, default False
        If True, include trend data in output.
    keep_elist : bool, default True
        If True, include expression data with weights in output (dict
        return path only; AnnData input always writes the weights layer
        regardless).
    key : str, default "pylimma"
        AnnData ``adata.uns`` key for the fit slots (mirrors
        :func:`lm_fit`).
    voom_key : str, default "vooma"
        AnnData ``adata.uns`` key for ancillary metadata (span,
        sample_weights, voom_xy, voom_line). Keeps the fit and the
        voom-like ancillaries in separate uns buckets, matching the
        ``vooma()`` + ``lm_fit()`` split.
    weights_layer : str, default "vooma_weights"
        AnnData output layer for the computed observation weights.
    layer : str, optional
        AnnData input layer to read expression from. Defaults to
        ``adata.X``.

    Returns
    -------
    dict or None
        For AnnData input: mutates adata (weights layer, fit in
        ``adata.uns[key]``, ancillaries in ``adata.uns[voom_key]``) and
        returns ``None``.
        For ndarray / EList / dict input: returns a dict with the fit
        plus span, targets (if requested), EList (if keep_elist), and
        voom_xy / voom_line (if save_plot).

    Notes
    -----
    This function combines vooma() and lm_fit() with optional iterative
    refinement of sample weights and intra-block correlation.
    """
    from .lmfit import lm_fit
    from .weights import array_weights
    from .dups import duplicate_correlation
    from .classes import _as_matrix_weights

    # Polymorphic input dispatch (ndarray / dict / EList / AnnData),
    # mirroring voom / vooma. Pull design and prior_weights off the
    # input when the caller left them unset.
    original_input = y
    eawp = get_eawp(y, layer=layer)
    y = np.asarray(eawp["exprs"], dtype=np.float64)
    if design is None and eawp.get("design") is not None:
        design = eawp["design"]
    if prior_weights is None and eawp.get("weights") is not None:
        prior_weights = np.asarray(eawp["weights"], dtype=np.float64)
    n_genes, n_samples = y.shape

    if n_samples < 2:
        raise ValueError("Too few samples")
    if n_genes < 2:
        raise ValueError("Need multiple rows")

    # Compute row means
    A = np.nanmean(y, axis=1)
    if np.any(np.isnan(A)):
        raise ValueError("y contains entirely NA rows")

    # Validate predictor (mirrors vooma())
    if predictor is not None:
        predictor = np.asarray(predictor, dtype=np.float64)
        if predictor.ndim == 1:
            if predictor.shape[0] != n_genes:
                raise ValueError("predictor is of wrong dimension")
            predictor = np.broadcast_to(
                predictor[:, np.newaxis], (n_genes, n_samples)
            ).copy()
        elif predictor.ndim == 2:
            if predictor.shape[0] != n_genes:
                raise ValueError("predictor is of wrong dimension")
            if predictor.shape[1] == 1:
                predictor = np.broadcast_to(
                    predictor, (n_genes, n_samples)
                ).copy()
            elif predictor.shape[1] != n_samples:
                raise ValueError("predictor is of wrong dimension")
        else:
            raise ValueError("predictor is of wrong dimension")
        if np.any(np.isnan(predictor)):
            y_has_na = np.any(np.isnan(y))
            if y_has_na:
                if np.any(np.isnan(predictor[~np.isnan(y)])):
                    raise ValueError(
                        "All observed y values must have non-NA predictors"
                    )
            else:
                raise ValueError(
                    "All observed y values must have non-NA predictors"
                )

    # Parse design (formula strings via patsy + adata.obs when AnnData).
    sample_data = original_input.obs if _is_anndata(original_input) else None
    design, _ = _parse_design(design, data=sample_data, n_samples=n_samples)

    # Check for conflicting weight specifications
    use_sample_weights = sample_weights or var_design is not None or var_group is not None
    if prior_weights is not None and use_sample_weights:
        raise ValueError("Cannot specify prior_weights and estimate sample weights")

    use_block = block is not None

    # Initial fit
    fit = lm_fit(y, design, weights=prior_weights)

    # Compute fitted values
    if fit["rank"] < design.shape[1]:
        pivot = fit.get("pivot", np.arange(design.shape[1]))
        j = pivot[:fit["rank"]]
        fitted_values = fit["coefficients"][:, j] @ design[:, j].T
    else:
        fitted_values = fit["coefficients"] @ design.T

    # Prepare for trend fitting
    sx = A
    sy = np.sqrt(fit["sigma"])
    mu = fitted_values

    # Optionally combine ave log intensity with precision predictor
    sxc = None
    if predictor is not None:
        sxc = np.nanmean(predictor, axis=1)
        vartrend_design = np.column_stack([np.ones(n_genes), sx, sxc])
        beta, *_ = linalg.lstsq(vartrend_design, sy)
        sx = vartrend_design @ beta
        mu = beta[0] + beta[1] * mu + beta[2] * predictor

    # Choose span
    if span is None:
        if legacy_span:
            span = choose_lowess_span(n_genes, small_n=10, min_span=0.3, power=0.5)
        else:
            span = choose_lowess_span(n_genes, small_n=50, min_span=0.3, power=1/3)

    # Fit LOWESS trend
    x_range = sx.max() - sx.min()
    lowess_result = sm_lowess(sy, sx, frac=span, it=3, delta=0.01 * x_range, return_sorted=True)
    x_sorted = lowess_result[:, 0]
    y_sorted = lowess_result[:, 1]

    # Create interpolating function
    f = interpolate.interp1d(
        x_sorted, y_sorted,
        kind="linear",
        bounds_error=False,
        fill_value=(y_sorted[0], y_sorted[-1])
    )

    # Compute vooma weights
    w = 1 / f(mu) ** 4

    # Combine with prior weights if provided
    if prior_weights is not None:
        # Reshape prior_weights to (n_genes, n_samples) via asMatrixWeights
        # before multiplying so probe-weight / array-weight / scalar shapes
        # all behave like R's vooma path.
        from .classes import _as_matrix_weights
        weights = w * _as_matrix_weights(prior_weights, (n_genes, n_samples))
    else:
        weights = w

    # Estimate sample weights if requested
    sw = None
    if use_sample_weights:
        sw = array_weights(
            {"E": y, "weights": weights},
            design=design,
            var_design=var_design,
            var_group=var_group,
            prior_n=prior_n,
        )
        if use_block:
            # Apply sample weights to observation weights
            weights = weights * sw[np.newaxis, :]

    # Estimate block correlation if requested
    correlation = None
    if use_block:
        dc = duplicate_correlation(y, design, block=block, weights=weights)
        correlation = dc["consensus_correlation"]
        if np.isnan(correlation):
            correlation = 0.0

    # Second iteration if block or sample weights requested
    if use_block or use_sample_weights:
        # Reset weights for refit
        if use_sample_weights:
            weights = np.broadcast_to(sw, (n_genes, n_samples)).copy()
        else:
            weights = prior_weights

        # Refit with correlation
        fit = lm_fit(y, design, block=block, correlation=correlation, weights=weights)

        # Recompute fitted values
        if fit["rank"] < design.shape[1]:
            pivot = fit.get("pivot", np.arange(design.shape[1]))
            j = pivot[:fit["rank"]]
            fitted_values = fit["coefficients"][:, j] @ design[:, j].T
        else:
            fitted_values = fit["coefficients"] @ design.T

        # Refit LOWESS trend
        sx = A
        sy = np.sqrt(fit["sigma"])
        mu = fitted_values

        # Re-apply predictor combination using the cached sxc
        if predictor is not None:
            vartrend_design = np.column_stack([np.ones(n_genes), sx, sxc])
            beta, *_ = linalg.lstsq(vartrend_design, sy)
            sx = vartrend_design @ beta
            mu = beta[0] + beta[1] * mu + beta[2] * predictor

        x_range = sx.max() - sx.min()
        lowess_result = sm_lowess(sy, sx, frac=span, it=3, delta=0.01 * x_range, return_sorted=True)
        x_sorted = lowess_result[:, 0]
        y_sorted = lowess_result[:, 1]

        f = interpolate.interp1d(
            x_sorted, y_sorted,
            kind="linear",
            bounds_error=False,
            fill_value=(y_sorted[0], y_sorted[-1])
        )

        # Recompute vooma weights
        w = 1 / f(mu) ** 4

        # Combine with prior weights
        if prior_weights is not None:
            # Reshape prior_weights to (n_genes, n_samples) via
            # asMatrixWeights before multiplying so probe-weight /
            # array-weight / scalar shapes all behave like R's vooma.
            from .classes import _as_matrix_weights
            weights = w * _as_matrix_weights(prior_weights, (n_genes, n_samples))
        else:
            weights = w

        # Re-estimate sample weights
        if use_sample_weights:
            sw = array_weights(
                {"E": y, "weights": weights},
                design=design,
                var_design=var_design,
                var_group=var_group,
                prior_n=prior_n,
            )
            weights = weights * sw[np.newaxis, :]

        # Re-estimate block correlation
        if use_block:
            dc = duplicate_correlation(y, design, block=block, weights=weights)
            correlation = dc["consensus_correlation"]
            if np.isnan(correlation):
                correlation = 0.0

    # Final fit
    fit = lm_fit(y, design, block=block, correlation=correlation, weights=weights)

    # Add span to output
    fit["span"] = span

    # Add sample weights to targets
    if use_sample_weights:
        fit["targets"] = {"sample_weights": sw}

    # Render trend plot if requested
    if plot:
        _draw_voom_trend(
            sx, sy, x_sorted, y_sorted,
            xlab=("Combined predictor" if predictor is not None
                  else "Average log2 expression"),
            ylab="Sqrt( standard deviation )",
            title=("vooma variance trend" if predictor is not None
                   else "vooma mean-variance trend"),
        )

    # Add plot data if requested
    if save_plot:
        fit["voom_xy"] = {
            "x": sx,
            "y": sy,
            "xlab": "Average log-expression",
            "ylab": "Sqrt( standard deviation )",
        }
        fit["voom_line"] = {
            "x": x_sorted,
            "y": y_sorted,
        }

    # Add EList if requested
    if keep_elist:
        fit["EList"] = {
            "E": y,
            "weights": weights,
        }

    # Polymorphic output dispatch. AnnData input splits the bundled
    # output across (layer, fit-uns, vooma-uns); ndarray / EList / dict
    # callers get the full single-dict return as today.
    if _is_anndata(original_input):
        adata = original_input
        # Observation weights layer (limma (n_genes, n_samples) -> AnnData
        # (n_samples, n_genes) via _as_matrix_weights normalisation + .T).
        W = _as_matrix_weights(weights, (n_genes, n_samples))
        adata.layers[weights_layer] = W.T
        # Fit slots under the lm_fit uns key (plain dict for h5ad compat).
        bundled = {"EList", "voom_xy", "voom_line", "targets", "span"}
        fit_slots = {k: v for k, v in fit.items() if k not in bundled}
        adata.uns[key] = dict(fit_slots)
        # Ancillary metadata under the vooma uns key. Mirrors what
        # vooma() writes, plus sample_weights when estimated.
        ancillary = {}
        if fit.get("span") is not None:
            ancillary["span"] = fit["span"]
        if use_sample_weights and isinstance(fit.get("targets"), dict):
            ancillary.update(fit["targets"])   # {"sample_weights": sw}
        if save_plot:
            if "voom_xy" in fit:
                ancillary["voom_xy"] = fit["voom_xy"]
            if "voom_line" in fit:
                ancillary["voom_line"] = fit["voom_line"]
        if ancillary:
            adata.uns[voom_key] = ancillary
        return None

    return fit


def vooma_by_group(
    y,
    group,
    design=None,
    block=None,
    correlation=None,
    span=None,
    legacy_span: bool = False,
    plot: bool = False,
    *,
    out_layer: str = "vooma_E",
    weights_layer: str = "vooma_weights",
    uns_key: str = "vooma",
    layer: str | None = None,
):
    """
    Vooma with group-specific mean-variance trends.

    Port of R limma's ``voomaByGroup`` (``vooma.R``). Fits one vooma
    trend per ``group`` level and stitches the observation weights
    back together. Plotting is not emitted; ``plot=True`` is accepted
    for API compatibility but ignored.

    Output
    ------
    Polymorphic, matching :func:`vooma`:

    - AnnData in: writes ``adata.layers[out_layer]`` (copy of E),
      ``adata.layers[weights_layer]`` (computed weights), and
      ``adata.uns[uns_key]`` (design + group). Returns ``None``.
    - EList in: returns a new ``EList`` with updated slots.
    - ndarray / dict in: returns a dict with ``E``, ``weights``,
      ``design``, ``group``.

    The default ``uns_key="vooma"`` means ``lm_fit(adata, layer="vooma_E")``
    picks up the design via the usual :func:`get_eawp` fallback.
    """
    import pandas as pd
    from .classes import EList, get_eawp, put_eawp

    original_input = y
    eawp = get_eawp(y, layer=layer)
    E = np.asarray(eawp["exprs"], dtype=np.float64)
    if design is None:
        design = eawp.get("design")
    ngenes, narrays = E.shape

    # plot=True is accepted for R-signature compatibility but pylimma
    # does not yet emit the per-group mean-variance figure. Warn so the
    # caller knows their request was silently ignored (R's voomaByGroup
    # at vooma.R:116-118 would draw it).
    if plot:
        warnings.warn(
            "vooma_by_group(plot=True) is not implemented in pylimma; "
            "no plot will be drawn. Pass plot=False to silence this "
            "warning.",
            UserWarning,
        )

    group_arr = np.asarray(pd.Categorical(group))
    levels = list(pd.Categorical(group).categories)
    ngroups = len(levels)
    if group_arr.size != narrays:
        raise ValueError("length(group) must equal ncol(y)")

    if design is None:
        # ~0 + group one-hot
        design = np.zeros((narrays, ngroups))
        for j, lev in enumerate(levels):
            design[group_arr == lev, j] = 1.0
    design = np.asarray(design, dtype=np.float64)

    weights = np.empty_like(E)
    for lev in levels:
        mask = group_arr == lev
        if mask.sum() < 2:
            # Singleton level: fall back to a global vooma fit.
            v_all = vooma({"E": E}, design=design, block=block,
                          correlation=correlation, span=span,
                          legacy_span=legacy_span, plot=False)
            weights[:, mask] = v_all["weights"][:, mask]
            continue
        sub = {"E": E[:, mask]}
        v = vooma(sub, design=np.ones((int(mask.sum()), 1)),
                  span=span, legacy_span=legacy_span, plot=False)
        weights[:, mask] = v["weights"]

    return put_eawp(
        {
            "E": E,
            "weights": weights,
            "design": design,
            "group": group_arr,
        },
        original_input,
        out_layer=out_layer,
        weights_layer=weights_layer,
        uns_key=uns_key,
    )

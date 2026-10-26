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
import pandas as pd
from scipy import linalg

from .classes import EList, _eawp_design, _is_anndata, _shape_error, get_eawp, put_eawp
from .lmfit import (
    _design_names,
    _fit_genes,
    _numeric_design,
    _parse_design,
    _qr_r_style,
    lm_fit,
)
from .utils import _block_labels, _r_approx_rule2, _r_factor, _r_lowess, choose_lowess_span


def _voom_design(design, sample_data, n_samples: int) -> tuple[np.ndarray, list[str] | None]:
    """The design of the voom family, parsed as lm_fit's; R
    names the default intercept-only design "GrandMean"."""
    if design is None:
        return np.ones((n_samples, 1)), ["GrandMean"]
    return _parse_design(design, data=sample_data, n_samples=n_samples)


def _named_design(design: np.ndarray, names: list[str] | None):
    """Output form of a parsed design: a DataFrame when column names are
    known (R keeps colnames(design) on v$design, so lm_fit(v) recovers
    coefficient names), otherwise the bare matrix. Either is a copy, so the
    output shares no memory with the caller's design."""
    if names is None:
        return design.copy()
    return pd.DataFrame(design, columns=names, copy=True)


def _fitted_values(fit, design: np.ndarray) -> np.ndarray:
    """Fitted values of a linear-model fit, genes x samples: R's
    coefficients %*% t(design), over the first ``rank`` pivoted columns
    when the design is rank deficient (voom.R, voomaLmFit.R)."""
    rank = fit["rank"]
    if rank < design.shape[1]:
        j = fit["pivot"][:rank]
        return fit["coefficients"][:, j] @ design[:, j].T
    return fit["coefficients"] @ design.T


def _vooma_trend_weights(A, sy, mu, predictor, span):
    """vooma's mean-variance trend and weights (vooma.R, voomaLmFit.R).

    The trend of ``sy`` (square-root residual standard deviations) is
    fitted against the average log-expression ``A``, or with a precision
    predictor against R's combined predictor
    (``lm.fit(cbind(1, sx, sxc), sy)``, with ``mu`` mapped the same way).
    A gene with no residual df has no variance (NaN) and is left out of the
    trend; R's lowess would compute with its NA (known_differences.rst).
    Returns ``(sx, trend_x, trend_y, weights)``.
    """
    in_trend = np.isfinite(sy)
    sx = A
    if predictor is not None:
        vartrend_design = np.column_stack([np.ones(A.size), A, np.nanmean(predictor, axis=1)])
        beta, *_ = linalg.lstsq(vartrend_design[in_trend], sy[in_trend])
        sx = vartrend_design @ beta
        mu = beta[0] + beta[1] * mu + beta[2] * predictor
    # R: l <- lowess(sx, sy, f=span); approxfun(l, rule=2, ties=list("ordered", mean))
    trend_x, trend_y = _r_lowess(sx[in_trend], sy[in_trend], f=span)
    return sx, trend_x, trend_y, 1 / _r_approx_rule2(trend_x, trend_y, mu) ** 4


def _validate_predictor(predictor, y: np.ndarray) -> np.ndarray:
    """Check vooma's precision predictor against y (genes x samples) and
    expand it to a full matrix, matching R's shape and NA handling."""
    n_genes, n_samples = y.shape
    predictor = np.asarray(predictor, dtype=np.float64)
    if predictor.ndim == 1:
        if predictor.shape[0] != n_genes:
            raise ValueError("predictor is of wrong dimension")
        predictor = np.broadcast_to(predictor[:, np.newaxis], (n_genes, n_samples)).copy()
    elif predictor.ndim == 2:
        if predictor.shape[0] != n_genes:
            raise _shape_error("predictor", predictor.shape, y.shape, "predictor is of wrong dimension")
        if predictor.shape[1] == 1:
            predictor = np.broadcast_to(predictor, (n_genes, n_samples)).copy()
        elif predictor.shape[1] != n_samples:
            raise ValueError("predictor is of wrong dimension")
    else:
        raise ValueError("predictor is of wrong dimension")
    if np.any(np.isnan(predictor)):
        y_has_na = np.any(np.isnan(y))
        if y_has_na:
            if np.any(np.isnan(predictor[~np.isnan(y)])):
                raise ValueError("All observed y values must have non-NA predictors")
        else:
            raise ValueError("All observed y values must have non-NA predictors")
    return predictor


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

    matplotlib is an optional dependency, imported here as by the other
    plotting functions.
    """
    from .plotting import _require_matplotlib

    plt = _require_matplotlib()
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
    from .plotting import _require_matplotlib

    plt = _require_matplotlib()
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
        Library sizes for each sample (or one value for all; another
        length raises, where R recycles it). If None, computed as column
        sums of counts.
    offset : ndarray, optional
        Offset matrix, shape (n_genes, n_samples). If provided without
        offset_prior, offset_prior is computed as offset - rowMeans(offset).
        Even for AnnData input a 2-D
        matrix is genes x samples (the transpose of ``adata.layers``).
    offset_prior : ndarray, optional
        Pre-centered offset matrix, shape (n_genes, n_samples). Applied as:
        lib_size_matrix = exp(log(lib_size_matrix) + offset_prior).
        Takes precedence over offset if both are provided.
        Even for AnnData input a 2-D
        matrix is genes x samples (the transpose of ``adata.layers``).
    normalize_method : str, default "none"
        Normalization method. Currently only "none" is supported.
    block : ndarray, optional
        Factor indicating blocking structure for samples.
    correlation : float, optional
        Intra-block correlation (required if block is specified).
    weights : ndarray, optional
        Prior weights for samples or observations. Even for AnnData input a 2-D
        matrix is genes x samples (the transpose of ``adata.layers``).
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
    **AnnData views.** Results are written into the AnnData. If it is a
    view (e.g. ``adata[:, mask]``), anndata first turns it into a
    standalone copy, with an ``ImplicitModificationWarning``, and the
    results land on that copy rather than on the parent. Assign the view
    to a variable first (``sub = adata[:, mask]``) or pass
    ``adata[:, mask].copy()``; a bare ``fn(adata[:, mask])`` call
    discards the results.

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
    # See ``docs/validation/known_differences.rst``.
    _input_is_elist = isinstance(original_input, EList)
    if design is None:
        design = _eawp_design(eawp)
        if design is not None and _input_is_elist:
            warnings.warn(
                "pylimma's voom is using 'design' from the EList's design "
                "slot. R's voom would silently discard it via "
                "as.matrix(EList). To match R behaviour, pass "
                "design=el['design'] explicitly (or clear the slot).",
                UserWarning,
            )
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
    design, design_names = _voom_design(design, sample_data, n_samples)

    # Check lib_size
    if lib_size is None:
        lib_size = np.sum(counts, axis=0)
    # a copy: lib_size is returned (R's copy-on-modify)
    lib_size = np.array(lib_size, dtype=np.float64)
    # R's matrix(lib.size, byrow=TRUE) recycles any other length
    if lib_size.size not in (1, n_samples):
        raise ValueError(
            f"lib_size must have one value per sample ({n_samples}) or a single value, "
            f"not {lib_size.size}"
        )

    lib_size_matrix = np.broadcast_to(lib_size.reshape(-1), (n_genes, n_samples)).copy()

    # Handle offset parameters
    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64)
        if offset.shape != counts.shape:
            raise _shape_error("offset", offset.shape, counts.shape, "counts and offset must have equal dimensions")
        if offset_prior is None:
            offset_prior = offset - np.mean(offset, axis=1, keepdims=True)
        else:
            warnings.warn("Ignoring offset in favor of offset_prior. Should not set both.")
            offset = None

    if offset_prior is not None:
        offset_prior = np.asarray(offset_prior, dtype=np.float64)
        if offset_prior.shape != counts.shape:
            raise _shape_error(
                "offset_prior", offset_prior.shape, counts.shape, "counts and offset_prior must have equal dimensions"
            )
        lib_size_matrix = np.exp(np.log(lib_size_matrix) + offset_prior)

    # Choose span based on number of genes
    if adaptive_span:
        span = choose_lowess_span(n_genes, small_n=50, min_span=0.3, power=1 / 3)

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
            warnings.warn("The experimental design has no replication. Setting weights to 1.")
        elif n_with_reps == 1:
            warnings.warn("Only one gene with any replication. Setting weights to 1.")
        return put_eawp(
            {
                "E": y,
                "weights": np.ones_like(y),
                "design": _named_design(design, design_names),
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

    # R: l <- lowess(sx, sy, f=span)
    x_sorted, y_sorted = _r_lowess(sx_fit, sy_fit, f=span)

    # Compute fitted values from linear model
    fitted_values = _fitted_values(fit, design)

    # Convert to fitted counts
    # fitted.cpm = 2^fitted.values
    # fitted.count = 1e-6 * fitted.cpm * (lib.size.matrix+1)
    # fitted.logcount = log2(fitted.count)
    fitted_cpm = 2**fitted_values
    fitted_count = 1e-6 * fitted_cpm * (lib_size_matrix + 1)
    fitted_logcount = np.log2(fitted_count)

    # Apply trend to individual observations
    # w = 1/f(fitted.logcount)^4
    # R: f <- approxfun(l, rule=2, ties=list("ordered", mean))
    trend_values = _r_approx_rule2(x_sorted, y_sorted, fitted_logcount)
    w = 1 / trend_values**4

    # Build output
    out = {
        "E": y,
        "weights": w,
        "design": _named_design(design, design_names),
        "lib_size": lib_size,
    }

    if adaptive_span:
        out["span"] = span

    if plot:
        _draw_voom_trend(
            sx_fit,
            sy_fit,
            x_sorted,
            y_sorted,
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

    The sample weights are passed to the second voom as array weights. R
    passes them as a vector, which it reads as gene weights when the
    number of genes equals the number of samples, so results differ from
    R in that case only (docs/validation/known_differences.rst).

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
    **AnnData views.** Results are written into the AnnData. If it is a
    view (e.g. ``adata[:, mask]``), anndata first turns it into a
    standalone copy, with an ``ImplicitModificationWarning``, and the
    results land on that copy rather than on the parent. Assign the view
    to a variable first (``sub = adata[:, mask]``) or pass
    ``adata[:, mask].copy()``; a bare ``fn(adata[:, mask])`` call
    discards the results.

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
    if design is None:
        design = _eawp_design(eawp)
        if design is not None and isinstance(original_input, EList):
            warnings.warn(
                "pylimma's voom_with_quality_weights is using 'design' "
                "from the EList's design slot. R's "
                "voomWithQualityWeights would silently discard it via "
                "as.matrix(EList). To match R behaviour, pass "
                "design=el['design'] explicitly.",
                UserWarning,
            )

    # Parse design once, up front. Inner voom calls are fed counts_arr
    # (plain ndarray), so formula strings must be resolved here -
    # otherwise the inner calls lose the adata.obs context needed by patsy.
    sample_data = original_input.obs if _is_anndata(original_input) else None
    n_samples = counts_arr.shape[1]
    design, design_names = _voom_design(design, sample_data, n_samples)

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

    # Re-run voom with array weights, drawing the trend if plot=True. R
    # passes aw as a vector, which asMatrixWeights reads as gene weights
    # when n_genes == n_samples; a 1 x N row matrix is always array weights
    # (deliberate difference, docs/validation/known_differences.rst).
    v = voom(
        counts_arr,
        design=design,
        weights=aw.reshape(1, -1),
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
    v["design"] = _named_design(design, design_names)

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
        Expression matrix (log-scale), shape (n_genes, n_samples). May
        contain missing values (NaN), but no entirely missing rows and no
        infinite values. Each gene is then fitted on its observed values,
        as R's voomaLmFit does; a gene with no residual degrees of freedom
        is left out of the variance trend, and a gene with a coefficient
        that cannot be estimated gets NaN weights. R's vooma fails on
        missing values (see docs/validation/known_differences.rst).
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
        weights are derived from the predictor. Even for AnnData input a 2-D
        matrix is genes x samples (the transpose of ``adata.layers``).
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

    Notes
    -----
    **AnnData views.** Results are written into the AnnData. If it is a
    view (e.g. ``adata[:, mask]``), anndata first turns it into a
    standalone copy, with an ``ImplicitModificationWarning``, and the
    results land on that copy rather than on the parent. Assign the view
    to a variable first (``sub = adata[:, mask]``) or pass
    ``adata[:, mask].copy()``; a bare ``fn(adata[:, mask])`` call
    discards the results.
    """
    # Polymorphic input dispatch (R: vooma.EList and .default branches).
    original_input = y
    eawp = get_eawp(y, layer=layer)
    y = np.asarray(eawp["exprs"], dtype=np.float64)
    if design is None:
        design = _eawp_design(eawp)
    n_genes, n_samples = y.shape

    # Parse design (formula strings via patsy + adata.obs when AnnData).
    sample_data = original_input.obs if _is_anndata(original_input) else None
    design, design_names = _voom_design(design, sample_data, n_samples)

    # Compute row means
    A = np.nanmean(y, axis=1)
    if np.any(np.isnan(A)):
        raise ValueError("y contains entirely NA rows")
    n_infinite = int(np.isinf(y).sum())
    if n_infinite:
        raise ValueError(f"y contains {n_infinite} infinite values; vooma needs finite or missing values")

    # Validate predictor (matches R's shape + NA handling)
    if predictor is not None:
        predictor = _validate_predictor(predictor, y)

    # Fit linear model
    if block is not None:
        block = _block_labels(block, n_samples)
        if correlation is None:
            raise ValueError(
                "correlation must be specified when block is provided. "
                "Use duplicate_correlation() to estimate intra-block correlation."
            )
    if np.isnan(y).any():
        # limma intends vooma to allow missing values (NEWS 3.64.0), but R's
        # lm.fit fails on them; as voomaLmFit does, fit each gene on its
        # observed values with lm_fit (known_differences.rst)
        from .lmfit import lm_fit

        fit = lm_fit(y, design, block=block, correlation=correlation)
        fitted = _fitted_values(fit, design)
        s2 = fit["sigma"] ** 2
    elif block is None:
        # Simple OLS fit: R's lm.fit, whose QR is qr() (LINPACK dqrdc2,
        # rank tolerance 1e-7 relative to the column norms)
        q, _r, _pivot, rank = _qr_r_style(design)
        qty = q.T @ y.T
        fitted = (q[:, :rank] @ qty[:rank, :]).T

        # Residual variance
        residual_effects = qty[rank:, :]
        s2 = np.mean(residual_effects**2, axis=0)
    else:
        # With block correlation
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
        chol_v = linalg.cholesky(cormatrix, lower=False)

        # Transform data and design
        z = linalg.solve_triangular(chol_v, y.T, trans="T")
        X = linalg.solve_triangular(chol_v, design, trans="T")

        # Fit (R's lm.fit on the transformed design)
        q, _r, _pivot, rank = _qr_r_style(X)
        qtz = q.T @ z
        fitted_transformed = q[:, :rank] @ qtz[:rank, :]
        fitted = (chol_v.T @ fitted_transformed).T

        # Residual variance
        residual_effects = qtz[rank:, :]
        s2 = np.mean(residual_effects**2, axis=0)

    # Choose span (legacy vs default adaptive rule)
    if span is None:
        if legacy_span:
            span = choose_lowess_span(n_genes, small_n=10, min_span=0.3, power=0.5)
        else:
            span = choose_lowess_span(n_genes, small_n=50, min_span=0.3, power=1 / 3)

    sy = np.sqrt(np.sqrt(s2))  # Quarter-root variance = sqrt(sqrt(s2))
    sx, x_sorted, y_sorted, w = _vooma_trend_weights(A, sy, fitted, predictor, span)
    xlab = "Combined predictor" if predictor is not None else "Average log-expression"

    # Build output
    out = {
        "E": y,
        "weights": w,
        "design": _named_design(design, design_names),
        "span": span,
    }

    if plot:
        _draw_voom_trend(
            sx,
            sy,
            x_sorted,
            y_sorted,
            xlab=xlab,
            ylab="Sqrt( standard deviation )",
            title=(
                "vooma variance trend" if predictor is not None else "vooma mean-variance trend"
            ),
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
        Even for AnnData input a 2-D
        matrix is genes x samples (the transpose of ``adata.layers``).
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
        plus span, targets (with sample weights: the input's targets, or a
        new DataFrame, with a ``sample_weight`` column, as R), EList (if
        keep_elist), and voom_xy / voom_line (if save_plot).

    Notes
    -----
    **AnnData views.** Results are written into the AnnData. If it is a
    view (e.g. ``adata[:, mask]``), anndata first turns it into a
    standalone copy, with an ``ImplicitModificationWarning``, and the
    results land on that copy rather than on the parent. Assign the view
    to a variable first (``sub = adata[:, mask]``) or pass
    ``adata[:, mask].copy()``; a bare ``fn(adata[:, mask])`` call
    discards the results.

    This function combines vooma() and lm_fit() with optional iterative
    refinement of sample weights and intra-block correlation.
    """
    from .classes import as_matrix_weights
    from .dups import duplicate_correlation
    from .lmfit import lm_fit
    from .weights import array_weights

    # Polymorphic input dispatch (ndarray / dict / EList / AnnData),
    # mirroring voom / vooma. Pull design and prior_weights off the
    # input when the caller left them unset.
    original_input = y
    eawp = get_eawp(y, layer=layer)
    y = np.asarray(eawp["exprs"], dtype=np.float64)
    if design is None:
        design = _eawp_design(eawp)
    if prior_weights is None and eawp.get("weights") is not None:
        # R's voomaLmFit uses an EList weights slot only in its first lmFit
        # (lmFit falls back to y$weights); pylimma uses it as prior weights
        # throughout (docs/validation/known_differences.rst).
        if sample_weights or var_design is not None or var_group is not None:
            raise ValueError(
                "Cannot estimate sample weights when the input carries observation "
                "weights (an EList weights slot or AnnData weights layer), which "
                "vooma_lm_fit uses as prior_weights; remove them to estimate sample "
                "weights."
            )
        # An EList, or the plain dict voom / vooma return for matrix input
        if isinstance(original_input, dict):
            warnings.warn(
                "pylimma's vooma_lm_fit is using 'weights' from the EList's weights "
                "slot as prior_weights. R's voomaLmFit uses the slot only in its "
                "first internal lmFit, so results differ from R; see "
                "docs/validation/known_differences.rst.",
                UserWarning,
            )
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
        predictor = _validate_predictor(predictor, y)

    # Parse design (formula strings via patsy + adata.obs when AnnData).
    sample_data = original_input.obs if _is_anndata(original_input) else None
    design, design_names = _voom_design(design, sample_data, n_samples)

    # Check for conflicting weight specifications
    use_sample_weights = sample_weights or var_design is not None or var_group is not None
    if prior_weights is not None and use_sample_weights:
        raise ValueError("Cannot specify prior_weights and estimate sample weights")

    use_block = block is not None

    # Initial fit
    fit = lm_fit(y, design, weights=prior_weights)

    # Choose span
    if span is None:
        if legacy_span:
            span = choose_lowess_span(n_genes, small_n=10, min_span=0.3, power=0.5)
        else:
            span = choose_lowess_span(n_genes, small_n=50, min_span=0.3, power=1 / 3)

    # vooma's trend and weights
    sy = np.sqrt(fit["sigma"])
    sx, x_sorted, y_sorted, w = _vooma_trend_weights(A, sy, _fitted_values(fit, design), predictor, span)

    # Combine with prior weights if provided
    if prior_weights is not None:
        # Reshape prior_weights to (n_genes, n_samples) via asMatrixWeights
        # before multiplying so probe-weight / array-weight / scalar shapes
        # all behave like R's vooma path.
        from .classes import as_matrix_weights

        weights = w * as_matrix_weights(prior_weights, (n_genes, n_samples))
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

        # Refit the trend and weights
        sy = np.sqrt(fit["sigma"])
        sx, x_sorted, y_sorted, w = _vooma_trend_weights(A, sy, _fitted_values(fit, design), predictor, span)

        # Combine with prior weights
        if prior_weights is not None:
            # Reshape prior_weights to (n_genes, n_samples) via
            # asMatrixWeights before multiplying so probe-weight /
            # array-weight / scalar shapes all behave like R's vooma.
            from .classes import as_matrix_weights

            weights = w * as_matrix_weights(prior_weights, (n_genes, n_samples))
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
    # R's final lmFit runs on the EList, so fit$genes and colnames(design)
    # carry through (voomaLmFit.R:197-201); y here is a bare matrix.
    fit["genes"] = _fit_genes(original_input, eawp)
    if design_names is not None:
        fit["coef_names"] = design_names

    # Add span to output
    fit["span"] = span

    # R: fit$targets <- y$targets with a sample.weight column, else a data
    # frame of the sample weights named by the samples
    if use_sample_weights:
        targets = eawp.get("targets")
        if isinstance(targets, dict):
            # e.g. voom's {"lib_size": ...}
            targets = pd.DataFrame(targets)
        if isinstance(targets, pd.DataFrame):
            targets = targets.assign(sample_weight=sw)
        else:
            # R: colnames(y), of a DataFrame or of an EList's E (the sample
            # columns: get_eawp drops a leading gene-ID column)
            source = original_input.get("E") if isinstance(original_input, dict) else original_input
            sample_names = (
                source.columns[source.shape[1] - n_samples :]
                if isinstance(source, pd.DataFrame)
                else None
            )
            targets = pd.DataFrame({"sample_weight": sw}, index=sample_names)
        fit["targets"] = targets

    xlab = "Combined predictor" if predictor is not None else "Average log2 expression"
    # Render trend plot if requested
    if plot:
        _draw_voom_trend(
            sx,
            sy,
            x_sorted,
            y_sorted,
            xlab=xlab,
            ylab="Sqrt( standard deviation )",
            title=(
                "vooma variance trend" if predictor is not None else "vooma mean-variance trend"
            ),
        )

    # Add plot data if requested
    if save_plot:
        fit["voom_xy"] = {
            "x": sx,
            "y": sy,
            "xlab": xlab,
            "ylab": "Sqrt( standard deviation )",
        }
        fit["voom_line"] = {
            "x": x_sorted,
            "y": y_sorted,
        }

    # Add EList if requested
    if keep_elist:
        fit["EList"] = {
            "E": y.copy(),
            "weights": weights,
        }

    # Polymorphic output dispatch. AnnData input splits the bundled
    # output across (layer, fit-uns, vooma-uns); ndarray / EList / dict
    # callers get the full single-dict return as today.
    if _is_anndata(original_input):
        adata = original_input
        # Observation weights layer (limma (n_genes, n_samples) -> AnnData
        # (n_samples, n_genes) via as_matrix_weights normalisation + .T).
        W = as_matrix_weights(weights, (n_genes, n_samples))
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
        if use_sample_weights:
            ancillary["sample_weights"] = sw
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
      ``adata.uns[uns_key]`` (design, group, voom_xy and span). Returns
      ``None``.
    - EList in: returns a new ``EList`` with updated slots.
    - ndarray / dict in: returns a dict with ``E``, ``weights``,
      ``design``, ``group``, ``voom_xy`` and ``span``.

    As R, ``voom_xy["x"]`` and ``voom_xy["y"]`` hold each group's trend
    data, one column per group level (a singleton group takes the fit to
    all samples), and the default design is ``~0 + group`` with columns
    ``group<level>``.

    The default ``uns_key="vooma"`` means ``lm_fit(adata, layer="vooma_E")``
    picks up the design via the usual :func:`get_eawp` fallback.

    Notes
    -----
    **AnnData views.** Results are written into the AnnData. If it is a
    view (e.g. ``adata[:, mask]``), anndata first turns it into a
    standalone copy, with an ``ImplicitModificationWarning``, and the
    results land on that copy rather than on the parent. Assign the view
    to a variable first (``sub = adata[:, mask]``) or pass
    ``adata[:, mask].copy()``; a bare ``fn(adata[:, mask])`` call
    discards the results.
    """
    import pandas as pd

    from .classes import get_eawp, put_eawp

    original_input = y
    eawp = get_eawp(y, layer=layer)
    E = np.asarray(eawp["exprs"], dtype=np.float64)
    if design is None:
        design = _eawp_design(eawp)
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

    # R: as.factor(group); the level labels as R's as.character gives them
    levels, group_codes = _r_factor(group)
    ngroups = len(levels)
    if group_codes.size != narrays:
        raise ValueError("length(group) must equal ncol(y)")
    # R leaves such a sample's weights as its expression values
    if (group_codes < 0).any():
        raise ValueError("group contains missing values")
    group_sizes = np.bincount(group_codes, minlength=ngroups)
    # an unused level of a categorical; R fails in vooma on its empty
    # subset ("y contains entirely NA rows")
    if group_sizes.min() == 0:
        empty = [levels[j] for j in np.flatnonzero(group_sizes == 0)]
        raise ValueError(f"group levels with no samples: {empty}")

    if design is None:
        # R: model.matrix(~0+group), columns named group<level>
        design = np.column_stack([group_codes == j for j in range(ngroups)]).astype(np.float64)
        design_names = [f"group{lev}" for lev in levels]
    else:
        design, design_names = _numeric_design(design), _design_names(design)

    # Singleton levels take their weights from one fit to all samples (R's
    # voomall, computed once before the loop).
    if group_sizes.min() == 1:
        v_all = vooma(
            {"E": E},
            design=design,
            block=block,
            correlation=correlation,
            span=span,
            legacy_span=legacy_span,
            plot=False,
            save_plot=True,
        )

    # Each group's trend data. R: span <- voomi$span, of the last group
    # fitted on its own (the same for every group: it depends only on the
    # number of genes); voomall's when every group is a singleton, where R
    # fails ("object 'voomi' not found"; known differences)
    weights = np.empty_like(E)
    sx = np.zeros((ngenes, ngroups))
    sy = np.zeros((ngenes, ngroups))
    fitted_span = None
    for j in range(ngroups):
        mask = group_codes == j
        if mask.sum() < 2:
            v = v_all
            group_weights = v_all["weights"][:, mask]
        else:
            # R: vooma(y[, i], design = design[i, , drop = FALSE], block = block[i],
            #          correlation = correlation)
            v = vooma(
                {"E": E[:, mask]},
                design=design[mask, :],
                block=None if block is None else np.asarray(block)[mask],
                correlation=correlation,
                span=span,
                legacy_span=legacy_span,
                plot=False,
                save_plot=True,
            )
            group_weights = v["weights"]
            fitted_span = v["span"]
        weights[:, mask] = group_weights
        sx[:, j], sy[:, j] = v["voom_xy"]["x"], v["voom_xy"]["y"]

    return put_eawp(
        {
            "E": E,
            "weights": weights,
            "design": _named_design(design, design_names),
            "group": np.asarray(group),
            "voom_xy": {
                "x": pd.DataFrame(sx, columns=levels),
                "y": pd.DataFrame(sy, columns=levels),
                "xlab": "Average log-expression",
                "ylab": "Sqrt( standard deviation )",
            },
            "span": v_all["span"] if fitted_span is None else fitted_span,
        },
        original_input,
        out_layer=out_layer,
        weights_layer=weights_layer,
        uns_key=uns_key,
    )

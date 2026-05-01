# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   ebayes.R                   Copyright (C) 2003-2025 Gordon Smyth
# Python port: Copyright (C) 2026 John Mulvey
"""
Empirical Bayes moderation for pylimma.

Implements the core eBayes statistics from limma:
- e_bayes(): compute moderated t-statistics, p-values, B-statistics
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats
from scipy.special import gammaln

from .classes import MArrayLM, _resolve_fit_input
from .lmfit import is_fullrank
from .squeeze_var import squeeze_var

if TYPE_CHECKING:
    pass


def _t_isf_log(log_p: np.ndarray, df: float) -> np.ndarray:
    """
    Inverse survival function for t-distribution with log-scale input.

    R uses qt(log_p, df, lower.tail=FALSE, log.p=TRUE) which handles
    extreme p-values on the log scale without precision loss. Python's
    scipy.stats.t.isf requires the p-value itself, causing underflow for
    very small p-values.

    This function handles underflow by using an asymptotic approximation
    for very extreme values.

    Parameters
    ----------
    log_p : ndarray
        Log of tail probability (i.e., logsf values, all <= 0)
    df : float
        Degrees of freedom (scalar, common to all values)

    Returns
    -------
    ndarray
        Quantiles (positive values since we're dealing with upper tail)
    """
    result = np.empty_like(log_p)

    # Try standard method: isf(exp(log_p), df)
    p = np.exp(log_p)

    # Check for underflow: p == 0 but log_p is finite (not -inf)
    underflow = (p == 0) & np.isfinite(log_p)
    standard = ~underflow

    if np.any(standard):
        result[standard] = stats.t.isf(p[standard], df)

    if np.any(underflow):
        # Asymptotic approximation for extreme tail
        # For large t: sf(t, df) ~ c(df) * t^(-df)
        # where c(df) = Gamma((df+1)/2) / (sqrt(pi*df) * Gamma(df/2))
        # So: log_p ~ log(c(df)) - df * log(t)
        # => log(t) ~ (log(c(df)) - log_p) / df
        log_c = gammaln((df + 1) / 2) - 0.5 * np.log(np.pi * df) - gammaln(df / 2)
        log_t = (log_c - log_p[underflow]) / df
        result[underflow] = np.exp(log_t)

    return result


def _classify_tests_f(
    fit: dict,
    fstat_only: bool = True,
) -> np.ndarray:
    """
    Compute F-statistics from t-statistics.

    Internal wrapper that calls classify_tests_f from decide_tests module.
    """
    from .decide_tests import classify_tests_f

    return classify_tests_f(fit, p_value=0.01, fstat_only=fstat_only)


def _tmixture_vector(
    tstat: np.ndarray,
    stdev_unscaled: np.ndarray,
    df: np.ndarray | float,
    proportion: float,
    v0_lim: tuple[float, float] | None = None,
) -> float:
    """
    Estimate scale factor in mixture of two t-distributions.

    tstat is assumed to follow sqrt(1+v0/v1)*t(df) with probability `proportion`
    and t(df) otherwise. v1 is stdev_unscaled^2 and v0 is to be estimated.

    Parameters
    ----------
    tstat : ndarray
        t-statistics
    stdev_unscaled : ndarray
        Unscaled standard errors
    df : ndarray or float
        Degrees of freedom
    proportion : float
        Expected proportion of differentially expressed genes
    v0_lim : tuple, optional
        Limits for v0

    Returns
    -------
    float
        Estimated prior variance
    """
    # Remove missing values
    valid = ~np.isnan(tstat)
    if not np.all(valid):
        tstat = tstat[valid]
        stdev_unscaled = stdev_unscaled[valid]
        if isinstance(df, np.ndarray):
            df = df[valid]

    n_genes = len(tstat)
    n_target = int(np.ceil(proportion / 2 * n_genes))
    if n_target < 1:
        return np.nan

    # Ensure p at least matches selected proportion
    p = max(n_target / n_genes, proportion)

    # Work with absolute t-statistics
    tstat = np.abs(tstat)

    # Method requires equal df - adjust t-statistics to max df.
    # Copy to avoid mutating caller's df array (line below writes into df).
    df = np.atleast_1d(df).astype(np.float64, copy=True)
    if len(df) == 1:
        df = np.full(n_genes, df[0])
    max_df = np.max(df)

    needs_adjustment = df < max_df
    if np.any(needs_adjustment):
        # Convert via p-value on log scale (R parity: uses log.p=TRUE throughout)
        tail_p = stats.t.logsf(tstat[needs_adjustment], df[needs_adjustment])
        tstat[needs_adjustment] = _t_isf_log(tail_p, max_df)
        df[needs_adjustment] = max_df

    # Select top statistics. Use stable descending sort to match R's
    # order(..., decreasing=TRUE) (ebayes.R:143); default quicksort +
    # [::-1] has a reversed tie order which diverges on ties.
    order = np.argsort(-tstat, kind="stable")[:n_target]
    tstat = tstat[order]
    v1 = stdev_unscaled[order] ** 2

    # Compare to order statistics
    r = np.arange(1, n_target + 1)
    p0 = 2 * stats.t.sf(tstat, max_df)
    p_target = ((r - 0.5) / n_genes - (1 - p) * p0) / p

    v0 = np.zeros(n_target)
    pos = p_target > p0
    if np.any(pos):
        q_target = stats.t.isf(p_target[pos] / 2, max_df)
        v0[pos] = v1[pos] * ((tstat[pos] / q_target) ** 2 - 1)

    if v0_lim is not None:
        v0 = np.clip(v0, v0_lim[0], v0_lim[1])

    return float(np.mean(v0))


def _tmixture_matrix(
    tstat: np.ndarray,
    stdev_unscaled: np.ndarray,
    df: np.ndarray | float,
    proportion: float,
    v0_lim: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Estimate prior variance for each coefficient.

    Parameters
    ----------
    tstat : ndarray
        t-statistics (n_genes, n_coefs)
    stdev_unscaled : ndarray
        Unscaled standard errors (n_genes, n_coefs)
    df : ndarray or float
        Degrees of freedom
    proportion : float
        Expected proportion of DE genes
    v0_lim : tuple, optional
        Limits for v0

    Returns
    -------
    ndarray
        Prior variance for each coefficient
    """
    n_coefs = tstat.shape[1]
    v0 = np.zeros(n_coefs)
    for j in range(n_coefs):
        v0[j] = _tmixture_vector(tstat[:, j], stdev_unscaled[:, j], df, proportion, v0_lim)
    return v0


def e_bayes(
    data,
    proportion: float = 0.01,
    stdev_coef_lim: tuple[float, float] = (0.1, 4.0),
    trend: bool | np.ndarray = False,
    span: float | None = None,
    robust: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.1),
    legacy: bool | None = None,
    key: str = "pylimma",
) -> dict | None:
    """
    Empirical Bayes moderation of t-statistics.

    Computes moderated t-statistics, p-values, and B-statistics (log-odds
    of differential expression) by empirical Bayes shrinkage of the
    gene-wise sample variances towards a common prior.

    Parameters
    ----------
    data : AnnData or dict
        Either an AnnData object with fit results in adata.uns[key],
        or a dict returned by lm_fit() or contrasts_fit().
    proportion : float, default 0.01
        Expected proportion of differentially expressed genes.
    stdev_coef_lim : tuple, default (0.1, 4.0)
        Limits for the prior standard deviation of coefficients.
    trend : bool or array_like, default False
        If True, allow prior variance to depend on mean expression
        (Amean from the fit). If a numeric array, use that as the
        covariate for the mean-variance trend.
    span : float, optional
        Span for lowess smoothing when fitting the mean-variance trend.
        Only used when trend is True. If None, an appropriate span is
        chosen automatically.
    robust : bool, default False
        If True, use robust estimation of hyperparameters. Outlier
        variances are Winsorized and the prior df is estimated robustly.
    winsor_tail_p : tuple, default (0.05, 0.1)
        Winsorization proportions for robust estimation. The first value
        is for the lower tail, the second for the upper tail.
    legacy : bool, optional
        If True, use the original limma hyperparameter estimation method.
        If False, use the newer method which handles unequal residual df
        better. If None (default), auto-detect based on whether all
        residual df are equal.
    key : str, default "pylimma"
        Key for fit results in adata.uns (AnnData input only).

    Returns
    -------
    dict or None
        If input is dict, returns updated dict with moderated statistics.
        If input is AnnData, updates adata.uns[key] in place and returns None.

    Notes
    -----
    The moderated statistics added to the fit are:

    - t: moderated t-statistics
    - p_value: two-sided p-values
    - lods: B-statistics (log-odds of differential expression)
    - s2_prior: prior variance
    - df_prior: prior degrees of freedom
    - s2_post: posterior variance
    - df_total: total degrees of freedom
    - F: moderated F-statistic (if multiple contrasts)
    - F_p_value: F-statistic p-value (if multiple contrasts)

    References
    ----------
    Smyth, G. K. (2004). Linear models and empirical Bayes methods for
    assessing differential expression in microarray experiments.
    Statistical Applications in Genetics and Molecular Biology, 3(1), Article 3.
    """
    fit, adata, adata_key = _resolve_fit_input(data, key)
    is_anndata = adata is not None

    # Shallow-copy so e_bayes(fit) does not mutate the caller's
    # lm_fit/contrasts_fit output. Matches R's copy-on-modify:
    # `fit <- eBayes(fit)` returns a new list whether or not the
    # caller rebinds the original name. Slot arrays remain shared
    # references (none of the code below writes into them in place).
    fit = MArrayLM(fit)

    # Validate fit object
    coefficients = fit.get("coefficients")
    stdev_unscaled = fit.get("stdev_unscaled")
    sigma = fit.get("sigma")
    df_residual = fit.get("df_residual")

    if any(x is None for x in [coefficients, stdev_unscaled, sigma, df_residual]):
        raise ValueError("fit must contain coefficients, stdev_unscaled, sigma, df_residual")

    if np.max(df_residual) == 0:
        raise ValueError("No residual degrees of freedom in linear model fits")
    if not np.any(np.isfinite(sigma)):
        raise ValueError("No finite residual standard deviations")

    # Get covariate for trend estimation
    if isinstance(trend, bool):
        if trend:
            covariate = fit.get("Amean")
            if covariate is None:
                raise ValueError("Need Amean component in fit to estimate trend")
        else:
            covariate = None
    else:
        # trend is a numeric array
        trend = np.asarray(trend, dtype=np.float64)
        if len(trend) != len(sigma):
            raise ValueError(
                "If trend is numeric then it should have length equal to the number of genes"
            )
        covariate = trend

    # Squeeze variances
    sv = squeeze_var(
        sigma**2,
        df_residual,
        covariate=covariate,
        span=span,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
        legacy=legacy,
    )
    s2_prior = sv["var_prior"]
    s2_post = sv["var_post"]
    df_prior = sv["df_prior"]

    # Moderated t-statistics
    t_stat = coefficients / stdev_unscaled / np.sqrt(s2_post)[:, np.newaxis]

    # Total degrees of freedom (capped at pooled df)
    df_total = df_residual + df_prior
    df_pooled = np.nansum(df_residual)
    df_total = np.minimum(df_total, df_pooled)

    # P-values
    p_value = 2 * stats.t.sf(np.abs(t_stat), df_total[:, np.newaxis])

    # B-statistic (log-odds of DE)
    var_prior_lim = (
        stdev_coef_lim[0] ** 2 / np.median(s2_prior),
        stdev_coef_lim[1] ** 2 / np.median(s2_prior),
    )
    var_prior = _tmixture_matrix(t_stat, stdev_unscaled, df_total, proportion, var_prior_lim)

    if np.any(np.isnan(var_prior)):
        nan_mask = np.isnan(var_prior)
        s2_flat = np.atleast_1d(s2_prior).astype(np.float64).ravel()
        var_prior[nan_mask] = np.resize(1.0 / s2_flat, int(nan_mask.sum()))
        warnings.warn("Estimation of var_prior failed - set to default value")

    # Compute log-odds
    r = stdev_unscaled**2 + var_prior
    r = r / stdev_unscaled**2
    t2 = t_stat**2

    # Handle infinite df_prior (can be scalar or array in robust eBayes)
    inf_df = np.atleast_1d(df_prior > 1e6)
    if np.any(inf_df):
        # Some or all genes have infinite prior df - use limiting formula
        kernel = t2 * (1 - 1 / r) / 2
        if np.any(~inf_df):
            # Some genes have finite prior df - apply standard formula to those
            finite_mask = ~inf_df
            t2_f = t2[finite_mask]
            r_f = r[finite_mask]
            df_total_f = df_total[finite_mask]
            kernel[finite_mask] = (
                (1 + df_total_f[:, np.newaxis])
                / 2
                * np.log(
                    (t2_f + df_total_f[:, np.newaxis]) / (t2_f / r_f + df_total_f[:, np.newaxis])
                )
            )
    else:
        # All genes have finite prior df - use standard formula
        kernel = (
            (1 + df_total[:, np.newaxis])
            / 2
            * np.log((t2 + df_total[:, np.newaxis]) / (t2 / r + df_total[:, np.newaxis]))
        )

    lods = np.log(proportion / (1 - proportion)) - np.log(r) / 2 + kernel

    # Update fit
    fit["df_prior"] = df_prior
    fit["s2_prior"] = s2_prior
    fit["var_prior"] = var_prior
    # R's eBayes records the user-supplied DE proportion (ebayes.R:15);
    # stored for reproducibility/inspection - not read by downstream
    # pylimma code but part of the MArrayLM slot contract.
    fit["proportion"] = proportion
    fit["s2_post"] = s2_post
    fit["t"] = t_stat
    fit["df_total"] = df_total
    fit["p_value"] = p_value
    fit["lods"] = lods

    # F-statistic (if design is full rank)
    # For single coefficient, F = t^2 with df1 = 1 (handled by classify_tests_f)
    design = fit.get("design")
    if design is not None:
        if is_fullrank(design):
            f_stat, df1, df2 = _classify_tests_f(fit)
            fit["F"] = f_stat
            # df2 is a per-gene vector when df_residual varies
            # (NAs, ndups>1). Broadcast pf() over genes and fall back
            # to the chi-squared limit element-wise where df2 is Inf.
            df2_arr = np.asarray(df2, dtype=np.float64)
            if df2_arr.ndim == 0:
                if np.isinf(df2_arr):
                    fit["F_p_value"] = stats.chi2.sf(f_stat * df1, df1)
                else:
                    fit["F_p_value"] = stats.f.sf(f_stat, df1, df2_arr)
            else:
                mask_inf = np.isinf(df2_arr)
                df2_safe = np.where(mask_inf, 1.0, df2_arr)
                fp = stats.f.sf(f_stat, df1, df2_safe)
                if mask_inf.any():
                    fp = np.where(mask_inf, stats.chi2.sf(f_stat * df1, df1), fp)
                fit["F_p_value"] = fp

    if not isinstance(fit, MArrayLM):
        fit = MArrayLM(fit)

    if is_anndata:
        # Plain dict for h5ad compatibility; see lm_fit.
        data.uns[key] = dict(fit)
        return None
    return fit


def treat(
    data,
    fc: float = 1.2,
    lfc: float | None = None,
    trend: bool = False,
    robust: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.1),
    legacy: bool | None = None,
    upshot: bool = False,
    *,
    span: float | None = None,
    key: str = "pylimma",
) -> dict | None:
    r"""
    Moderated t-statistics relative to a log fold-change threshold.

    TREAT (t-tests relative to a threshold) tests for evidence that the
    true log fold-change is greater than a minimum value, rather than
    simply testing for non-zero effect.

    Parameters
    ----------
    data : AnnData or dict
        Either an AnnData object with fit results in adata.uns[key],
        or a dict returned by lm_fit() or contrasts_fit().
    fc : float, default 1.2
        Fold-change threshold. Used if lfc is None.
    lfc : float, optional
        Log2 fold-change threshold. If provided, overrides fc.
    trend : bool, default False
        If True, allow prior variance to depend on mean expression.
    span : float, optional
        Span for lowess smoothing when fitting the mean-variance trend.
        Only used when trend is True.
    robust : bool, default False
        If True, use robust estimation of hyperparameters.
    winsor_tail_p : tuple, default (0.05, 0.1)
        Winsorization proportions for robust estimation.
    legacy : bool, optional
        If True, use the original limma hyperparameter estimation method.
        If False, use the newer method which handles unequal residual df
        better. If None (default), auto-detect.
    upshot : bool, default False
        If True, use Gaussian quadrature to compute more accurate p-values
        when the log fold-change threshold is small. This averages the
        p-value over the interval [-lfc, lfc] rather than testing at the
        boundary.
    key : str, default "pylimma"
        Key for fit results in adata.uns (AnnData input only).

    Returns
    -------
    dict or None
        If input is dict, returns updated dict with TREAT statistics.
        If input is AnnData, updates adata.uns[key] in place and returns None.

    Notes
    -----
    The key difference from e_bayes() is that TREAT computes p-values for
    the hypothesis \|logFC\| > lfc, rather than logFC != 0. This is useful
    when you want to find genes with biologically meaningful effect sizes.

    The returned fit contains:
    - t: moderated t-statistics (for display, not for p-values)
    - p_value: p-values from TREAT test
    - treat_lfc: the log fold-change threshold used

    Note that B-statistics (lods) are not computed by TREAT.

    References
    ----------
    McCarthy, D. J. and Smyth, G. K. (2009). Testing significance relative
    to a fold-change threshold is a TREAT. Bioinformatics, 25(6), 765-771.
    """
    fit, adata, adata_key = _resolve_fit_input(data, key)
    is_anndata = adata is not None

    # Shallow-copy so treat(fit) does not mutate the caller's fit.
    # Matches R's copy-on-modify; see e_bayes for rationale.
    fit = MArrayLM(fit)

    # Validate fit object
    coefficients = fit.get("coefficients")
    stdev_unscaled = fit.get("stdev_unscaled")
    sigma = fit.get("sigma")
    df_residual = fit.get("df_residual")

    if any(x is None for x in [coefficients, stdev_unscaled, sigma, df_residual]):
        raise ValueError("fit must contain coefficients, stdev_unscaled, sigma, df_residual")

    if np.max(df_residual) == 0:
        raise ValueError("No residual degrees of freedom in linear model fits")
    if not np.any(np.isfinite(sigma)):
        raise ValueError("No finite residual standard deviations")

    # Clear any existing lods (B-statistics not computed for TREAT)
    fit["lods"] = None

    coefficients = np.asarray(coefficients)
    stdev_unscaled = np.asarray(stdev_unscaled)

    if trend:
        covariate = fit.get("Amean")
        if covariate is None:
            raise ValueError("Need Amean component in fit to estimate trend")
    else:
        covariate = None

    # Squeeze variances
    sv = squeeze_var(
        sigma**2,
        df_residual,
        covariate=covariate,
        span=span,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
        legacy=legacy,
    )
    fit["df_prior"] = sv["df_prior"]
    fit["s2_prior"] = sv["var_prior"]
    fit["s2_post"] = sv["var_post"]

    # Total degrees of freedom
    df_total = df_residual + sv["df_prior"]
    df_pooled = np.nansum(df_residual)
    df_total = np.minimum(df_total, df_pooled)
    fit["df_total"] = df_total

    # Determine log fold-change threshold
    if lfc is None:
        lfc = np.log2(fc)
    lfc = abs(lfc)
    fit["treat_lfc"] = lfc

    # Standard errors
    se = stdev_unscaled * np.sqrt(sv["var_post"])[:, np.newaxis]

    # Absolute coefficients
    acoef = np.abs(coefficients)

    # T-statistics for display (direction only)
    t_stat = np.zeros_like(coefficients)

    # Compute p-values using TREAT methodology
    if upshot and lfc > 0:
        # Use Gaussian quadrature for more accurate p-values
        # Matching R's gauss.quad.prob(16, dist="uniform", l=-lfc, u=lfc)
        from numpy.polynomial.legendre import leggauss

        nodes, weights = leggauss(16)
        # Transform nodes from [-1, 1] to [-lfc, lfc]
        nodes = lfc * nodes
        # Convert to probability weights (R divides by 2, not multiplies by interval)
        weights = weights / 2  # Now weights sum to 1.0 (probability weights)

        p_value = np.zeros_like(coefficients)
        # Use only the positive half (nodes 8:16 correspond to [0, lfc])
        # Positive half weights sum to 0.5
        for i in range(8, 16):
            lfci = nodes[i]
            tstat_right_i = (acoef - lfci) / se
            tstat_left_i = (acoef + lfci) / se
            p_value += weights[i] * (
                stats.t.sf(tstat_right_i, df_total[:, np.newaxis])
                + stats.t.sf(tstat_left_i, df_total[:, np.newaxis])
            )
        # Double for symmetry (we only integrated positive half)
        p_value = 2 * p_value

        # For display t-stat, use half the threshold (matching R's behaviour)
        lfc_display = lfc / 2
        tstat_right = np.maximum((acoef - lfc_display) / se, 0)
        tstat_left = (acoef + lfc_display) / se
        # In upshot mode, R uses lfc/2 for direction comparison
        lfc_threshold = lfc_display
    else:
        # Standard TREAT p-values
        tstat_right = (acoef - lfc) / se
        tstat_left = (acoef + lfc) / se

        # P-value is sum of both tails
        p_value = stats.t.sf(tstat_right, df_total[:, np.newaxis]) + stats.t.sf(
            tstat_left, df_total[:, np.newaxis]
        )

        # For display t-statistics, use the more conservative (closer to 0)
        tstat_right = np.maximum(tstat_right, 0)
        # In non-upshot mode, use original lfc for direction threshold
        lfc_threshold = lfc

    # Handle NaN coefficients
    coefficients_clean = coefficients.copy()
    coefficients_clean[np.isnan(coefficients_clean)] = 0

    # Assign t-statistics based on direction
    fc_up = coefficients_clean > lfc_threshold
    fc_down = coefficients_clean < -lfc_threshold
    t_stat[fc_up] = tstat_right[fc_up]
    t_stat[fc_down] = -tstat_right[fc_down]

    fit["t"] = t_stat
    fit["p_value"] = p_value

    if not isinstance(fit, MArrayLM):
        fit = MArrayLM(fit)

    if is_anndata:
        # Plain dict for h5ad compatibility; see lm_fit.
        data.uns[key] = dict(fit)
        return None
    return fit


def top_treat(
    fit,
    coef: int | str = 0,
    sort_by: str = "p",
    resort_by: str | None = None,
    **kwargs,
):
    """
    Top-ranked genes after a :func:`treat` fit.

    Thin wrapper around :func:`pylimma.top_table` that enforces the
    two guardrails R limma's ``topTreat`` enforces: ``coef`` must be a
    single coefficient, and ``sort_by="b"`` / ``resort_by="b"`` are
    rejected because :func:`treat` does not produce a B-statistic.

    Parameters
    ----------
    fit : MArrayLM, AnnData, or dict
        Fit object produced by :func:`treat`.
    coef : int or str, default 0
        Single coefficient index or name. Lists are not allowed; if a
        sequence is supplied only the first element is used and a
        warning is emitted.
    sort_by : str, default "p"
        Column to sort by. Passing ``"b"`` raises ``ValueError``.
    resort_by : str, optional
        Secondary sort column. Passing ``"b"`` raises ``ValueError``.
    **kwargs
        Forwarded to :func:`pylimma.top_table`.

    Returns
    -------
    pandas.DataFrame
        Ranked-gene table, same schema as :func:`top_table` minus the
        ``b`` column (which :func:`treat` does not compute).

    Notes
    -----
    R limma's own ``topTreat`` source comments that the function "may
    become deprecated soon as topTable() takes over"; pylimma keeps
    it for direct R-to-pylimma call-site compatibility.
    """
    from .toptable import top_table

    if np.ndim(coef) > 0 and len(coef) > 1:  # type: ignore[arg-type]
        warnings.warn(
            "treat is for single coefficients: only first value of coef being used",
            stacklevel=2,
        )
        coef = coef[0]  # type: ignore[index]

    # R treat.R:86-87 raises whenever sort.by/resort.by names B; the
    # comparison is on the raw user value, so capital "B" must also
    # be rejected.
    if isinstance(sort_by, str) and sort_by.lower() == "b":
        raise ValueError('Trying to sort_by="b", but treat does not produce a B-statistic')
    if isinstance(resort_by, str) and resort_by.lower() == "b":
        raise ValueError('Trying to resort_by="b", but treat does not produce a B-statistic')

    return top_table(fit, coef=coef, sort_by=sort_by, resort_by=resort_by, **kwargs)


def pred_fcm(
    fit,
    coef: int = 1,
    var_indep_of_fc: bool = True,
    all_de: bool = True,
    prop_true_null_method: str = "lfdr",
    *,
    key: str = "pylimma",
) -> np.ndarray:
    """
    Predictive (empirical-Bayes shrunken) fold changes.

    Port of R limma's ``predFCm`` (``predFCm.R``). Uses the eBayes
    posterior variance together with an estimated proportion of true
    nulls to shrink log-fold-changes toward zero.

    Parameters
    ----------
    fit : AnnData, MArrayLM, or dict
        Fit from :func:`lm_fit` + :func:`e_bayes`. For AnnData input,
        the fit is read from ``adata.uns[key]``.
    coef : int, default 1
        Zero-based coefficient index (R uses 1-based).
    var_indep_of_fc : bool, default True
        If True, assume the prior variance is independent of
        fold-change magnitude.
    all_de : bool, default True
        If True, assume all genes are differentially expressed.
    prop_true_null_method : {"lfdr", "convest", "mean", "hist"}
        Forwarded to :func:`prop_true_null`.
    key : str, default "pylimma"
        Key for fit results in adata.uns (AnnData input only).
    """
    from .utils import fit_gamma_intercept, prop_true_null

    # Accept AnnData-stored fits (parity with contrasts_fit / e_bayes /
    # treat / top_table / decide_tests).
    fit, _adata, _adata_key = _resolve_fit_input(fit, key)

    if fit.get("p_value") is None:
        fit = e_bayes(fit)

    coef = int(coef)
    p_col = fit["p_value"][:, coef]
    p = 1.0 - prop_true_null(p_col, method=prop_true_null_method)
    if p == 0:
        p = 1e-8
    trend = np.asarray(fit["s2_prior"]).size > 1
    robust = np.asarray(fit["df_prior"]).size > 1

    fit = e_bayes(fit, proportion=p, trend=trend, robust=robust)
    v = float(np.asarray(fit["cov_coefficients"])[coef, coef])
    coef_col = np.asarray(fit["coefficients"])[:, coef]
    s2_post = np.asarray(fit["s2_post"])

    if var_indep_of_fc:
        v0 = fit_gamma_intercept(coef_col**2, offset=v * s2_post)
        if v0 < 0:
            v0 = 1e-8
        pfc = coef_col * v0 / (v0 + v * s2_post)
        if not all_de:
            A = p / (1 - p)
            B = np.sqrt(v * s2_post / (v * s2_post + v0))
            C = np.exp(coef_col**2 * v0 / (2 * v**2 * s2_post**2 + 2 * v * v0 * s2_post))
            lods = np.log(A * B * C)
            prob_de = np.exp(lods) / (1 + np.exp(lods))
            prob_de[lods > 700] = 1.0
            pfc = pfc * prob_de
    else:
        b2 = coef_col**2 / s2_post
        v0 = fit_gamma_intercept(b2, offset=v)
        # R uses pmin here (bug-for-bug port).
        v0 = np.minimum(v0, 1e-8)
        pfc = coef_col * v0 / (v0 + v)
        if not all_de:
            A = p / (1 - p)
            B = np.sqrt(v / (v + v0))
            C = np.exp(coef_col**2 * v0 / (2 * v**2 * s2_post + 2 * v * v0 * s2_post))
            lods = np.log(A * B * C)
            prob_de = np.exp(lods) / (1 + np.exp(lods))
            prob_de[lods > 700] = 1.0
            pfc = pfc * prob_de
    return pfc

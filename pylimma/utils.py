# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   chooseLowessSpan.R         Copyright (C) 2020-2024 Gordon Smyth
#   weightedLowess.R (loess_fit port)
#                              Copyright (C) 2014-2020 Aaron Lun
#   qqt.R (qqt)                Copyright (C) 2002      Gordon Smyth
#   qqt.R (qqf)                Copyright (C) 2012      Belinda Phipson
#   ebayes.R (trigamma_inverse helper)
#                              Copyright (C) 2002-2004 Gordon Smyth
#   zscore.R                   Copyright (C) 2003-2020 Gordon Smyth
#   zscoreHyper.R              Copyright (C) 2012      Gordon Smyth
#   tricubeMovingAverage.R     Copyright (C) 2014-2015 Gordon Smyth,
#                                                      Yifang Hu
#   convest.R                  Copyright (C) 2004-2020 Egil Ferkingstad,
#                                                      Mette Langaas,
#                                                      Gordon Smyth,
#                                                      Marcus Davy
#   propTrueNull.R             Copyright (C) 2012      Belinda Phipson,
#                                                      Gordon Smyth
#   detectionPValues.R         Copyright (C) 2016      Gordon Smyth
#   logsumexp.R (logcosh, logsumexp)
#                              Copyright (C) 2007-2018 Gordon Smyth
#   bwss.R (bwss, bwss_matrix) Copyright (C) 2002      Gordon Smyth
#   utility.R (is_numeric, block_diag)
#                              Copyright (C) 2003-2004 Gordon Smyth
#   combine.R (make_unique)    Copyright (C) 2003-2016 Gordon Smyth
#
# logmdigamma() is a port of the same-named function from the R statmod
# package (not from limma):
#   statmod::logmdigamma       Copyright (C) Gordon Smyth, Lizhong Chen;
#                              GPL-2 | GPL-3
# Python port: Copyright (C) 2026 John Mulvey
"""
Utility functions for pylimma.

Internal helper functions used throughout the package.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats
from scipy.special import digamma, polygamma


def trigamma_inverse(x: np.ndarray | float) -> np.ndarray | float:
    """
    Solve trigamma(y) = x for y.

    Uses Newton iteration on the inverse of trigamma. Port of R limma's
    trigammaInverse function (Gordon Smyth, 2002-2004).

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the inverse trigamma function.
        Must be positive for valid results.

    Returns
    -------
    ndarray or float
        The value y such that trigamma(y) = x.

    Notes
    -----
    - Returns NaN for negative inputs (with warning)
    - For very large x (> 1e7), uses asymptotic approximation 1/sqrt(x)
    - For very small x (< 1e-6), uses asymptotic approximation 1/x
    """
    x = np.asarray(x)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x).astype(np.float64)

    y = np.empty_like(x)

    # Handle special cases (order matters - check sequentially)
    na_mask = np.isnan(x)
    neg_mask = ~na_mask & (x < 0)
    large_mask = ~na_mask & ~neg_mask & (x > 1e7)
    small_mask = ~na_mask & ~neg_mask & (x < 1e-6)
    normal_mask = ~(na_mask | neg_mask | large_mask | small_mask)

    # Propagate NaN
    y[na_mask] = np.nan

    # Negative values -> NaN with warning
    if np.any(neg_mask):
        import warnings
        warnings.warn("NaNs produced", RuntimeWarning)
        y[neg_mask] = np.nan

    # Asymptotic approximations
    y[large_mask] = 1.0 / np.sqrt(x[large_mask])
    y[small_mask] = 1.0 / x[small_mask]

    # Newton iteration for normal range
    if np.any(normal_mask):
        x_norm = x[normal_mask]
        # Initial guess: 1/trigamma(y) is approximately y - 0.5 for moderate y
        y_iter = 0.5 + 1.0 / x_norm

        for _ in range(50):
            tri = polygamma(1, y_iter)  # trigamma
            psigamma2 = polygamma(2, y_iter)  # tetragamma
            dif = tri * (1.0 - tri / x_norm) / psigamma2
            y_iter = y_iter + dif
            if np.max(-dif / y_iter) < 1e-8:
                break
        else:
            import warnings
            warnings.warn("Iteration limit exceeded in trigamma_inverse")

        y[normal_mask] = y_iter

    if scalar_input:
        return float(y[0])
    return y


def logmdigamma(x: np.ndarray | float) -> np.ndarray | float:
    """
    Compute log(x) - digamma(x).

    This function is used in the moment estimation of scaled F-distributions.
    Imported from statmod in R limma.

    Parameters
    ----------
    x : array_like
        Input values. Should be positive.

    Returns
    -------
    ndarray or float
        log(x) - digamma(x)
    """
    x = np.asarray(x)
    return np.log(x) - digamma(x)


def qqt(
    y: np.ndarray,
    df: float = np.inf,
    plot_it: bool = True,
    **kwargs,
) -> dict:
    """
    Student's t probability plot (Q-Q plot).

    Produces a Q-Q plot comparing sample quantiles against theoretical
    quantiles from a t-distribution.

    Parameters
    ----------
    y : array_like
        Sample values (e.g., t-statistics).
    df : float, default np.inf
        Degrees of freedom for the t-distribution.
        Use np.inf for a normal distribution.
    plot_it : bool, default True
        If True, produce the Q-Q plot. Requires matplotlib.
    **kwargs
        Additional arguments passed to matplotlib's plot function.

    Returns
    -------
    dict
        x : ndarray
            Theoretical quantiles.
        y : ndarray
            Sample quantiles (sorted input values).

    Examples
    --------
    >>> fit = e_bayes(lm_fit(expr, design))
    >>> qqt(fit["t"][:, 0], df=fit["df_total"][0])
    """
    from scipy import stats

    y = np.asarray(y)
    y = y[~np.isnan(y)]
    n = len(y)
    if n == 0:
        raise ValueError("y is empty or all NaN")

    # Probability points
    p = (np.arange(1, n + 1) - 0.5) / n

    # Theoretical quantiles
    x = stats.t.ppf(p, df)[np.argsort(np.argsort(y))]

    if plot_it:
        try:
            import matplotlib.pyplot as plt

            plot_kwargs = {
                "marker": "o",
                "linestyle": "none",
                "markersize": 3,
            }
            plot_kwargs.update(kwargs)

            fig, ax = plt.subplots()
            ax.plot(x, y, **plot_kwargs)

            # Add reference line
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            min_val = max(xlim[0], ylim[0])
            max_val = min(xlim[1], ylim[1])
            ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.7)

            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            title = "Student's t Q-Q Plot" if np.isfinite(df) else "Normal Q-Q Plot"
            ax.set_title(title)
            plt.tight_layout()

        except ImportError:
            import warnings
            warnings.warn("matplotlib not available for plotting")

    return {"x": x, "y": y}


def qqf(
    y: np.ndarray,
    df1: float,
    df2: float,
    plot_it: bool = True,
    **kwargs,
) -> dict:
    """
    F-distribution probability plot (Q-Q plot).

    Produces a Q-Q plot comparing sample quantiles against theoretical
    quantiles from an F-distribution.

    Parameters
    ----------
    y : array_like
        Sample values (e.g., F-statistics or variance ratios).
    df1 : float
        Numerator degrees of freedom.
    df2 : float
        Denominator degrees of freedom.
    plot_it : bool, default True
        If True, produce the Q-Q plot. Requires matplotlib.
    **kwargs
        Additional arguments passed to matplotlib's plot function.

    Returns
    -------
    dict
        x : ndarray
            Theoretical quantiles.
        y : ndarray
            Sample quantiles (sorted input values).
    """
    from scipy import stats

    y = np.asarray(y)
    y = y[~np.isnan(y)]
    n = len(y)
    if n == 0:
        raise ValueError("y is empty or all NaN")

    # Probability points
    p = (np.arange(1, n + 1) - 0.5) / n

    # Theoretical quantiles
    x = stats.f.ppf(p, df1, df2)[np.argsort(np.argsort(y))]

    if plot_it:
        try:
            import matplotlib.pyplot as plt

            plot_kwargs = {
                "marker": "o",
                "linestyle": "none",
                "markersize": 3,
            }
            plot_kwargs.update(kwargs)

            fig, ax = plt.subplots()
            ax.plot(x, y, **plot_kwargs)

            # Add reference line
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            min_val = max(xlim[0], ylim[0])
            max_val = min(xlim[1], ylim[1])
            ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.7)

            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            ax.set_title("F Distribution Q-Q Plot")
            plt.tight_layout()

        except ImportError:
            import warnings
            warnings.warn("matplotlib not available for plotting")

    return {"x": x, "y": y}


def choose_lowess_span(
    n: int = 1000,
    small_n: int = 50,
    min_span: float = 0.3,
    power: float = 1/3,
) -> float:
    """
    Choose optimal span for LOWESS smoothing of variance trends.

    Larger spans are used for small datasets and smaller spans for larger
    datasets.

    Parameters
    ----------
    n : int, default 1000
        Number of data points.
    small_n : int, default 50
        Threshold below which maximum smoothing is used.
    min_span : float, default 0.3
        Minimum span for large datasets.
    power : float, default 1/3
        Power for scaling between small and large datasets.

    Returns
    -------
    float
        Recommended LOWESS span parameter.
    """
    return min(min_span + (1 - min_span) * (small_n / n) ** power, 1.0)


def _find_span_window(
    idx: int,
    x_sorted: np.ndarray,
    weights_sorted: np.ndarray,
    span_weight: float,
    n: int,
) -> tuple:
    """
    Find the span window for a given point using cumulative weights.

    Extends left and right from idx until cumulative weight reaches span_weight.
    Returns (left, right, max_dist) where max_dist is the maximum distance
    to either boundary.
    """
    left = idx
    right = idx
    cur_weight = weights_sorted[idx]
    max_dist = 0.0

    at_start = (left == 0)
    at_end = (right == n - 1)

    while cur_weight < span_weight and (not at_start or not at_end):
        if at_end:
            # Can only extend left
            left -= 1
            cur_weight += weights_sorted[left]
            if left == 0:
                at_start = True
            ldist = x_sorted[idx] - x_sorted[left]
            max_dist = max(max_dist, ldist)
        elif at_start:
            # Can only extend right
            right += 1
            cur_weight += weights_sorted[right]
            if right == n - 1:
                at_end = True
            rdist = x_sorted[right] - x_sorted[idx]
            max_dist = max(max_dist, rdist)
        else:
            # Extend in direction of closer point
            ldist = x_sorted[idx] - x_sorted[left - 1]
            rdist = x_sorted[right + 1] - x_sorted[idx]
            if ldist < rdist:
                left -= 1
                cur_weight += weights_sorted[left]
                if left == 0:
                    at_start = True
                max_dist = max(max_dist, ldist)
            else:
                right += 1
                cur_weight += weights_sorted[right]
                if right == n - 1:
                    at_end = True
                max_dist = max(max_dist, rdist)

    # Extend to include tied x values
    while left > 0 and x_sorted[left] == x_sorted[left - 1]:
        left -= 1
    while right < n - 1 and x_sorted[right] == x_sorted[right + 1]:
        right += 1

    return left, right, max_dist


def _weighted_local_regression(
    x_fit: float,
    x_window: np.ndarray,
    y_window: np.ndarray,
    obs_weights: np.ndarray,
    rob_weights: np.ndarray,
    max_dist: float,
) -> float:
    """
    Fit weighted local linear regression at a single point.

    Combined weight = tricube(distance) * observation_weight * robustness_weight
    """
    threshold = 1e-7

    # If max_dist is tiny, return weighted mean
    if max_dist < threshold:
        w = obs_weights * rob_weights
        total_w = np.sum(w)
        if total_w == 0:
            return 0.0
        return np.sum(y_window * w) / total_w

    # Tricube kernel weights: (1 - |u|^3)^3
    u = np.abs(x_window - x_fit) / max_dist
    kernel_w = np.where(u < 1, (1 - u ** 3) ** 3, 0.0)

    # Combined weights
    w = kernel_w * obs_weights * rob_weights
    total_w = np.sum(w)

    if total_w == 0:
        return 0.0

    # Weighted means
    x_mean = np.sum(w * x_window) / total_w
    y_mean = np.sum(w * y_window) / total_w

    # Weighted variance and covariance for local linear fit
    x_centered = x_window - x_mean
    var = np.sum(w * x_centered ** 2)
    covar = np.sum(w * x_centered * (y_window - y_mean))

    # If variance is tiny, return weighted mean
    if var < threshold:
        return y_mean

    # Local linear fit: y = slope * x + intercept
    slope = covar / var
    return slope * x_fit + y_mean - slope * x_mean


def _weighted_median_abs_deviation(
    residuals: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute weighted median absolute deviation scaled by 6."""
    order = np.argsort(np.abs(residuals))
    sorted_resid = np.abs(residuals[order])
    sorted_weights = weights[order]

    total_weight = np.sum(sorted_weights)
    half_weight = total_weight / 2.0
    cumw = 0.0

    for i in range(len(residuals)):
        cumw += sorted_weights[i]
        if cumw == half_weight and i < len(residuals) - 1:
            return 3.0 * (sorted_resid[i] + sorted_resid[i + 1])
        elif cumw > half_weight:
            return 6.0 * sorted_resid[i]

    return 6.0 * sorted_resid[-1]


def loess_fit(
    y: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray | None = None,
    span: float = 0.3,
    iterations: int = 4,
    min_weight: float = 1e-5,
    max_weight: float = 1e5,
) -> dict:
    """
    Weighted LOWESS fit for univariate x and y.

    Implements weighted local regression matching R limma's weightedLowess.
    Observation weights affect both span window selection (which neighbours
    to include) and the local regression fitting.

    Parameters
    ----------
    y : array_like
        Response values.
    x : array_like
        Predictor values.
    weights : array_like, optional
        Observation weights. Higher weights give more influence to points
        and effectively make them "count more" when determining the span
        neighbourhood.
    span : float, default 0.3
        Smoothing parameter (fraction of total weight used for each fit).
    iterations : int, default 4
        Number of robustifying iterations (1 = no robustness iterations).
    min_weight : float, default 1e-5
        Minimum weight value.
    max_weight : float, default 1e5
        Maximum weight value.

    Returns
    -------
    dict
        fitted : ndarray
            Fitted values.
        residuals : ndarray
            Residuals (y - fitted).
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(y)

    if len(x) != n:
        raise ValueError("x and y have different lengths")

    # Initialize output
    fitted = np.full(n, np.nan)
    residuals = np.full(n, np.nan)

    # Find valid observations
    obs = np.isfinite(y) & np.isfinite(x)
    if not np.any(obs):
        return {"fitted": fitted, "residuals": residuals}

    x_obs = x[obs]
    y_obs = y[obs]
    n_obs = int(np.sum(obs))

    # Check span
    if span < 1 / n_obs:
        fitted[obs] = y_obs
        residuals[obs] = 0
        return {"fitted": fitted, "residuals": residuals}

    # Handle weights
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != n:
            raise ValueError("y and weights have different lengths")
        w_obs = weights[obs].copy()
        w_obs = np.nan_to_num(w_obs, nan=0)
        w_obs = np.clip(w_obs, min_weight, max_weight)
    else:
        w_obs = np.ones(n_obs, dtype=np.float64)

    # Sort by x (stable sort to match R behaviour)
    order = np.argsort(x_obs, kind='mergesort')
    x_sorted = x_obs[order]
    y_sorted = y_obs[order]
    w_sorted = w_obs[order]

    # Compute span weight threshold
    total_weight = np.sum(w_sorted)
    span_weight = total_weight * span

    # Precompute span windows for all points
    windows = [
        _find_span_window(i, x_sorted, w_sorted, span_weight, n_obs)
        for i in range(n_obs)
    ]

    # Initialize robustness weights
    rob_weights = np.ones(n_obs, dtype=np.float64)
    fitted_sorted = np.zeros(n_obs, dtype=np.float64)

    # Iterative fitting with robustness weights
    threshold = 1e-7
    for _it in range(iterations):
        # Fit at each point
        for i in range(n_obs):
            left, right, max_dist = windows[i]
            fitted_sorted[i] = _weighted_local_regression(
                x_sorted[i],
                x_sorted[left:right + 1],
                y_sorted[left:right + 1],
                w_sorted[left:right + 1],
                rob_weights[left:right + 1],
                max_dist,
            )

        # Last iteration doesn't need robustness weight update
        if _it == iterations - 1:
            break

        # Compute residuals and update robustness weights
        resid = y_sorted - fitted_sorted
        abs_resid = np.abs(resid)
        resid_scale = np.mean(abs_resid)

        # Weighted median absolute deviation
        cmad = _weighted_median_abs_deviation(resid, w_sorted)

        # Check convergence
        if cmad <= threshold * resid_scale:
            break

        # Update robustness weights using bisquare kernel
        u = abs_resid / cmad
        rob_weights = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)

    # Map back to original order
    inv_order = np.argsort(order)
    fitted_obs = fitted_sorted[inv_order]

    fitted[obs] = fitted_obs
    residuals[obs] = y_obs - fitted_obs

    return {"fitted": fitted, "residuals": residuals}


def weighted_lowess(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    delta: float | None = None,
    npts: int = 200,
    span: float = 0.3,
    iterations: int = 4,
    output_style: str = "loess",
) -> dict:
    """
    Weighted LOWESS smoother - R-compatible entry point.

    Thin wrapper around :func:`loess_fit` that mirrors R limma's
    ``weightedLowess(x, y, ...)`` argument order and parameter names.
    Use this when porting R code that calls ``weightedLowess``
    directly. New Python code should prefer :func:`loess_fit`.

    Parameters
    ----------
    x, y : array_like
        Predictor and response (R's argument order - note that
        :func:`loess_fit` takes ``(y, x)``).
    weights : array_like, optional
        Per-observation weights.
    delta : float, optional
        Clustering tolerance for the anchor-point-based smoother.
        Ignored by the current implementation, which uses the
        exact (non-clustered) weighted LOWESS algorithm. A warning is
        emitted if supplied.
    npts : int, default 200
        Number of anchor points. Ignored by the current
        implementation for the same reason as ``delta``. A warning is
        emitted if the caller supplies a non-default value.
    span, iterations :
        See :func:`loess_fit`.
    output_style : {"loess", "lowess"}, default "loess"
        - ``"loess"``: return ``{"fitted", "residuals", "weights",
          "delta"}`` in the caller's original point order (matches
          R's ``loess()`` / ``loessFit()`` output shape and
          :func:`loess_fit`).
        - ``"lowess"``: return ``{"x", "y", "delta"}`` with both
          arrays sorted by ``x`` ascending (matches R's ``lowess()``
          output shape).

    Returns
    -------
    dict
        Schema depends on ``output_style`` - see above.
    """
    import warnings as _warnings

    if delta is not None:
        _warnings.warn(
            "weighted_lowess: 'delta' is ignored by pylimma's exact "
            "weighted LOWESS implementation",
            stacklevel=2,
        )
    if npts != 200:
        _warnings.warn(
            "weighted_lowess: 'npts' is ignored by pylimma's exact "
            "weighted LOWESS implementation (no anchor-point "
            "clustering is performed)",
            stacklevel=2,
        )
    output_style = output_style.lower()
    if output_style not in ("loess", "lowess"):
        raise ValueError(
            "output_style must be 'loess' or 'lowess', got "
            f"{output_style!r}"
        )

    fit = loess_fit(y, x, weights=weights, span=span, iterations=iterations)
    if output_style == "loess":
        fit["delta"] = delta
        if weights is not None:
            fit["weights"] = np.asarray(weights, dtype=np.float64)
        return fit

    # "lowess" output: order-by-x, drop residuals, expose {x, y, delta}.
    x_arr = np.asarray(x, dtype=np.float64)
    order = np.argsort(x_arr, kind="stable")
    return {
        "x": x_arr[order],
        "y": np.asarray(fit["fitted"], dtype=np.float64)[order],
        "delta": delta,
    }


def p_adjust(p: np.ndarray, method: str = "BH") -> np.ndarray:
    """
    Adjust p-values for multiple testing.

    Wrapper around statsmodels multipletests for convenience.

    Parameters
    ----------
    p : array_like
        Raw p-values.
    method : str, default "BH"
        Correction method. Options: "BH" (Benjamini-Hochberg), "BY"
        (Benjamini-Yekutieli), "holm", "hochberg", "hommel",
        "bonferroni", "none".

    Returns
    -------
    ndarray
        Adjusted p-values.
    """
    from statsmodels.stats.multitest import multipletests

    p = np.asarray(p, dtype=np.float64)
    m = method.lower()

    if m == "none":
        return p.copy()

    valid_methods = {"bh", "fdr", "by", "bonferroni", "holm", "hommel", "hochberg"}
    if m not in valid_methods:
        raise ValueError(
            f"method '{method}' not recognized. Valid methods: "
            "'BH', 'BY', 'holm', 'hochberg', 'hommel', 'bonferroni', 'none'"
        )

    valid = ~np.isnan(p)
    if not np.any(valid):
        return p.copy()

    adjusted = np.full_like(p, np.nan)
    pv = p[valid]

    if m == "hochberg":
        # Port of R's p.adjust(method="hochberg"): step-up procedure.
        # R code: pmin(1, cummin((n - i + 1L) * p[o]))[ro]
        # where o = order(p, decreasing=TRUE), i = n:1.
        n = pv.size
        o = np.argsort(-pv, kind="stable")
        ro = np.argsort(o, kind="stable")
        multipliers = np.arange(1, n + 1, dtype=np.float64)
        adj_sorted = np.minimum.accumulate(multipliers * pv[o])
        adj_valid = np.minimum(adj_sorted[ro], 1.0)
    else:
        method_map = {
            "bh": "fdr_bh",
            "fdr": "fdr_bh",
            "by": "fdr_by",
            "bonferroni": "bonferroni",
            "holm": "holm",
            "hommel": "hommel",
        }
        _, adj_valid, _, _ = multipletests(pv, method=method_map[m])

    adjusted[valid] = adj_valid
    return adjusted


# ---------------------------------------------------------------------------
# zscore family (port of R limma's zscore, zscoreGamma, zscoreHyper)
# ---------------------------------------------------------------------------


_ZSCORE_DISTRIBUTIONS = {
    "norm": _scipy_stats.norm,
    "t": _scipy_stats.t,
    "f": _scipy_stats.f,
    "chisq": _scipy_stats.chi2,
    "beta": _scipy_stats.beta,
    "exp": _scipy_stats.expon,
    "binom": _scipy_stats.binom,
    "pois": _scipy_stats.poisson,
}


def zscore(
    q: np.ndarray | float,
    distribution: str,
    **dist_kwargs,
) -> np.ndarray:
    """
    Z-score equivalents for deviates from a specified distribution.

    Port of R limma's ``zscore``. Computes the z-score of a value drawn
    from ``distribution`` by mapping the corresponding tail probability
    onto a standard normal quantile, preserving log-scale accuracy in
    the tails.

    Parameters
    ----------
    q : array_like
        Deviates whose z-score equivalents are wanted.
    distribution : str
        Distribution name. R's naming convention (without the leading
        ``p``) is used: ``"norm"``, ``"t"``, ``"f"``, ``"chisq"``,
        ``"beta"``, ``"exp"``, ``"binom"``, ``"pois"``. Use the
        dedicated :func:`zscore_gamma` and :func:`zscore_hyper`
        functions for the gamma and hypergeometric cases.
    **dist_kwargs
        Distribution parameters passed through to scipy.stats. Names
        mirror R's ``p<distribution>`` arguments (e.g. ``df`` for
        ``"t"``; ``df1`` and ``df2`` for ``"f"``).

    Returns
    -------
    ndarray
        Z-scores with the same shape as ``q``.
    """
    from scipy.special import ndtri_exp

    if distribution == "gamma":
        return zscore_gamma(q, **dist_kwargs)
    if distribution == "hyper":
        return zscore_hyper(q, **dist_kwargs)
    if distribution not in _ZSCORE_DISTRIBUTIONS:
        raise ValueError(
            f"distribution '{distribution}' not recognised. "
            "Must be one of 'norm', 't', 'f', 'gamma', 'hyper', "
            "'chisq', 'beta', 'exp', 'binom', 'pois'."
        )
    dist = _ZSCORE_DISTRIBUTIONS[distribution]

    q = np.asarray(q, dtype=np.float64)
    z = q.astype(np.float64, copy=True)

    pupper = dist.logsf(q, **dist_kwargs)
    plower = dist.logcdf(q, **dist_kwargs)
    up = pupper < plower
    if np.any(up):
        z[up] = -ndtri_exp(pupper[up])
    if np.any(~up):
        z[~up] = ndtri_exp(plower[~up])
    return z


def zscore_gamma(
    q: np.ndarray | float,
    shape: np.ndarray | float,
    rate: np.ndarray | float = 1.0,
    scale: np.ndarray | float | None = None,
) -> np.ndarray:
    """
    Z-score equivalents for gamma deviates.

    Port of R limma's ``zscoreGamma``.

    Parameters
    ----------
    q : array_like
        Gamma deviates.
    shape : array_like or float
        Shape parameter. Recycled to the length of ``q``.
    rate : array_like or float, default 1.0
        Rate parameter. Used only if ``scale`` is omitted; ``scale``
        defaults to ``1 / rate`` to mirror R's argument convention.
    scale : array_like or float, optional
        Scale parameter (``1 / rate``). When supplied takes precedence
        over ``rate``.

    Returns
    -------
    ndarray
        Z-score equivalents with the same shape as ``q``.
    """
    from scipy.special import ndtri_exp

    q = np.asarray(q, dtype=np.float64)
    n = q.size
    shape = np.broadcast_to(np.asarray(shape, dtype=np.float64), n).copy()
    if scale is None:
        scale = 1.0 / np.asarray(rate, dtype=np.float64)
    scale = np.broadcast_to(np.asarray(scale, dtype=np.float64), n).copy()
    z = q.astype(np.float64, copy=True)

    q_flat = q.reshape(-1)
    z_flat = z.reshape(-1)
    up = q_flat > shape * scale
    if np.any(up):
        logp = _scipy_stats.gamma.logsf(
            q_flat[up], a=shape[up], scale=scale[up]
        )
        z_flat[up] = -ndtri_exp(logp)
    if np.any(~up):
        logp = _scipy_stats.gamma.logcdf(
            q_flat[~up], a=shape[~up], scale=scale[~up]
        )
        z_flat[~up] = ndtri_exp(logp)
    return z


def zscore_hyper(
    q: np.ndarray | float,
    m: np.ndarray | float,
    n: np.ndarray | float,
    k: np.ndarray | float,
) -> np.ndarray:
    """
    Z-score equivalents for hypergeometric deviates.

    Port of R limma's ``zscoreHyper``. Adds a half point-probability
    correction to either the upper or lower tail before mapping to a
    standard normal quantile, preserving log-scale accuracy.

    Parameters
    ----------
    q : array_like
        Number of successes observed.
    m : array_like or float
        Number of white balls in the urn.
    n : array_like or float
        Number of black balls in the urn.
    k : array_like or float
        Number of balls drawn.

    Returns
    -------
    ndarray
        Z-score equivalents with the same shape as ``q``.
    """
    from scipy.special import ndtri_exp

    q = np.asarray(q, dtype=np.float64)
    z = q.astype(np.float64, copy=True)
    M = np.asarray(m, dtype=np.float64) + np.asarray(n, dtype=np.float64)
    m_arr = np.asarray(m, dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)

    with np.errstate(invalid="ignore"):
        d = _scipy_stats.hypergeom.logpmf(q, M, m_arr, k_arr) - np.log(2.0)
        pupper = _scipy_stats.hypergeom.logsf(q, M, m_arr, k_arr)
        plower = _scipy_stats.hypergeom.logcdf(q - 1, M, m_arr, k_arr)

    d = np.where(np.isnan(d), -np.inf, d)
    pupper = np.where(np.isnan(pupper), -np.inf, pupper)
    plower = np.where(np.isnan(plower), -np.inf, plower)

    # Add half point probability to upper tail in log-space.
    a = np.where(d > pupper, d, pupper)
    b = -np.abs(d - pupper)
    pmidupper = a + np.log1p(np.exp(b))
    inf_a = np.isinf(a)
    pmidupper = np.where(inf_a, a, pmidupper)

    # Similarly for lower tail.
    a = np.where(d > plower, d, plower)
    b = -np.abs(d - plower)
    pmidlower = a + np.log1p(np.exp(b))
    inf_a = np.isinf(a)
    pmidlower = np.where(inf_a, a, pmidlower)

    up = pmidupper < pmidlower
    if np.any(up):
        z[up] = -ndtri_exp(pmidupper[up])
    if np.any(~up):
        z[~up] = ndtri_exp(pmidlower[~up])
    return z


# ---------------------------------------------------------------------------
# zscore_t (port of R limma's zscoreT and its dot-prefixed helpers)
# ---------------------------------------------------------------------------


def _zscore_t_quantile(x: np.ndarray, df: np.ndarray) -> np.ndarray:
    # qnorm(pt(abs(x), df, lower.tail=FALSE, log.p=TRUE),
    #       lower.tail=FALSE, log.p=TRUE) * sign(x)
    logp = _scipy_stats.t.logsf(np.abs(x), df)
    return _scipy_stats.norm.isf(np.exp(logp)) * np.sign(x)


def _zscore_t_wallace(x: np.ndarray, df: np.ndarray) -> np.ndarray:
    return (
        (df + 0.125) / (df + 0.375)
        * np.sqrt(df * np.log1p(x / df * x))
        * np.sign(x)
    )


def _zscore_t_bailey(x: np.ndarray, df: np.ndarray) -> np.ndarray:
    return (
        (df + 0.125) / (df + 1.125)
        * np.sqrt((df + 19.0 / 12.0) * np.log1p(x / (df + 1.0 / 12.0) * x))
        * np.sign(x)
    )


def _zscore_t_hill(x: np.ndarray, df: np.ndarray) -> np.ndarray:
    A = df - 0.5
    B = 48.0 * A * A
    z = A * np.log1p(x / df * x)
    z = (
        (((((-0.4 * z - 3.3) * z - 24.0) * z - 85.5)
          / (0.8 * z * z + 100.0 + B) + z + 3.0) / B + 1.0
        ) * np.sqrt(z)
    )
    return z * np.sign(x)


def zscore_t(
    x: np.ndarray,
    df: np.ndarray | float,
    approx: bool = False,
    method: str = "bailey",
) -> np.ndarray:
    """
    Z-score equivalents of t-statistics.

    Port of R limma's ``zscoreT``. When ``approx=True`` the calculation uses
    one of three closed-form approximations selected by ``method``
    (``"bailey"``, ``"hill"``, ``"wallace"``). When ``approx=False`` the
    z-scores are computed via the exact quantile transformation using the
    log-scale t-distribution survival function; ``method`` is ignored on
    that path.

    Parameters
    ----------
    x : array_like
        t-statistics (any shape).
    df : array_like or float
        Degrees of freedom. Broadcasts against ``x``.
    approx : bool, default False
        If True, use a closed-form approximation.
    method : {"bailey", "hill", "wallace"}, default "bailey"
        Approximation to use when ``approx=True``. Ignored when
        ``approx=False``.

    Returns
    -------
    ndarray
        Z-score equivalents with the same shape as ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    df = np.asarray(df, dtype=np.float64)
    df = np.broadcast_to(df, x.shape).astype(np.float64, copy=True)

    if approx:
        df = np.minimum(df, 1e100)
        if method not in ("bailey", "hill", "wallace"):
            raise ValueError(
                f"method '{method}' not recognized. "
                "Must be one of 'bailey', 'hill', 'wallace'."
            )
        if method == "bailey":
            return _zscore_t_bailey(x, df)
        if method == "hill":
            return _zscore_t_hill(x, df)
        return _zscore_t_wallace(x, df)
    return _zscore_t_quantile(x, df)


# ---------------------------------------------------------------------------
# tricube_moving_average (port of R limma's tricubeMovingAverage)
# ---------------------------------------------------------------------------


def tricube_moving_average(
    x: np.ndarray,
    span: float = 0.5,
    power: int = 3,
) -> np.ndarray:
    """
    Moving average filter with tricube weights for a time series.

    Port of R limma's ``tricubeMovingAverage``.

    Parameters
    ----------
    x : array_like
        1-D series.
    span : float, default 0.5
        Fraction of ``x`` to include in the smoothing window (clamped to
        ``(0, 1]``; values <= 0 return ``x`` unchanged).
    power : int, default 3
        Power applied to the tricube weights.

    Returns
    -------
    ndarray
        Smoothed series of the same length as ``x``.
    """
    import warnings

    if hasattr(span, "__len__") and len(span) > 1:
        warnings.warn("only first value of span used")
        span = span[0]
    if span > 1:
        span = 1.0
    x = np.asarray(x, dtype=np.float64)
    if span <= 0:
        return x.copy()

    if hasattr(power, "__len__") and len(power) > 1:
        warnings.warn("only first value of power used")
        power = power[0]
    if power < 0:
        power = 0

    n = x.size
    width_f = span * n
    hwidth = int(width_f) // 2
    width = 2 * hwidth + 1

    if width > n:
        width -= 2
        hwidth -= 1

    if hwidth <= 0:
        return x.copy()

    u = np.linspace(-1.0, 1.0, width) * width / (width + 1)
    weights = (1 - np.abs(u) ** 3) ** power
    weights = weights / np.sum(weights)

    # R's filter(..., convolution) centres the kernel on each point and
    # uses the kernel as-is (no flipping for symmetric tricube weights).
    # Extend the series with hwidth zeros on each side and convolve.
    z = np.zeros(hwidth)
    padded = np.concatenate([z, x, z])
    # np.convolve with mode='valid' produces length (len(padded) - width + 1)
    smoothed = np.convolve(padded, weights, mode="valid")
    # R indexes (hwidth+1):(n+hwidth) in 1-based -> 0:n in 0-based from
    # the valid-mode result (which starts at the position where the kernel
    # fits entirely, corresponding to R's output at index hwidth+1).
    out = smoothed[:n].copy()

    # Rescale boundary values to remove influence of outside zeros
    cw = np.cumsum(weights)
    # R: x[1:hwidth] / cw[(width-hwidth):(width-1)]  (1-based, inclusive)
    # 0-based: out[0:hwidth] /= cw[width-hwidth-1:width-1]
    out[:hwidth] = out[:hwidth] / cw[width - hwidth - 1: width - 1]
    # R: x[(n-hwidth+1):n] / cw[(width-1):(width-hwidth)]  (descending)
    # 0-based: out[n-hwidth:n] /= cw[width-2:width-hwidth-2:-1]
    # Construct the descending slice via index arithmetic.
    idx = np.arange(width - 2, width - hwidth - 2, -1)
    out[n - hwidth: n] = out[n - hwidth: n] / cw[idx]

    return out


# ---------------------------------------------------------------------------
# convest (port of R limma's convest)
# ---------------------------------------------------------------------------


def convest(
    p: np.ndarray,
    niter: int = 100,
    plot: bool = False,
    report: bool = False,
    tol: float = 1e-6,
    file: str = "",
) -> float:
    """
    Estimate pi0 using a convex decreasing density estimate.

    Port of R limma's ``convest``. The ``file`` argument controls where
    the diagnostic report is written when ``report=True``: the default
    empty string writes to stdout (matching R's ``file=""`` convention);
    any non-empty string is opened as a path.

    Parameters
    ----------
    p : array_like
        Observed p-values in [0, 1].
    niter : int, default 100
        Number of iterations of the EM-like algorithm.
    plot : bool, default False
        If True, draw the estimated density at each iteration.
    report : bool, default False
        If True, print a per-iteration diagnostic table.
    tol : float, default 1e-6
        Accuracy of the bisection search for the convex combination.
    file : str, default ""
        Destination for the report. Empty string -> stdout.

    Returns
    -------
    float
        Estimated proportion of true null hypotheses.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    if np.any(np.isnan(p)):
        raise ValueError("Missing values in p not allowed")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("All p-values must be between 0 and 1")

    k = int(niter)
    ny = float(tol)
    p = np.sort(p)
    m = p.size
    p_c = np.ceil(100.0 * p) / 100.0
    p_f = np.floor(100.0 * p) / 100.0
    t_grid = np.arange(1, 101) / 100.0
    x_grid = np.arange(0, 101) / 100.0

    f_hat = np.ones(101)
    f_hat_p = np.ones(m)

    # theta.hat = 0.01 * which.max(apply(..., sum((2*(theta-p)*(p<theta)/theta^2))))
    def _argmax_theta(f_hat_p_cur):
        # For each theta in t_grid, compute sum((2*(theta-p)*(p<theta)/theta^2)/f_hat_p_cur)
        # but the initial theta.hat uses sum without division.
        vals = np.empty(100)
        for i, theta in enumerate(t_grid):
            mask = p < theta
            vals[i] = np.sum((2.0 * (theta - p[mask]) / theta ** 2) / f_hat_p_cur[mask])
        return 0.01 * (int(np.argmax(vals)) + 1)

    # Initial theta.hat with f_hat_p = 1
    theta_hat = _argmax_theta(np.ones_like(p))
    f_theta_hat = 2.0 * (theta_hat - x_grid) * (x_grid < theta_hat) / theta_hat ** 2
    f_theta_hat_p = 2.0 * (theta_hat - p) * (p < theta_hat) / theta_hat ** 2

    thetas = []

    # Report output stream
    report_fh = None
    if report:
        if file == "":
            report_fh = sys.stdout
            report_close = False
        else:
            report_fh = open(file, "w")
            report_close = True
        report_fh.write("j\tpi0\ttheta.hat\t\tepsilon\tD\n")

    pi_0_hat = 1.0
    for j in range(1, k + 1):
        f_hat_diff = f_hat_p - f_theta_hat_p
        if np.sum(f_hat_diff / f_hat_p) > 0:
            eps = 0.0
        else:
            l = 0.0
            u = 1.0
            while abs(u - l) > ny:
                eps = (l + u) / 2.0
                # Only include elements where f.hat.p > 0 (R parity)
                mask_pos = f_hat_p > 0
                denom = (1.0 - eps) * f_hat_p[mask_pos] + eps * f_theta_hat_p[mask_pos]
                if np.sum(f_hat_diff[mask_pos] / denom) < 0:
                    l = eps
                else:
                    u = eps

        f_hat = (1.0 - eps) * f_hat + eps * f_theta_hat
        pi_0_hat = f_hat[100]
        d = np.sum(f_hat_diff / f_hat_p)
        if report:
            report_fh.write(f"{j}\t{pi_0_hat}\t{theta_hat}\t{eps}\t{d}\n")

        # f_hat_p <- 100*(f_hat[100*p_f+1] - f_hat[100*p_c+1])*(p_c-p) + f_hat[100*p_c+1]
        idx_f = (100.0 * p_f).astype(int)
        idx_c = (100.0 * p_c).astype(int)
        f_hat_p = 100.0 * (f_hat[idx_f] - f_hat[idx_c]) * (p_c - p) + f_hat[idx_c]

        # Pick new theta.hat
        vals = np.empty(100)
        for i, theta in enumerate(t_grid):
            mask = p < theta
            vals[i] = np.sum(
                (2.0 * (theta - p[mask]) / theta ** 2) / f_hat_p[mask]
            )
        theta_hat = 0.01 * (int(np.argmax(vals)) + 1)
        f_theta_hat = 2.0 * (theta_hat - x_grid) * (x_grid < theta_hat) / theta_hat ** 2
        f_theta_hat_p = 2.0 * (theta_hat - p) * (p < theta_hat) / theta_hat ** 2

        # Check if the Unif[0,1]-density is the new f_theta_hat
        if np.sum(f_theta_hat_p / f_hat_p) < np.sum(1.0 / f_hat_p):
            theta_hat = 0.0
            f_theta_hat = np.ones(101)
            f_theta_hat_p = np.ones(m)

        if theta_hat not in thetas:
            thetas.append(theta_hat)
            thetas.sort()

        if plot:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot(x_grid, f_hat)
                ax.set_ylim(0, 1.2)
                ax.set_title(f"{round(pi_0_hat, 5):.5f}")
                theta_arr = np.asarray(thetas)
                ax.plot(
                    theta_arr,
                    f_hat[(100.0 * theta_arr).astype(int)],
                    "bo",
                )
                plt.show(block=False)
            except ImportError:
                pass

    if report and report_close:
        report_fh.close()

    return float(pi_0_hat)


# ---------------------------------------------------------------------------
# prop_true_null (port of R limma's propTrueNull)
# ---------------------------------------------------------------------------


def _prop_true_null_by_local_fdr(p: np.ndarray) -> float:
    n = p.size
    # R: i <- n:1L; sort(p, decreasing=TRUE); q <- pmin(n/i * p, 1)
    p_sorted = np.sort(p)[::-1]
    i = np.arange(n, 0, -1, dtype=np.float64)
    q = np.minimum(n / i * p_sorted, 1.0)
    n1 = n + 1
    return float(np.sum(i * q) / n / n1 * 2.0)


def _prop_true_null_by_mean_p(p: np.ndarray) -> float:
    n = p.size
    p_sorted = np.sort(p)
    i = np.arange(1, n + 1, dtype=np.float64)
    q = np.minimum(p_sorted, (i - 0.5) / n)
    return float(2.0 * np.mean(q))


def _prop_true_null_from_histogram(p: np.ndarray, nbins: int = 20) -> float:
    # bin <- c(-0.1, (1:nbins)/nbins)
    edges = np.concatenate([[-0.1], np.arange(1, nbins + 1) / nbins])
    # tab <- tabulate(cut(p, bin))  -> counts in each bin
    bin_counts = np.zeros(nbins, dtype=np.int64)
    # np.histogram uses (left, right] by default (include right edge), matching
    # R's cut() default right=TRUE.
    counts, _ = np.histogram(p, bins=edges)
    bin_counts[: counts.size] = counts
    # tail.means <- rev(cumsum(rev(bin.counts))/(1:nbins))
    rev_counts = bin_counts[::-1]
    cum = np.cumsum(rev_counts)
    denom = np.arange(1, nbins + 1, dtype=np.float64)
    tail_means = (cum / denom)[::-1]
    # index <- which(tail.means >= bin.counts)[1]
    mask = tail_means >= bin_counts
    idx = int(np.argmax(mask))  # argmax returns first True
    # Guard: if mask is all False, argmax returns 0 but mask[0] is False
    if not mask[idx]:
        return float("nan")
    if tail_means[0] == 0:
        return float("nan")
    return float(tail_means[idx] / tail_means[0])


def prop_true_null(
    p: np.ndarray,
    method: str = "lfdr",
    nbins: int = 20,
    **convest_kwargs,
) -> float:
    """
    Estimate the proportion of null p-values.

    Port of R limma's ``propTrueNull``. Dispatches to one of four
    estimators.

    Parameters
    ----------
    p : array_like
        Observed p-values.
    method : {"lfdr", "mean", "hist", "convest"}, default "lfdr"
        Estimator to use.
    nbins : int, default 20
        Number of histogram bins when ``method="hist"``.
    **convest_kwargs
        Extra keyword arguments passed to :func:`convest` when
        ``method="convest"``.

    Returns
    -------
    float
        Estimated proportion of true nulls.
    """
    p = np.asarray(p, dtype=np.float64)
    if method not in ("lfdr", "mean", "hist", "convest"):
        raise ValueError(
            f"method '{method}' not recognized. "
            "Must be one of 'lfdr', 'mean', 'hist', 'convest'."
        )
    if method == "lfdr":
        return _prop_true_null_by_local_fdr(p)
    if method == "mean":
        return _prop_true_null_by_mean_p(p)
    if method == "hist":
        return _prop_true_null_from_histogram(p, nbins=nbins)
    return convest(p, **convest_kwargs)


# ---------------------------------------------------------------------------
# detection_p_values (port of R limma's detectionPValues.default)
# ---------------------------------------------------------------------------


def detection_p_values(
    x: np.ndarray,
    status: np.ndarray | list,
    negctrl: str = "negative",
) -> np.ndarray:
    """
    Detection p-values from negative controls.

    Port of R limma's ``detectionPValues.default`` (matrix path). Returns
    a matrix the same shape as ``x`` containing per-probe detection
    p-values computed from the empirical distribution of the negative
    controls within each column.

    Parameters
    ----------
    x : array_like
        Probe intensity matrix, shape (n_probes, n_arrays).
    status : array_like
        Per-probe status labels. Probes whose status equals ``negctrl``
        are treated as negative controls.
    negctrl : str, default "negative"
        Label identifying negative controls in ``status``.

    Returns
    -------
    ndarray
        Detection p-values with the same shape as ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    status = np.asarray(status)
    if status.size != x.shape[0]:
        raise ValueError("length of status must match nrow(x)")

    isneg = (status == negctrl).astype(np.int64)
    notneg = 1 - isneg
    nneg = int(np.sum(isneg))
    if nneg == 0:
        raise ValueError("No negative controls")

    n_probes, n_arrays = x.shape
    out = np.empty_like(x)
    for j in range(n_arrays):
        col = x[:, j]
        # R: order(x[,j], isneg, decreasing=TRUE)
        # stable sort by -x then by -isneg (because decreasing=TRUE for both)
        # Use lexsort: last key is primary.
        o1 = np.lexsort((-isneg, -col))
        # R: order(x[,j], notneg, decreasing=TRUE)
        o2 = np.lexsort((-notneg, -col))
        cs1 = np.empty(n_probes, dtype=np.int64)
        cs2 = np.empty(n_probes, dtype=np.int64)
        cs1[o1] = np.cumsum(isneg[o1])
        cs2[o2] = np.cumsum(isneg[o2])
        out[:, j] = cs1 + cs2

    return out / (2.0 * nneg)


# ============================================================================
# Simple utilities ported from R limma
# ============================================================================


def is_numeric(x) -> bool:
    """
    Test whether the argument is numeric.

    Port of R limma's ``isNumeric(x)`` (``utility.R``). True for numeric
    arrays / scalars, and for DataFrames whose columns are all numeric.
    """
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return False
        return bool(x.select_dtypes(include="number").shape[1] == x.shape[1])
    try:
        arr = np.asarray(x)
    except Exception:
        return False
    return np.issubdtype(arr.dtype, np.number)


def block_diag(*matrices) -> np.ndarray:
    """
    Build a block-diagonal matrix from the given inputs.

    Port of R limma's ``blockDiag(...)`` (``utility.R``). Delegates to
    ``scipy.linalg.block_diag`` while matching R's behaviour of raising
    when any argument is not 2-D.
    """
    from scipy.linalg import block_diag as _scipy_block_diag

    mats = []
    for m in matrices:
        arr = np.asarray(m)
        if arr.ndim != 2:
            raise ValueError("all arguments must be matrices")
        mats.append(arr)
    if not mats:
        return np.zeros((0, 0))
    return _scipy_block_diag(*mats)


def make_unique(x) -> np.ndarray:
    """
    Disambiguate a string vector by appending zero-padded indices to any
    value that repeats.

    Port of R limma's ``makeUnique(x)`` (``combine.R``). If a value
    occurs ``n`` times (``n > 1``) each occurrence is replaced with
    ``<value><k>`` where ``k`` is ``1..n`` left-padded to
    ``floor(log10(n)) + 1`` digits. Values that occur only once are
    unchanged.
    """
    x = np.asarray([str(v) for v in x], dtype=object)
    # Count occurrences, order by first appearance to match R's table().
    unique_vals, counts = np.unique(x, return_counts=True)
    for val, n in zip(unique_vals, counts):
        if n <= 1:
            continue
        width = 1 + int(np.floor(np.log10(n)))
        positions = np.where(x == val)[0]
        for k, pos in enumerate(positions, start=1):
            x[pos] = f"{val}{k:0{width}d}"
    return x


def logcosh(x) -> np.ndarray:
    """
    Compute ``log(cosh(x))`` without floating-point over/underflow.

    Port of R limma's ``logcosh`` (``logsumexp.R``).
    """
    x = np.asarray(x, dtype=np.float64)
    absx = np.abs(x)
    y = absx - np.log(2.0)
    # Small-argument Taylor
    small = absx < 1e-4
    y = np.where(small, 0.5 * x ** 2, y)
    # Mid-range direct
    mid = (~small) & (absx < 17.0)
    if np.any(mid):
        y = np.where(mid, np.log(np.cosh(np.where(mid, x, 0.0))), y)
    return y


def logsumexp(x, y=None) -> np.ndarray:
    """
    Compute ``log(exp(x) + exp(y))`` without floating over/underflow.

    Port of R limma's ``logsumexp(x, y)`` (``logsumexp.R``). When only
    one positional argument is given it reduces to the standard
    ``scipy.special.logsumexp`` of an array.
    """
    if y is None:
        from scipy.special import logsumexp as _lse
        return _lse(np.asarray(x, dtype=np.float64))

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x, y = np.broadcast_arrays(x, y)
    mi = np.minimum(x, y)
    ma = np.maximum(x, y)

    result = np.empty_like(ma)
    is_nan = np.isnan(ma)
    inf_ma = np.isposinf(ma)
    inf_mi = np.isneginf(mi)
    special = is_nan | inf_ma | inf_mi

    # Finite path
    finite = ~special
    if np.any(finite):
        m = (x[finite] + y[finite]) / 2.0
        result[finite] = m + logcosh(ma[finite] - m) + np.log(2.0)

    # Special handling matching R: Inf in ma -> Inf; -Inf in mi -> ma;
    # NaN -> propagate.
    result[is_nan] = np.nan
    result[inf_ma & ~is_nan] = np.inf
    both_inf = inf_mi & ~is_nan & ~inf_ma
    if np.any(both_inf):
        result[both_inf] = ma[both_inf]
    return result


def bwss(x, group) -> dict:
    """
    Between- and within-group sums of squares.

    Port of R limma's ``bwss(x, group)`` (``bwss.R``). Returns a dict
    with ``bss``, ``wss``, ``bdf`` (between-group df = n_groups - 1),
    and ``wdf`` (within-group df = sum(n_i - 1)).
    """
    x = np.asarray(x, dtype=np.float64)
    group = np.asarray(group)
    keep = ~(np.isnan(x) | pd.isna(group))
    x = x[keep]
    group = group[keep]
    if x.size == 0:
        return {"bss": np.nan, "wss": np.nan, "bdf": np.nan, "wdf": np.nan}

    unique, counts = np.unique(group, return_counts=True)
    means = np.array([np.mean(x[group == g]) for g in unique])
    # R uses ddof=1 sample variance; when n==1 variance is NaN
    variances = np.array([
        np.var(x[group == g], ddof=1) if counts[i] > 1 else np.nan
        for i, g in enumerate(unique)
    ])
    keep_grp = counts > 0
    means = means[keep_grp]; variances = variances[keep_grp]; counts = counts[keep_grp]

    total_n = counts.sum()
    grand_mean = np.sum(counts * means) / total_n
    wss = float(np.nansum((counts - 1) * variances))
    bss = float(np.sum(counts * (means - grand_mean) ** 2))
    wdf = int(np.sum(counts - 1))
    bdf = int(len(means) - 1)
    return {"bss": bss, "wss": wss, "bdf": bdf, "wdf": wdf}


def pool_var(var, df=None, multiplier=None, n=None) -> dict:
    """
    Pool sample variances allowing for unequal true variances via
    Satterthwaite's method.

    Port of R limma's ``poolVar(var, df=n-1, multiplier=1/n, n)``
    (``poolvar.R``). ``df`` and ``multiplier`` default to ``n-1`` and
    ``1/n`` respectively when ``n`` is supplied - matching R's default
    expressions.
    """
    var = np.atleast_1d(np.asarray(var, dtype=np.float64))
    if df is None:
        if n is None:
            raise ValueError("Either df or n must be provided")
        df = np.atleast_1d(np.asarray(n, dtype=np.float64)) - 1.0
    else:
        df = np.atleast_1d(np.asarray(df, dtype=np.float64))
    if multiplier is None:
        if n is None:
            raise ValueError("Either multiplier or n must be provided")
        multiplier = 1.0 / np.atleast_1d(np.asarray(n, dtype=np.float64))
    else:
        multiplier = np.atleast_1d(np.asarray(multiplier, dtype=np.float64))
    if np.min(multiplier) < 0:
        raise ValueError("Multipliers must be non-negative")
    if np.max(multiplier) == 0:
        return {"var": 0.0, "df": 0.0, "multiplier": 0.0}
    sm = float(np.sum(multiplier))
    mnorm = multiplier / sm
    pooled_var = float(np.sum(mnorm * var))
    denom = float(np.sum(mnorm ** 2 * var ** 2 / df))
    pooled_df = pooled_var ** 2 / denom if denom > 0 else float("inf")
    return {"var": pooled_var, "df": pooled_df, "multiplier": sm}


def cum_overlap(ol1, ol2) -> dict:
    """
    Cumulative-overlap analysis of two ordered ID lists.

    Port of R limma's ``cumOverlap(ol1, ol2)`` (``cumOverlap.R``).
    Returns a dict with ``n_total``, ``n_min``, ``p_min``,
    ``n_overlap``, ``id_overlap``, ``p_value``, ``adj_p_value``.
    """
    from scipy import stats as _stats

    ol1 = list(ol1)
    ol2 = list(ol2)
    if len(set(ol1)) != len(ol1):
        raise ValueError("Duplicate IDs found in ol1")
    if len(set(ol2)) != len(ol2):
        raise ValueError("Duplicate IDs found in ol2")

    set2 = set(ol2)
    ol1 = [v for v in ol1 if v in set2]
    set1 = set(ol1)
    ol2 = [v for v in ol2 if v in set1]

    ngenes = len(ol1)
    if ngenes == 0:
        return {"n_total": 0}

    rank_map = {v: i for i, v in enumerate(ol2)}
    m = np.array([rank_map[v] for v in ol1], dtype=np.int64)
    noverlap = np.empty(ngenes, dtype=np.int64)
    for j in range(ngenes):
        noverlap[j] = int(np.sum(m[: j + 1] <= j))

    i = np.arange(1, ngenes + 1, dtype=np.int64)
    # phyper(noverlap - 0.5, m=i, n=ngenes - i, k=i, lower.tail=FALSE)
    # = 1 - P(X <= noverlap - 1) under hypergeometric(good=i, bad=ngenes-i, drawn=i).
    p = 1.0 - _stats.hypergeom.cdf(
        np.maximum(noverlap - 1, -1), ngenes, i, i
    )

    p_b = p * i
    nmin = int(np.argmin(p_b))
    p_b = np.minimum(p_b, 1.0)
    id_overlap = [ol1[j] for j in range(nmin + 1) if m[j] <= nmin]

    return {
        "n_total": ngenes,
        "n_min": nmin + 1,
        "p_min": float(p_b[nmin]),
        "n_overlap": noverlap,
        "id_overlap": id_overlap,
        "p_value": p,
        "adj_p_value": p_b,
    }


def propexpr(
    x,
    neg_x=None,
    status=None,
    labels: tuple[str, str] = ("negative", "regular"),
) -> np.ndarray:
    """
    Estimate the proportion of expressed probes on each array.

    Port of R limma's ``propexpr`` (``propexpr.R``). Requires either
    explicit negative-control probes ``neg_x`` or a ``status`` vector
    that identifies ``labels[0]`` ("negative") and optionally
    ``labels[1]`` ("regular") probes.
    """
    x_mat = np.asarray(x, dtype=np.float64)
    if neg_x is None:
        if status is None:
            raise ValueError(
                "Either neg_x or status must be supplied so the "
                "negative-control probes can be located."
            )
        status_arr = np.asarray(status, dtype=object)
        sl = np.array([str(s).lower() for s in status_arr])
        ineg = np.array([labels[0].lower() in s for s in sl])
        if len(labels) > 1:
            ireg = np.array([labels[1].lower() in s for s in sl])
        else:
            ireg = ~ineg
        neg_mat = x_mat[ineg, :]
        x_mat = x_mat[ireg, :]
    else:
        neg_mat = np.asarray(neg_x, dtype=np.float64)

    narrays = x_mat.shape[1]
    pi1 = np.empty(narrays, dtype=np.float64)

    for i in range(narrays):
        b = neg_mat[:, i]
        b = b[~np.isnan(b)]
        r = x_mat[:, i]
        r = r[~np.isnan(r)]
        mu = np.mean(b)
        alpha = max(np.mean(r) - mu, 10.0)
        b1 = float(np.median(b))
        # Exponential CDF on (b1 - b) with rate = 1/alpha.
        diffs = b1 - b
        p1_i = float(np.mean(1.0 - np.exp(-np.clip(diffs, 0, None) / alpha)))
        pb_i = float((np.sum(b < b1) + np.sum(b == b1) / 2.0) / len(b))
        p_i = float((np.sum(r < b1) + np.sum(r == b1) / 2.0) / len(r))
        denom = pb_i - p1_i
        pi1[i] = (pb_i - p_i) / denom if denom != 0 else np.nan

    pi1 = np.clip(pi1, 0.0, 1.0)
    return pi1


def fit_gamma_intercept(y, offset=0.0, maxit: int = 1000) -> float:
    """
    Estimate the intercept of an additive gamma GLM.

    Port of R limma's ``fitGammaIntercept`` (``fitGammaIntercept.R``).
    Iterative root-find on
    ``sum(y / (offset + x)) - n = 0``.
    """
    y = np.asarray(y, dtype=np.float64)
    if np.any(y < 0):
        raise ValueError("negative y not permitted")
    offset = np.asarray(offset, dtype=np.float64)
    if np.any(offset < 0):
        raise ValueError("offsets must be positive")

    r0, r1 = float(np.min(offset)), float(np.max(offset))
    if r0 + 1e-14 > r1:
        return float(np.mean(y) - r0)

    n = y.size
    x = 0.0
    for _ in range(maxit):
        denom = offset + x
        Q = float(np.sum(y / denom))
        dQ = float(np.sum(y / denom ** 2))
        if dQ == 0:
            break
        dif = (Q - n) / dQ
        x = x + dif
        if abs(dif) < 1e-8:
            break
    return x


def bwss_matrix(x) -> dict:
    """
    Between- and within-column sums of squares of a matrix.

    Port of R limma's ``bwss.matrix(x)`` (``bwss.R``).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be a 2-D matrix")
    counts = np.sum(~np.isnan(x), axis=0)
    keep = counts > 0
    x = x[:, keep]
    counts = counts[keep]
    if counts.size == 0:
        return {"bss": np.nan, "wss": np.nan, "bdf": np.nan, "wdf": np.nan}

    means = np.nanmean(x, axis=0)
    # Sample variance per column (ddof=1); NaN-safe
    variances = np.array([
        np.nanvar(x[:, j], ddof=1) if counts[j] > 1 else np.nan
        for j in range(x.shape[1])
    ])
    total_n = counts.sum()
    grand_mean = np.sum(counts * means) / total_n
    wss = float(np.nansum((counts - 1) * variances))
    bss = float(np.sum(counts * (means - grand_mean) ** 2))
    wdf = int(np.sum(counts - 1))
    bdf = int(x.shape[1] - 1)
    return {"bss": bss, "wss": wss, "bdf": bdf, "wdf": wdf}

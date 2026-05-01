# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   squeezeVar.R               Copyright (C) 2004-2025 Gordon Smyth
#   fitFDist.R                 Copyright (C) 2002-2024 Gordon Smyth,
#                                                      Belinda Phipson
#   fitFDistRobustly.R         Copyright (C) 2012-2024 Gordon Smyth,
#                                                      Belinda Phipson
#   fitFDistUnequalDF1.R       Copyright (C) 2024-2025 Gordon Smyth,
#                                                      Lizhong Chen
# Python port: Copyright (C) 2026 John Mulvey
"""
Empirical Bayes variance shrinkage for pylimma.

Implements the core eBayes variance moderation from limma:
- fit_f_dist(): estimate prior variance and degrees of freedom
- squeeze_var(): compute posterior variances
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg
from scipy.optimize import brentq
from scipy.special import polygamma
from scipy.stats import f as f_dist

from .utils import logmdigamma, trigamma_inverse

# Pre-compute 128-point Gauss-Legendre quadrature nodes and weights
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(128)
# Transform from [-1, 1] to [0, 1] for uniform distribution
_GAUSS_NODES_UNIFORM = (_GAUSS_NODES + 1) / 2
_GAUSS_WEIGHTS_UNIFORM = _GAUSS_WEIGHTS / 2


def _winsorized_moments(
    df1: float,
    df2: float,
    winsor_tail_p: tuple[float, float],
) -> tuple[float, float]:
    """
    Compute theoretical mean and variance of Winsorized log-F distribution.

    Uses 128-point Gaussian quadrature to integrate over the F-distribution,
    matching R's fitFDistRobustly implementation.

    Parameters
    ----------
    df1 : float
        Numerator degrees of freedom.
    df2 : float
        Denominator degrees of freedom (can be np.inf).
    winsor_tail_p : tuple
        (lower, upper) tail proportions for Winsorization.

    Returns
    -------
    tuple
        (mean, variance) of the Winsorized log-F distribution.
    """
    prob_lower, prob_upper = winsor_tail_p

    # Quantiles of F-distribution at Winsorization points
    if np.isinf(df2):
        # F with df2=Inf is chi-squared(df1)/df1
        from scipy.stats import chi2

        fq_lower = chi2.ppf(prob_lower, df1) / df1
        fq_upper = chi2.ppf(1 - prob_upper, df1) / df1
    else:
        fq_lower = f_dist.ppf(prob_lower, df1, df2)
        fq_upper = f_dist.ppf(1 - prob_upper, df1, df2)

    # Log quantiles (Winsorization bounds for z = log(F))
    zq_lower = np.log(fq_lower)
    zq_upper = np.log(fq_upper)

    # Transform F quantiles using link function: q = f/(1+f)
    # This maps (0, inf) to (0, 1) for better numerical integration
    def linkfun(f):
        return f / (1 + f)

    def linkinv(q):
        return q / (1 - q)

    q_lower = linkfun(fq_lower)
    q_upper = linkfun(fq_upper)

    # Quadrature nodes in transformed space [q_lower, q_upper]
    q_range = q_upper - q_lower
    nodes_q = q_lower + q_range * _GAUSS_NODES_UNIFORM
    nodes_f = linkinv(nodes_q)
    nodes_z = np.log(nodes_f)

    # F-distribution density at nodes, with Jacobian for transformation
    # d/dq[linkinv(q)] = 1/(1-q)^2
    if np.isinf(df2):
        from scipy.stats import chi2

        # F = chi2/df1, so pdf_F(f) = df1 * pdf_chi2(f*df1)
        pdf_f = df1 * chi2.pdf(nodes_f * df1, df1)
    else:
        pdf_f = f_dist.pdf(nodes_f, df1, df2)

    # Jacobian: df/dq = 1/(1-q)^2
    jacobian = 1 / (1 - nodes_q) ** 2
    integrand_weight = pdf_f * jacobian

    # Compute mean: integral of z*f(z) over middle + contributions from tails
    # Middle part via quadrature
    middle_mean = q_range * np.sum(_GAUSS_WEIGHTS_UNIFORM * integrand_weight * nodes_z)
    # Tail contributions (Winsorized values times tail probabilities)
    tail_mean = zq_lower * prob_lower + zq_upper * prob_upper
    mean = middle_mean + tail_mean

    # Compute variance: E[(z-mean)^2]
    middle_var = q_range * np.sum(_GAUSS_WEIGHTS_UNIFORM * integrand_weight * (nodes_z - mean) ** 2)
    tail_var = (zq_lower - mean) ** 2 * prob_lower + (zq_upper - mean) ** 2 * prob_upper
    var = middle_var + tail_var

    return mean, var


def _natural_spline_basis(
    x: np.ndarray,
    df: int,
    intercept: bool = True,
    boundary_knots: tuple | None = None,
    interior_knots: np.ndarray | None = None,
    return_knots: bool = False,
):
    """
    Create natural cubic spline basis matrix matching R's splines::ns().

    Uses B-spline basis with natural boundary constraints (second derivative = 0
    at boundaries), matching R's implementation exactly.

    Parameters
    ----------
    x : ndarray
        Covariate values.
    df : int
        Degrees of freedom (number of basis functions).
    intercept : bool
        Whether to include intercept column.
    boundary_knots, interior_knots : optional
        Pre-computed knots. When supplied, the basis is evaluated at
        ``x`` using those knots (matching R's
        ``predict(ns_obj, newx=...)`` semantics). When None, knots are
        derived from ``x`` itself.
    return_knots : bool
        When True, also return the (boundary_knots, interior_knots)
        tuple alongside the basis so callers can re-evaluate at new x.

    Returns
    -------
    ndarray (or (ndarray, knots) tuple)
        Spline basis matrix of shape (len(x), df).
    """
    from scipy.interpolate import BSpline

    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    if df < 1:
        raise ValueError("df must be at least 1")

    # For df=1, just return intercept or constant
    if df == 1:
        basis = np.ones((n, 1))
        if return_knots:
            return basis, ((x.min(), x.max()), np.array([]))
        return basis

    # Number of interior knots (matching R's ns logic)
    n_interior = df - 1 - int(intercept)
    if n_interior < 0:
        n_interior = 0

    # Boundary knots: derive from x unless caller supplied them.
    if boundary_knots is None:
        boundary = (x.min(), x.max())
    else:
        boundary = boundary_knots

    # Handle edge case where all x values are identical
    if boundary[0] == boundary[1]:
        basis = np.ones((n, 1)) if df == 1 else np.column_stack([np.ones(n), np.zeros(n)])[:, :df]
        if return_knots:
            return basis, (boundary, np.array([]))
        return basis

    # Interior knots: derive from x unless caller supplied them.
    if interior_knots is not None:
        interior_knots = np.asarray(interior_knots, dtype=np.float64)
    elif n_interior > 0:
        knot_probs = np.linspace(0, 1, n_interior + 2)[1:-1]
        interior_knots = np.quantile(x, knot_probs)
    else:
        interior_knots = np.array([])

    # Augmented knot vector: boundary repeated 4 times (for cubic splines, order=4)
    # plus interior knots, matching R's: sort(c(rep(Boundary.knots, 4), knots))
    knots = np.sort(
        np.concatenate([np.repeat(boundary[0], 4), interior_knots, np.repeat(boundary[1], 4)])
    )

    # Number of B-spline basis functions
    n_basis = len(knots) - 4

    # Create B-spline basis matrix
    basis = np.zeros((n, n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        spline = BSpline(knots, c, k=3, extrapolate=True)
        basis[:, i] = spline(x)

    # Compute constraint matrix: second derivatives at boundary knots
    # For natural splines, second derivative must be 0 at boundaries
    const = np.zeros((2, n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        spline = BSpline(knots, c, k=3)
        spline_d2 = spline.derivative(2)
        const[0, i] = spline_d2(boundary[0])
        const[1, i] = spline_d2(boundary[1])

    # Remove intercept column before applying constraint if needed
    # (matching R's order of operations)
    if not intercept:
        const = const[:, 1:]
        basis = basis[:, 1:]

    # Apply natural spline constraint via QR decomposition
    # This matches R's: (t(qr.qty(qr.const, t(basis))))[, -(1:2)]
    Q, _ = linalg.qr(const.T, mode="full")
    transformed = Q.T @ basis.T
    basis_final = transformed.T[:, 2:]  # Drop first 2 columns (constrained directions)

    if return_knots:
        return basis_final, (boundary, interior_knots)
    return basis_final


def _fit_spline_trend(e: np.ndarray, covariate: np.ndarray, splinedf: int) -> tuple:
    """
    Fit spline trend to adjusted log-variances.

    Parameters
    ----------
    e : ndarray
        Adjusted log-variances (log(x) + logmdigamma(df1/2)).
    covariate : ndarray
        Covariate values (e.g., mean expression).
    splinedf : int
        Spline degrees of freedom.

    Returns
    -------
    tuple
        (fitted_values, residual_variance, coefficients, design,
        spline_knots). ``spline_knots`` is ``(boundary, interior)`` or
        None when the linear fallback fired; callers can replay the
        basis at new x via ``_natural_spline_basis(...,
        boundary_knots=..., interior_knots=...)``.
    """
    n = len(e)

    # Create spline basis. Capture the knots so callers can re-evaluate
    # the basis at new covariate points (matching R's
    # `predict(design, newx=...)` in fitFDist.R:90-97).
    spline_knots = None
    try:
        design, spline_knots = _natural_spline_basis(
            covariate, df=splinedf, intercept=True, return_knots=True
        )
    except Exception:
        # Fall back to simple linear fit
        design = np.column_stack([np.ones(n), covariate])

    # Fit linear model
    q, r = linalg.qr(design, mode="economic")
    rank = np.sum(np.abs(np.diag(r)) > 1e-10)

    # Solve for coefficients
    qty = q.T @ e
    coef = linalg.solve_triangular(r[:rank, :rank], qty[:rank])

    # Fitted values
    fitted = design[:, :rank] @ coef

    # Residual variance. With economic QR, effects has only `p` entries, so
    # residual SS is ||e||^2 - ||qty||^2 (the portion of e outside the column
    # space of design). Divide by n - rank (R's df.residual convention).
    df_resid = n - rank
    if df_resid > 0:
        residual_ss = np.sum(e**2) - np.sum(qty**2)
        # Guard against negative from floating-point noise near zero
        residual_var = max(residual_ss, 0.0) / df_resid
    else:
        residual_var = 0.0

    return fitted, residual_var, coef, design, spline_knots


def fit_f_dist(
    x: np.ndarray,
    df1: np.ndarray | float,
    covariate: np.ndarray | None = None,
) -> dict:
    """
    Fit a scaled F-distribution to sample variances.

    Estimates the scale factor (prior variance) and denominator degrees of
    freedom (prior df) by the method of moments on log(F). Port of R limma's
    fitFDist function (Gordon Smyth).

    Parameters
    ----------
    x : array_like
        Sample variances. Should be positive.
    df1 : array_like or float
        Numerator degrees of freedom for each variance (residual df).
    covariate : array_like, optional
        If provided, allows the scale to vary as a function of the covariate
        (e.g., mean expression). Uses natural cubic splines to fit the trend.

    Returns
    -------
    dict
        scale : float or ndarray
            Estimated prior variance (s0^2). If covariate is provided,
            this is an array of gene-specific prior variances.
        df2 : float
            Estimated prior degrees of freedom (d0).

    Notes
    -----
    Uses the relationship that if s^2 ~ s0^2 * F(d1, d0), then
    E[log(s^2)] = log(s0^2) + digamma(d1/2) - digamma(d0/2) + log(d0/d1)
    and Var[log(s^2)] = trigamma(d1/2) + trigamma(d0/2).

    When covariate is provided, the scale (s0^2) is allowed to vary as a
    smooth function of the covariate, typically the average log-expression.

    References
    ----------
    Smyth, G. K. (2004). Linear models and empirical Bayes methods for
    assessing differential expression in microarray experiments.
    Statistical Applications in Genetics and Molecular Biology, 3(1), Article 3.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    if n == 0:
        return {"scale": np.nan, "df2": np.nan}
    if n == 1:
        return {"scale": float(x[0]), "df2": 0.0}

    # Handle df1. R fitFDist.R:13-19 checks scalar df1 validity BEFORE
    # broadcasting; an invalid scalar returns NA/NA immediately rather
    # than falling through to per-element handling with an all-False
    # ok mask.
    df1 = np.asarray(df1, dtype=np.float64)
    if df1.ndim == 0:
        if not (np.isfinite(df1) and df1 > 1e-15):
            return {"scale": np.nan, "df2": np.nan}
        df1 = np.full(n, float(df1))
    elif len(df1) != n:
        # R fitFDist.R:21: `if(length(df1) != n) stop(...)`. Without
        # this check pylimma falls through to a numpy IndexError.
        raise ValueError("x and df1 have different lengths")

    # Check covariate
    if covariate is not None:
        covariate = np.asarray(covariate, dtype=np.float64)
        if len(covariate) != n:
            raise ValueError("x and covariate must be of same length")
        if np.any(np.isnan(covariate)):
            raise ValueError("NA covariate values not allowed")
        # Handle infinite covariate values
        finite_cov = np.isfinite(covariate)
        if not np.all(finite_cov):
            if np.any(finite_cov):
                cov_range = (np.min(covariate[finite_cov]), np.max(covariate[finite_cov]))
                covariate = covariate.copy()
                covariate[covariate == -np.inf] = cov_range[0] - 1
                covariate[covariate == np.inf] = cov_range[1] + 1
            else:
                covariate = np.sign(covariate)

    # Check for valid df1
    ok = np.isfinite(df1) & (df1 > 1e-15)
    if df1.size == 1 and not ok[0]:
        return {"scale": np.nan, "df2": np.nan}

    # Remove invalid values
    ok = ok & np.isfinite(x) & (x > -1e-15)
    nok = np.sum(ok)
    notallok = nok < n

    if nok == 1:
        return {"scale": float(x[ok][0]), "df2": 0.0}

    # Store indices for later expansion
    if notallok:
        x = x[ok]
        if len(df1) > 1:
            df1 = df1[ok]
        if covariate is not None:
            covariate_notok = covariate[~ok]
            covariate = covariate[ok]

    # Determine spline df for trend
    if covariate is not None:
        # Auto-select spline df based on sample size (like R)
        splinedf = 1 + int(nok >= 3) + int(nok >= 6) + int(nok >= 30)
        splinedf = min(splinedf, len(np.unique(covariate)))
        # If covariate has too few unique values, recall without covariate
        if splinedf < 2:
            result = fit_f_dist(x=x, df1=df1, covariate=None)
            result["scale"] = np.full(n, result["scale"])
            return result

    # Avoid exactly zero variances
    x = np.maximum(x, 0.0)
    m = np.median(x)
    if m == 0:
        warnings.warn("More than half of residual variances are exactly zero: eBayes unreliable")
        m = 1.0
    elif np.any(x == 0):
        warnings.warn("Zero sample variances detected, have been offset away from zero")
    x = np.maximum(x, 1e-5 * m)

    # Work on log scale
    z = np.log(x)
    e = z + logmdigamma(df1 / 2)

    if covariate is None:
        # Simple mean
        emean = np.mean(e)
        evar = np.var(e, ddof=1)
    else:
        # Fit spline trend
        emean, evar, coef, design, spline_knots = _fit_spline_trend(e, covariate, splinedf)

        # Expand emean to full length if needed. Reuse the spline_knots
        # captured during the fit so the basis is evaluated at the
        # not-ok covariate points using the SAME knots - matching R's
        # `design2 <- predict(design, newx=covariate.notok)`
        # (fitFDist.R:91). Rebuilding the basis from
        # ``covariate_notok`` would derive new boundary/interior
        # knots and yield a different function.
        if notallok:
            emean_full = np.zeros(n)
            emean_full[ok] = emean
            try:
                if spline_knots is not None:
                    design_notok = _natural_spline_basis(
                        covariate_notok,
                        df=splinedf,
                        intercept=True,
                        boundary_knots=spline_knots[0],
                        interior_knots=spline_knots[1],
                    )
                else:
                    # Linear fallback path: replay [1, x] design.
                    design_notok = np.column_stack([np.ones(len(covariate_notok)), covariate_notok])
                emean_full[~ok] = design_notok[:, : len(coef)] @ coef
            except Exception:
                # Fall back to nearest neighbor or mean
                emean_full[~ok] = np.mean(emean)
            emean = emean_full

    # Estimate scale and df2
    evar = evar - np.mean(polygamma(1, df1 / 2))  # subtract trigamma

    # R fitFDist.R: NaN evar (e.g. all-NaN var input) propagates as NA
    # df2 / NA scale; the `if(any(x==0))` and `if(evar > 0)` branches
    # raise "missing value where TRUE/FALSE needed" downstream. Mirror
    # that signal here so the caller's `anyNA(df.prior)` check fires.
    if np.isnan(evar):
        return {"scale": np.nan, "df2": np.nan}
    if evar > 0:
        df2 = 2 * trigamma_inverse(evar)
        s20 = np.exp(emean - logmdigamma(df2 / 2))
    else:
        df2 = np.inf
        if covariate is None:
            # Use simple pooled variance (MLE when df2 is infinite)
            s20 = np.mean(x)
        else:
            # Use trend-based estimate
            s20 = np.exp(emean)

    # Return appropriate type for scale
    if covariate is None:
        return {"scale": float(s20), "df2": float(df2)}
    else:
        return {"scale": s20, "df2": float(df2)}


def fit_f_dist_robustly(
    x: np.ndarray,
    df1: np.ndarray | float,
    covariate: np.ndarray | None = None,
    winsor_tail_p: tuple[float, float] = (0.05, 0.1),
    trace: bool = False,
) -> dict:
    """
    Robust estimation of scaled F-distribution parameters.

    Estimates the scale factor and denominator degrees of freedom using
    Winsorized moments of log(F) values, which provides robustness to
    outlier variances.

    Parameters
    ----------
    x : array_like
        Sample variances. Should be positive.
    df1 : array_like or float
        Numerator degrees of freedom for each variance.
    covariate : array_like, optional
        If provided, allows the scale to vary as a function of the covariate.
        Not yet fully implemented.
    winsor_tail_p : tuple of float, default (0.05, 0.1)
        Lower and upper tail proportions for Winsorization.

    Returns
    -------
    dict
        scale : float or ndarray
            Estimated prior variance (s0^2).
        df2 : float
            Estimated prior degrees of freedom (d0).
        df2_shrunk : ndarray
            Gene-wise shrunken prior df, accounting for outliers.

    Notes
    -----
    This function is more robust than fit_f_dist() when there are outlier
    variances. It uses Winsorization to limit the influence of extreme
    values on the moment estimates.

    The df2_shrunk values are shrunk towards the pooled df for genes
    identified as potential outliers (having unusually large variances).

    References
    ----------
    Phipson, B. and Smyth, G. K. (2016). Robust hyperparameter estimation
    protects against hypervariable genes and improves power to detect
    differential expression. Annals of Applied Statistics, 10(2), 946-963.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    # Handle edge cases
    if n < 2:
        return {"scale": np.nan, "df2": np.nan, "df2_shrunk": np.full(n, np.nan)}
    if n == 2:
        result = fit_f_dist(x=x, df1=df1, covariate=covariate)
        return {
            "scale": result["scale"],
            "df2": result["df2"],
            "df2_shrunk": np.full(n, result["df2"]),
        }

    # Handle df1
    df1 = np.asarray(df1, dtype=np.float64)
    if df1.ndim == 0:
        df1 = np.full(n, float(df1))

    # Filter valid observations
    ok = ~np.isnan(x) & np.isfinite(df1) & (df1 > 1e-6)
    if not np.all(ok):
        # Recurse on valid subset
        x_ok = x[ok]
        df1_ok = df1[ok] if len(df1) > 1 else df1
        cov_ok = covariate[ok] if covariate is not None else None

        sub_fit = fit_f_dist_robustly(
            x=x_ok, df1=df1_ok, covariate=cov_ok, winsor_tail_p=winsor_tail_p, trace=trace
        )

        # Expand results
        df2_shrunk = np.full(n, sub_fit["df2"])
        df2_shrunk[ok] = sub_fit["df2_shrunk"]

        if covariate is None:
            scale = sub_fit["scale"]
        else:
            scale = np.full(n, np.nan)
            scale[ok] = sub_fit["scale"]
            # Interpolate for non-ok values. R uses approxfun(rule=2):
            # linear interpolation within range, boundary-value clamp outside.
            if isinstance(sub_fit["scale"], np.ndarray):
                from scipy.interpolate import interp1d

                x_ok = covariate[ok]
                y_ok = np.log(sub_fit["scale"])
                sort_idx = np.argsort(x_ok, kind="stable")
                x_sorted = x_ok[sort_idx]
                y_sorted = y_ok[sort_idx]
                f = interp1d(
                    x_sorted,
                    y_sorted,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(y_sorted[0], y_sorted[-1]),
                )
                scale[~ok] = np.exp(f(covariate[~ok]))

        return {"scale": scale, "df2": sub_fit["df2"], "df2_shrunk": df2_shrunk}

    # Avoid zero or negative variances
    m = np.median(x)
    if m <= 0:
        raise ValueError("Variances are mostly <= 0")
    x = np.maximum(x, m * 1e-12)

    # Get non-robust estimates as baseline
    non_robust = fit_f_dist(x=x, df1=df1, covariate=covariate)

    # Check winsorization proportions
    prob_lower = winsor_tail_p[0]
    prob_upper = winsor_tail_p[1]
    if prob_lower < 1 / n and prob_upper < 1 / n:
        return {
            "scale": non_robust["scale"],
            "df2": non_robust["df2"],
            "df2_shrunk": np.full(n, non_robust["df2"]),
        }

    # Work with log scale
    z = np.log(x)

    # De-mean (or de-trend if covariate provided)
    if covariate is None:
        # Trimmed mean - R's mean(z, trim=t) trims t proportion from BOTH ends
        # Use winsor.tail.p[2] (prob_upper) as the trim proportion
        trim_frac = prob_upper
        lo = int(np.floor(n * trim_frac))
        hi = n - lo
        z_sorted = np.sort(z)
        z_trimmed = z_sorted[lo:hi] if lo < hi else z_sorted
        z_trend = np.mean(z_trimmed)
        z_resid = z - z_trend
    else:
        # Use LOWESS for trend. R calls `limma::loessFit(z, covariate, span=0.4)`
        # which dispatches to base R's `lowess()` when no weights are supplied.
        # R's lowess() defaults `delta = 0.01 * diff(range(x))` (skips
        # computation at nearby points); statsmodels defaults delta=0, which
        # disagrees with R by up to ~3e-3. Pass R's default explicitly.
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            delta = 0.01 * (np.max(covariate) - np.min(covariate))
            smoothed = lowess(z, covariate, frac=0.4, delta=delta, return_sorted=False)
            z_trend = smoothed
            z_resid = z - z_trend
        except ImportError:
            warnings.warn("statsmodels not available for LOWESS; using simple mean")
            z_trend = np.mean(z)
            z_resid = z - z_trend

    # Winsorize residuals
    q_lower = np.quantile(z_resid, prob_lower)
    q_upper = np.quantile(z_resid, 1 - prob_upper)
    z_wins = np.clip(z_resid, q_lower, q_upper)

    # Moments of Winsorized residuals
    z_wins_mean = np.mean(z_wins)
    z_wins_var = np.var(z_wins, ddof=1)
    if trace:
        print(f"Variance of Winsorized Fisher-z: {z_wins_var}")

    # Use constant df1 (take max if variable)
    if len(np.unique(df1)) > 1:
        df1_use = np.max(df1)
    else:
        df1_use = df1[0] if isinstance(df1, np.ndarray) else df1

    # Check if df2=Inf fits the data using Gaussian quadrature
    # Compute theoretical Winsorized variance for df2=Inf
    mom_inf = _winsorized_moments(df1_use, np.inf, (prob_lower, prob_upper))
    fun_val_inf = np.log(z_wins_var / mom_inf[1])

    if fun_val_inf <= 0:
        # df2 is effectively infinite
        df2 = np.inf

        # Correct trend for bias using theoretical Winsorized mean
        z_trend_corrected = z_trend + z_wins_mean - mom_inf[0]
        s20 = np.exp(z_trend_corrected)

        # Identify outliers
        f_stat = np.exp(z - z_trend_corrected)
        from scipy.stats import chi2

        tail_p = chi2.sf(f_stat * df1_use, df1_use)

        # Empirical tail probability
        r = np.argsort(np.argsort(f_stat)[::-1]) + 1  # rank from largest
        emp_tail_p = (r - 0.5) / n

        # Probability of not being an outlier
        prob_not_outlier = np.minimum(tail_p / emp_tail_p, 1.0)

        # Shrink df for outliers
        df_pooled = n * df1_use
        df2_shrunk = np.full(n, df2)
        outlier_mask = prob_not_outlier < 1
        if np.any(outlier_mask):
            df2_shrunk[outlier_mask] = prob_not_outlier[outlier_mask] * df_pooled
            # Make monotonic
            order = np.argsort(tail_p)
            df2_ordered = df2_shrunk[order]
            df2_shrunk[order] = np.maximum.accumulate(df2_ordered)

        return {
            "scale": s20,
            "df2": df2,
            "df2_shrunk": df2_shrunk,
            "tail_p_value": tail_p,  # R parity: diagnostic return
        }

    # Estimate df2 by matching Winsorized variance using root-finding
    # Start from non-robust estimate as lower bound
    if not np.isfinite(non_robust["df2"]):
        return {
            "scale": non_robust["scale"],
            "df2": non_robust["df2"],
            "df2_shrunk": np.full(n, non_robust["df2"]),
        }

    # Link function to map df2 from (0, inf) to (0, 1) for root-finding
    def linkfun(x):
        return x / (1 + x)

    def linkinv(x):
        return x / (1 - x)

    # Objective function: log(observed_var / theoretical_var)
    # We want to find df2 where this equals zero
    def objective(x_transformed):
        df2_try = linkinv(x_transformed)
        mom = _winsorized_moments(df1_use, df2_try, (prob_lower, prob_upper))
        return np.log(z_wins_var / mom[1])

    # Use non-robust estimate as lower bound
    x_lower = linkfun(non_robust["df2"])
    fun_val_lower = objective(x_lower)

    if fun_val_lower >= 0:
        # Non-robust estimate already satisfies the constraint
        df2 = non_robust["df2"]
    else:
        # Root is between non-robust estimate and infinity
        # Use brentq for root-finding
        try:
            x_root = brentq(objective, x_lower, 1.0 - 1e-10, xtol=1e-8)
            df2 = linkinv(x_root)
        except ValueError:
            # Fallback if root-finding fails
            df2 = non_robust["df2"]

    # Compute scale using corrected trend
    mom = _winsorized_moments(df1_use, df2, (prob_lower, prob_upper))
    z_trend_corrected = z_trend + z_wins_mean - mom[0]
    s20 = np.exp(z_trend_corrected)

    # Outlier detection and df shrinkage
    f_stat = np.exp(z - z_trend_corrected)
    tail_p = f_dist.sf(f_stat, df1_use, df2)

    # Empirical tail probability
    r = np.argsort(np.argsort(f_stat)[::-1]) + 1
    emp_tail_p = (r - 0.5) / n

    # Probability of being an outlier
    log_tail_p = np.log(np.maximum(tail_p, 1e-300))
    log_emp_tail_p = np.log(emp_tail_p)
    log_prob_not_outlier = np.minimum(log_tail_p - log_emp_tail_p, 0)
    prob_not_outlier = np.exp(log_prob_not_outlier)
    prob_outlier = 1 - prob_not_outlier

    # Compute df2 for outliers
    if np.any(log_prob_not_outlier < 0):
        min_log_tail_p = np.min(log_tail_p)
        if min_log_tail_p == -np.inf:
            df2_outlier = 0
            df2_shrunk = prob_not_outlier * df2
        else:
            # df2_outlier makes max F-stat the median of the distribution
            df2_outlier = np.log(0.5) / min_log_tail_p * df2
            # Iterate for accuracy (matching R's refinement step)
            new_log_tail_p = f_dist.logsf(np.max(f_stat), df1_use, df2_outlier)
            df2_outlier = np.log(0.5) / new_log_tail_p * df2_outlier
            df2_outlier = max(df2_outlier, 0)
            df2_shrunk = prob_not_outlier * df2 + prob_outlier * df2_outlier

        # Make monotonic in tail p-value
        order = np.argsort(log_tail_p)
        df2_ordered = df2_shrunk[order]
        # Cumulative minimum from smallest tail p
        cum_mean = np.cumsum(df2_ordered) / np.arange(1, n + 1)
        i_min = np.argmin(cum_mean)
        df2_ordered[: i_min + 1] = cum_mean[i_min]
        df2_shrunk[order] = np.maximum.accumulate(df2_ordered)
    else:
        df2_shrunk = np.full(n, df2)

    return {
        "scale": s20,
        "df2": df2,
        "df2_shrunk": df2_shrunk,
        "tail_p_value": tail_p,  # R parity: diagnostic return
    }


def fit_f_dist_unequal_df1(
    x: np.ndarray,
    df1: np.ndarray,
    covariate: np.ndarray | None = None,
    span: float | None = None,
    robust: bool = False,
    prior_weights: np.ndarray | None = None,
) -> dict:
    """
    Fit scaled F-distribution with unequal df1 values.

    Robust estimation of the parameters of a scaled F-distribution when df1
    varies substantially between observations. Uses maximum likelihood
    estimation with inverse-trigamma weighting.

    This version is preferred when df1 values are small or vary across genes,
    such as from edgeR quasi-likelihood pipelines.

    Parameters
    ----------
    x : array_like
        Sample variances. Should be positive.
    df1 : array_like
        Numerator degrees of freedom for each variance (can vary per gene).
    covariate : array_like, optional
        If provided, allows the scale to vary as a function of the covariate.
    span : float, optional
        LOWESS span for covariate trend. If None, chosen automatically.
    robust : bool, default False
        Use FDR-based outlier detection and re-weighting.
    prior_weights : array_like, optional
        Prior weights for each observation.

    Returns
    -------
    dict
        scale : float or ndarray
            Estimated prior variance (s0^2).
        df2 : float
            Estimated prior degrees of freedom (d0).
        df2_shrunk : ndarray, optional
            Gene-wise shrunken df2 (only if robust=True and outliers found).

    Notes
    -----
    Key differences from fit_f_dist:
    - Uses inverse-trigamma weighting: w = 1/trigamma(df1/2)
    - Uses maximum likelihood optimization instead of method of moments
    - Better handles small or varying df1 values

    References
    ----------
    Smyth, G. K. and Chen, L. (2024). limma package source code.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln

    x = np.asarray(x, dtype=np.float64)
    df1 = np.asarray(df1, dtype=np.float64)
    n = len(x)

    # Validate inputs
    if df1.ndim == 0:
        df1 = np.full(n, float(df1))
    if len(df1) != n:
        raise ValueError("x and df1 are different lengths")
    if np.any(np.isnan(df1)):
        raise ValueError("NA df1 values")

    # Check covariate
    if covariate is not None:
        covariate = np.asarray(covariate, dtype=np.float64)
        if len(covariate) != n:
            raise ValueError("x and covariate are different lengths")
        if np.any(np.isnan(covariate)):
            raise ValueError("covariate contains NA values")

    # Check prior_weights
    if prior_weights is not None:
        prior_weights = np.asarray(prior_weights, dtype=np.float64)
        if len(prior_weights) != n:
            raise ValueError("x and prior_weights are different lengths")
        if np.any(np.isnan(prior_weights)):
            raise ValueError("prior_weights contain NA values")
        if np.any(prior_weights < 0):
            raise ValueError("prior_weights are negative")

    # Handle NAs in x
    if np.any(np.isnan(x)):
        na_mask = np.isnan(x)
        if prior_weights is None:
            prior_weights = (~na_mask).astype(np.float64)
        else:
            prior_weights = prior_weights.copy()
            prior_weights[na_mask] = 0
        x = x.copy()
        x[na_mask] = 0

    # Treat small df1 values as uninformative
    if np.min(df1) < 0.01:
        small_df1 = df1 < 0.01
        if prior_weights is None:
            prior_weights = (~small_df1).astype(np.float64)
        else:
            prior_weights = prior_weights.copy()
            prior_weights[small_df1] = 0
        df1 = df1.copy()
        df1[small_df1] = 1

    has_prior_weights = prior_weights is not None

    # Check for informative observations
    informative = x > 0
    if has_prior_weights:
        informative = informative & (prior_weights > 0)
    n_informative = np.sum(informative)

    if n_informative < 2:
        return {"scale": np.nan, "df2": np.nan}
    if n_informative == 2:
        covariate = None
        robust = False
        prior_weights = None
        has_prior_weights = False

    # Avoid exactly zero x values
    m = np.median(x[informative])
    xpos = np.maximum(x, 1e-12 * m)

    # Work on log(F) scale
    z = np.log(xpos)

    # Average log(F) adjusted for df1
    d1 = df1 / 2
    e = z + logmdigamma(d1)

    # Inverse trigamma weights
    w = 1.0 / polygamma(1, d1)
    if has_prior_weights:
        w = w * prior_weights

    # Compute emean (mean or trend)
    if covariate is None:
        emean = np.sum(w * e) / np.sum(w)
    else:
        from .utils import choose_lowess_span, loess_fit

        if span is None:
            span = choose_lowess_span(n, small_n=500)
        # Normalize weights for LOWESS
        w_norm = w / np.quantile(w, 0.75)
        w_norm = np.clip(w_norm, 1e-8, 1e2)
        fit = loess_fit(e, covariate, weights=w_norm, span=span, iterations=1)
        emean = fit["fitted"]

    # Maximum likelihood optimization
    # Reparameterize: par = d2/(1+d2), so d2 = par/(1-par)
    d1x = d1 * xpos

    def neg_twice_log_lik(par):
        if par <= 0 or par >= 1:
            return np.inf
        d2 = par / (1 - par)
        d2s20 = d2 * np.exp(emean - logmdigamma(d2))
        # Ensure d2s20 is positive
        d2s20 = np.maximum(d2s20, 1e-100)
        # Log-likelihood terms
        ll = (
            -(d1 + d2) * np.log1p(d1x / d2s20) - d1 * np.log(d2s20) + gammaln(d1 + d2) - gammaln(d2)
        )
        if has_prior_weights:
            ll = prior_weights * ll
        return -2 * np.sum(ll)

    # Optimize (search in range corresponding to df2 from 1 to 5000)
    # Use R's default tolerance: .Machine$double.eps^0.25
    # This matters when the likelihood surface is flat (large df2)
    r_tol = np.finfo(float).eps ** 0.25
    result = minimize_scalar(
        neg_twice_log_lik, bounds=(0.5, 0.9998), method="bounded", options={"xatol": r_tol}
    )
    par_opt = result.x
    d2 = par_opt / (1 - par_opt)
    s20 = np.exp(emean - logmdigamma(d2))
    df2 = 2 * d2

    # Return non-robust result if robust=False
    if not robust:
        return {"scale": s20, "df2": df2}

    # Robust mode: FDR-based outlier detection
    from scipy.stats import f as f_dist

    from .utils import p_adjust

    f_stat = x / s20
    right_p = f_dist.sf(f_stat, df1, df2)
    left_p = 1 - right_p

    # Handle very small left_p
    small_left = left_p < 0.001
    if np.any(small_left):
        left_p[small_left] = f_dist.cdf(f_stat[small_left], df1[small_left], df2)

    two_sided_p = 2 * np.minimum(left_p, right_p)
    fdr = p_adjust(two_sided_p, method="BH")
    fdr[fdr > 0.3] = 1.0

    # If no outliers, return non-robust estimates
    if np.min(fdr) == 1:
        return {"scale": s20, "df2": df2}

    # Refit with FDR as prior weights
    refit = fit_f_dist_unequal_df1(
        x=x, df1=df1, covariate=covariate, span=span, robust=False, prior_weights=fdr
    )
    s20 = refit["scale"]
    df2 = refit["df2"]

    # Identify right outliers using QQ-type method
    r = np.argsort(np.argsort(f_stat)[::-1]) + 1  # rank from largest
    uniform_p = (n - r + 0.5) / n
    prob_not_outlier = np.minimum(right_p / uniform_p, 1.0)

    # If no right outliers, return robust estimates without df2 shrinkage
    if np.min(prob_not_outlier) == 1:
        return {"scale": s20, "df2": df2}

    # Compute shrunk df2 for outliers
    i_min = np.argmin(right_p)
    min_right_p = right_p[i_min]

    if min_right_p == 0:
        df2_outlier = 0.0
        df2_shrunk = prob_not_outlier * df2
    else:
        # Find df2_outlier to make max F-stat the median of distribution
        df2_outlier = np.log(0.5) / np.log(min_right_p) * df2
        # Iterate for accuracy
        new_log_right_p = f_dist.logsf(f_stat[i_min], df1[i_min], df2_outlier)
        if new_log_right_p != 0:
            df2_outlier = np.log(0.5) / new_log_right_p * df2_outlier
        df2_shrunk = prob_not_outlier * df2 + (1 - prob_not_outlier) * df2_outlier

    # Force df2_shrunk to be monotonic in right_p
    order = np.argsort(right_p)
    df2_ordered = df2_shrunk[order]
    m_cumsum = np.cumsum(df2_ordered) / np.arange(1, n + 1)
    i_min_cumsum = np.argmin(m_cumsum)
    df2_ordered[: i_min_cumsum + 1] = m_cumsum[i_min_cumsum]
    df2_shrunk[order] = np.maximum.accumulate(df2_ordered)

    return {"scale": s20, "df2": df2, "df2_outlier": df2_outlier, "df2_shrunk": df2_shrunk}


def _squeeze_var_core(
    var: np.ndarray,
    df: np.ndarray,
    var_prior: float | np.ndarray,
    df_prior: float | np.ndarray,
) -> np.ndarray:
    """
    Compute posterior variances given hyperparameters.

    Internal function implementing the Bayesian shrinkage formula.

    Parameters
    ----------
    var : ndarray
        Sample variances.
    df : ndarray
        Residual degrees of freedom.
    var_prior : float or ndarray
        Prior variance (s0^2).
    df_prior : float or ndarray
        Prior degrees of freedom (d0).

    Returns
    -------
    ndarray
        Posterior variances.
    """
    var = np.asarray(var, dtype=np.float64)
    df = np.asarray(df, dtype=np.float64)
    df_prior = np.asarray(df_prior, dtype=np.float64)
    var_prior = np.asarray(var_prior, dtype=np.float64)

    n = len(var)

    # Broadcast df if scalar
    if df.ndim == 0:
        df = np.full(n, float(df))

    # Check if all df_prior are finite
    if np.all(np.isfinite(df_prior)):
        # Standard shrinkage formula
        return (df * var + df_prior * var_prior) / (df + df_prior)

    # Handle case where some or all df_prior are infinite
    if var_prior.ndim == 0:
        var_post = np.full(n, float(var_prior))
    else:
        var_post = var_prior.copy()

    # Check if all df_prior are effectively infinite
    if np.all(df_prior > 1e100):
        return var_post

    # Mixed case: some finite, some infinite
    finite_mask = np.isfinite(df_prior)
    if np.any(finite_mask):
        df_i = df[finite_mask] if len(df) > 1 else df
        df_prior_i = df_prior[finite_mask] if df_prior.ndim > 0 else df_prior
        var_post[finite_mask] = (df_i * var[finite_mask] + df_prior_i * var_post[finite_mask]) / (
            df_i + df_prior_i
        )

    return var_post


def squeeze_var(
    var: np.ndarray,
    df: np.ndarray | float,
    covariate: np.ndarray | None = None,
    span: float | None = None,
    robust: bool = False,
    winsor_tail_p: tuple[float, float] = (0.05, 0.1),
    legacy: bool | None = None,
) -> dict:
    """
    Empirical Bayes posterior variances.

    Squeeze sample variances towards a common prior using empirical Bayes.
    This is the core variance moderation step in eBayes.

    Parameters
    ----------
    var : array_like
        Sample variances.
    df : array_like or float
        Residual degrees of freedom for each variance.
    covariate : array_like, optional
        Covariate for mean-variance trend (e.g., average log-expression).
        When provided, the prior variance is allowed to vary as a smooth
        function of the covariate.
    span : float, optional
        Span for lowess smoothing when fitting the mean-variance trend.
        Only used when legacy=False or when span is explicitly provided.
        If None, an appropriate span is chosen automatically.
    robust : bool, default False
        Use robust estimation of hyperparameters.
    winsor_tail_p : tuple of float, default (0.05, 0.1)
        Winsorization proportions for robust estimation.
    legacy : bool, optional
        If True, use the original limma hyperparameter estimation method
        (fit_f_dist or fit_f_dist_robustly). If False, use the newer method
        (fit_f_dist_unequal_df1) which handles unequal df1 values better.
        If None (default), auto-detect: use legacy=True if all df are equal,
        legacy=False otherwise. Setting span explicitly forces legacy=False.

    Returns
    -------
    dict
        var_post : ndarray
            Posterior variances.
        var_prior : float or ndarray
            Prior variance (s0^2).
        df_prior : float
            Prior degrees of freedom (d0).

    Notes
    -----
    The posterior variance is a weighted average of the sample variance and
    prior variance:

        var_post = (df * var + df_prior * var_prior) / (df + df_prior)

    References
    ----------
    Smyth, G. K. (2004). Linear models and empirical Bayes methods for
    assessing differential expression in microarray experiments.
    Statistical Applications in Genetics and Molecular Biology, 3(1), Article 3.
    """
    var = np.asarray(var, dtype=np.float64)
    n = len(var)

    if n == 0:
        raise ValueError("var is empty")

    # Not enough observations for empirical Bayes
    if n < 3:
        return {"var_post": var.copy(), "var_prior": var.copy(), "df_prior": 0.0}

    df = np.asarray(df, dtype=np.float64)
    if df.ndim == 0:
        df = np.full(n, float(df))

    # Guard against missing/infinite values when df==0
    if len(df) > 1:
        var = var.copy()
        var[df == 0] = 0.0

    # span is only implemented for new hyperparameter function
    if span is not None:
        legacy = False

    # Choose legacy or new hyperparameter method depending on whether
    # df are unequal. R squeezeVar.R:23-26: when df has no positive
    # entries, `identical(min(empty), max(empty))` is `identical(Inf,
    # -Inf)` which is FALSE, so R routes to the unequal-df1 fitter
    # (which then errors with "Could not estimate prior df").
    if legacy is None:
        df_pos = df[df > 0]
        if len(df_pos) > 0:
            legacy = np.min(df_pos) == np.max(df_pos)
        else:
            legacy = False

    # Estimate hyperparameters
    if legacy:
        if robust:
            fit = fit_f_dist_robustly(var, df1=df, covariate=covariate, winsor_tail_p=winsor_tail_p)
            df_prior = fit["df2"]
            df_prior_shrunk = fit["df2_shrunk"]
        else:
            fit = fit_f_dist(var, df1=df, covariate=covariate)
            df_prior = fit["df2"]
            df_prior_shrunk = None
    else:
        fit = fit_f_dist_unequal_df1(var, df1=df, covariate=covariate, span=span, robust=robust)
        df_prior_shrunk = fit.get("df2_shrunk")
        df_prior = df_prior_shrunk if df_prior_shrunk is not None else fit["df2"]

    if np.isscalar(df_prior) and np.isnan(df_prior):
        raise ValueError("Could not estimate prior df")
    elif not np.isscalar(df_prior) and np.any(np.isnan(df_prior)):
        raise ValueError("Could not estimate prior df")

    # Compute posterior variances
    if robust and df_prior_shrunk is not None:
        # Use gene-wise shrunken df_prior for outlier robustness
        # R's squeezeVar returns df2.shrunk as df.prior in robust mode
        var_post = _squeeze_var_core(
            var=var, df=df, var_prior=fit["scale"], df_prior=df_prior_shrunk
        )
        return {
            "df_prior": df_prior_shrunk,  # Per-gene values, matching R
            "var_prior": fit["scale"],
            "var_post": var_post,
        }
    else:
        var_post = _squeeze_var_core(var=var, df=df, var_prior=fit["scale"], df_prior=df_prior)
        return {"df_prior": df_prior, "var_prior": fit["scale"], "var_post": var_post}

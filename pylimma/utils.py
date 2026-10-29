# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   chooseLowessSpan.R         Copyright (C) 2020-2024 Gordon Smyth
#   weightedLowess.R, src/weighted_lowess.c
#                              Copyright (C) 2014-2020 Aaron Lun
#   loessFit.R                 Copyright (C) 2003-2015 Gordon Smyth
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
# _r_lowess() is a port of base R's lowess (not from limma), which limma's
# loessFit, voom and vooma call:
#   stats/src/lowess.c         Copyright (C) 1996 Robert Gentleman and Ross
#                              Ihaka; (C) 1999-2025 The R Core Team;
#                              GPL (>= 2)
#   stats/R/lowess.R           Copyright (C) 1995-2016 The R Core Team;
#                              GPL (>= 2)
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

import math
import numbers
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import optimize as _scipy_optimize
from scipy import sparse as _scipy_sparse
from scipy import stats as _scipy_stats
from scipy.special import betaln, digamma, ndtri_exp, polygamma


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
    # R: if(!is.numeric(x)) stop(...); logical, character and complex are not numeric
    if x.dtype.kind not in "iuf":
        raise TypeError("Non-numeric argument to mathematical function")
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
    with np.errstate(divide="ignore"):  # R's 1/0 is Inf without a warning
        y[small_mask] = 1.0 / x[small_mask]

    # Newton iteration for normal range
    if np.any(normal_mask):
        x_norm = x[normal_mask]
        # Initial guess: 1/trigamma(y) is approximately y - 0.5 for moderate y
        y_iter = 0.5 + 1.0 / x_norm

        # R: repeat { iter <- iter+1; ...; if(converged) break;
        #   if(iter > 50) { warning("Iteration limit exceeded"); break } }
        iteration = 0
        while True:
            iteration += 1
            tri = polygamma(1, y_iter)  # trigamma
            psigamma2 = polygamma(2, y_iter)  # tetragamma
            dif = tri * (1.0 - tri / x_norm) / psigamma2
            y_iter = y_iter + dif
            if np.max(-dif / y_iter) < 1e-8:
                break
            if iteration > 50:
                import warnings

                warnings.warn("Iteration limit exceeded")
                break

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


def _r_scalar(value):
    """A one-element array, list, tuple or Series as its element (R has no
    scalars: a length-1 vector is one); anything else unchanged."""
    if isinstance(value, (np.ndarray, list, tuple, pd.Series)) and np.size(value) == 1:
        return np.asarray(value).item()
    return value


def _whole_count(value, name: str) -> int:
    """A count argument (nrot, nsim, ...) as a whole number of at least 1.
    R fails obscurely below 1 or silently truncates a fraction (nrot gave
    meaningless p-values: random numbers recycled into a smaller rotation
    matrix, the fraction in the denominators); clear errors instead.
    Whole-number floats such as 1e4 (and one-element arrays) are accepted."""
    value = _r_scalar(value)
    if (
        not isinstance(value, numbers.Real)
        or isinstance(value, (bool, np.bool_))
        or not math.isfinite(value)
        or value != int(value)
    ):
        raise ValueError(f"{name} must be a whole number")
    count = int(value)
    if count < 1:
        raise ValueError(f"{name} must be at least 1")
    return count


def _r_nelder_mead(fun, x0, args=()):
    """scipy's Nelder-Mead with the settings of R's optim(method =
    "Nelder-Mead"): the initial simplex steps every coordinate by
    0.1 * max(|x0|) (0.1 when x0 is all zero), the search stops when the
    function values of the simplex lie within reltol * (|f(x0)| + reltol)
    of each other (reltol = sqrt(machine epsilon), fixed from the start, as
    optim's "scaled convergence tolerance"), and it allows 500 function
    evaluations (maxit). scipy's reflection and shrink rules are kept, so
    results are close to R's, not always identical."""
    from scipy.optimize import minimize

    x0 = np.asarray(x0, dtype=np.float64)
    step = 0.1 * np.max(np.abs(x0)) if np.any(x0 != 0) else 0.1
    simplex = np.vstack([x0, x0 + step * np.eye(x0.size)])
    reltol = np.sqrt(np.finfo(np.float64).eps)
    f0 = fun(x0, *args)
    if not np.isfinite(f0):
        raise ValueError("function cannot be evaluated at initial parameters")
    fatol = reltol * (abs(f0) + reltol)
    return minimize(
        fun,
        x0,
        args=args,
        method="Nelder-Mead",
        options={
            "initial_simplex": simplex,
            "xatol": np.inf,
            "fatol": fatol,
            "maxfev": 500,
            "maxiter": np.iinfo(np.int64).max,
            "adaptive": False,
        },
    )


def _block_labels(block, n: int) -> np.ndarray:
    """A block factor as an array, one label per array. R's block code has
    no check for missing labels and fails obscurely (chol in gls.series,
    model.matrix in duplicateCorrelation), so a missing label (None, NaN,
    pandas NA) raises a clear error."""
    block = np.asarray(block)
    if len(block) != n:
        raise ValueError("Length of block does not match number of arrays")
    if pd.isna(block).any():
        raise ValueError("block contains missing values; give every array a block label")
    return block


def _r_ppoints(n: int) -> np.ndarray:
    """R's ppoints(n): (1:n - a) / (n + 1 - 2a), a = 3/8 for n <= 10, else 1/2."""
    a = 3 / 8 if n <= 10 else 0.5
    return (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)


def _r_qf(p, df1: float, df2: float, lower_tail: bool = True) -> np.ndarray:
    """R's qf(p, df1, df2, lower.tail) (nmath qf.c), which takes infinite
    degrees of freedom (scipy's f.ppf gives NaN) and uses the chi-squared
    limits for a large df: qchisq(p, df1, lower.tail) / df1 when df1 <= df2
    and df2 > 4e5 (1 when df1 is infinite too), df2 / qchisq(p, df2,
    !lower.tail) when df1 > 4e5."""
    from scipy import stats

    # R recycles p, df1 and df2 against each other
    p, df1, df2 = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (p, df1, df2)))
    large_df2 = (df1 <= df2) & (df2 > 4e5)
    large_df1 = ~large_df2 & (df1 > 4e5)
    chi_lower, chi_upper = (stats.chi2.ppf, stats.chi2.isf)[:: 1 if lower_tail else -1]
    f_quantile = stats.f.ppf if lower_tail else stats.f.isf
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            large_df2,
            np.where(np.isinf(df1), 1.0, chi_lower(p, df1) / df1),
            np.where(large_df1, df2 / chi_upper(p, df2), f_quantile(p, df1, df2)),
        )


def _qq(y, quantile, plot_it, main, xlab, ylab, ylim, kwargs) -> dict:
    """The theoretical quantiles of a Q-Q plot of y (NaN dropped) under the
    distribution whose quantile function is ``quantile``, drawn as R's
    plot(x, y, main, xlab, ylab, ylim) when ``plot_it``; shared by qqt and
    qqf."""
    # R: y[!is.na(y)], a matrix read column by column
    y = np.asarray(y).ravel(order="F")
    y = y[~np.isnan(y)]
    n = len(y)
    if n == 0:
        raise ValueError("y is empty or all NaN")

    # R: quantile(ppoints(n))[order(order(y))], ties in the order of y
    x = quantile(_r_ppoints(n))[np.argsort(np.argsort(y, kind="stable"))]

    if plot_it:
        from .plotting import _require_matplotlib

        plt = _require_matplotlib()
        _fig, ax = plt.subplots()
        ax.plot(x, y, **{"marker": "o", "linestyle": "none", "markersize": 3, **kwargs})
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(main)

    return {"x": x, "y": y}


def qqt(
    y: np.ndarray,
    df: float = np.inf,
    plot_it: bool = True,
    main: str = "Student's t Q-Q Plot",
    xlab: str = "Theoretical Quantiles",
    ylab: str = "Sample Quantiles",
    ylim=None,
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
    main, xlab, ylab : str
        Title and axis labels, R's defaults.
    ylim : tuple, optional
        y-axis limits (R's default ``range(y)`` is matplotlib's own
        autoscaling here).
    **kwargs
        Additional arguments passed to matplotlib's plot function.

    Returns
    -------
    dict
        x : ndarray
            Theoretical quantiles.
        y : ndarray
            The input values with NaN dropped, in their original order (a
            matrix read column by column), as R.

    Examples
    --------
    >>> fit = e_bayes(lm_fit(expr, design))
    >>> qqt(fit["t"][:, 0], df=fit["df_total"][0])
    """
    from scipy import stats

    return _qq(y, lambda p: stats.t.ppf(p, df), plot_it, main, xlab, ylab, ylim, kwargs)


def qqf(
    y: np.ndarray,
    df1: float,
    df2: float,
    plot_it: bool = True,
    main: str = "F Distribution Q-Q Plot",
    xlab: str = "Theoretical Quantiles",
    ylab: str = "Sample Quantiles",
    ylim=None,
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
    main, xlab, ylab : str
        Title and axis labels, R's defaults.
    ylim : tuple, optional
        y-axis limits (R's default ``range(y)`` is matplotlib's own
        autoscaling here).
    **kwargs
        Additional arguments passed to matplotlib's plot function.

    Returns
    -------
    dict
        x : ndarray
            Theoretical quantiles.
        y : ndarray
            The input values with NaN dropped, in their original order (a
            matrix read column by column), as R.
    """
    return _qq(y, lambda p: _r_qf(p, df1, df2), plot_it, main, xlab, ylab, ylim, kwargs)


def _as_matrix(values) -> np.ndarray:
    """R's ``as.matrix`` of a vector or matrix: a vector is one column."""
    values = np.asarray(values)
    return values.reshape(-1, 1) if values.ndim == 1 else values


def _as_dense_array(x, dtype=None) -> np.ndarray:
    """In-memory dense ndarray, as R's as.matrix() for a sparse Matrix.

    scipy.sparse matrices, anndata backed or lazy datasets (``to_memory``)
    and dask arrays (``compute``) are loaded and densified first; a plain
    np.asarray would turn them into a 0-d object array. Masked cells of a
    numpy masked array become NaN (R's NA); np.asarray would drop the mask
    and use the hidden values. Only a callable ``to_memory`` / ``compute``
    marks a lazy array: pandas exposes columns as attributes, so a sample
    named "compute" is a Series, not a method.
    """
    for method in ("to_memory", "compute"):
        load = getattr(x, method, None)
        if callable(load):
            x = load()
    if _scipy_sparse.issparse(x):
        x = x.toarray()
    if isinstance(x, np.ma.MaskedArray):
        x = np.ma.filled(x.astype(np.float64), np.nan)
    return np.asarray(x, dtype=dtype)


def _r_as_character(values) -> np.ndarray:
    """Element-wise strings as R's as.character() gives them for the values
    that matter when comparing labels: logicals are "TRUE" / "FALSE", whole
    numbers have no decimals ("1", not "1.0"), other doubles have up to 15
    significant digits ("0.3" for 0.1 + 0.2) and non-finite doubles are
    "NaN", "Inf" and "-Inf". Strings are unchanged; None (R's NA) stays None,
    which never compares equal. Returns an object array.

    An approximation: R chooses scientific notation when it is shorter
    (1e5 is "1e+05" in R, "100000" here) and has further edge cases for
    doubles; both sides of a comparison should go through this helper.
    """

    def one(v):
        if v is None:
            return None
        if isinstance(v, (bytes, np.bytes_)):
            return v.decode()
        if isinstance(v, (bool, np.bool_)):
            return "TRUE" if v else "FALSE"
        if isinstance(v, (int, float, np.integer, np.floating)):
            v = float(v)
            if np.isnan(v):
                return "NaN"
            if np.isinf(v):
                return "Inf" if v > 0 else "-Inf"
            if v.is_integer():
                return str(int(v))
            return format(v, ".15g")
        return str(v)

    return np.array([one(v) for v in np.asarray(values, dtype=object).ravel()], dtype=object)


def _r_factor(values, *, nan_is_level: bool = False) -> tuple[list, np.ndarray]:
    """R's factor(values): the level labels and each value's level index
    (-1 for a missing value). A pandas Categorical keeps its categories;
    numbers sort numerically, values whose as.character() labels coincide
    (0.3 and 0.1 + 0.2) sharing a level; FALSE comes before TRUE; strings
    sort by code point (R uses the locale). None and pandas NA are
    missing; so is NaN unless ``nan_is_level`` and the values are numbers
    (as.factor() makes a numeric NaN a level, table() leaves it out; among
    strings NaN is pandas' missing value, R's NA)."""
    if isinstance(values, pd.Series) and isinstance(values.dtype, pd.CategoricalDtype):
        values = values.array
    if isinstance(values, pd.Categorical):
        levels = list(_r_as_character(np.asarray(values.categories, dtype=object)))
        return levels, np.asarray(values.codes, dtype=np.int64)
    raw = np.asarray(values, dtype=object).ravel()
    numbers = (int, float, np.integer, np.floating)
    present = [v for v in raw if v is not None and v is not pd.NA]
    numeric = bool(present) and all(
        isinstance(v, numbers) and not isinstance(v, (bool, np.bool_)) for v in present
    )
    nan_level = nan_is_level and numeric

    def is_missing(v):
        if v is None or v is pd.NA:
            return True
        return isinstance(v, (float, np.floating)) and np.isnan(v) and not nan_level

    missing = np.array([is_missing(v) for v in raw], dtype=bool)
    observed = raw[~missing]
    codes = np.full(raw.size, -1, dtype=np.int64)
    if numeric:
        floats = observed.astype(np.float64)
        labels = _r_as_character(floats)
        levels = list(dict.fromkeys(_r_as_character(np.unique(floats))))
    else:
        labels = _r_as_character(observed)
        levels = sorted(set(labels))
    position = {level: k for k, level in enumerate(levels)}
    codes[~missing] = [position[label] for label in labels]
    return levels, codes


def _r_trimmed_mean(x: np.ndarray, trim: float) -> float:
    """R's mean(x, trim=trim): the plain mean when trim <= 0, the median
    when trim >= 0.5, otherwise floor(n * trim) values dropped at each end."""
    n = len(x)
    if trim <= 0:
        return float(np.mean(x))
    if trim >= 0.5:
        return float(np.median(x))
    lo = int(np.floor(n * trim))
    return float(np.mean(np.sort(x)[lo : n - lo]))


def choose_lowess_span(
    n: int | np.ndarray = 1000,
    small_n: int = 50,
    min_span: float = 0.3,
    power: float = 1 / 3,
) -> float | np.ndarray:
    """
    Choose optimal span for LOWESS smoothing of variance trends.

    Larger spans are used for small datasets and smaller spans for larger
    datasets.

    Parameters
    ----------
    n : int or array_like, default 1000
        Number of data points; an array gives a span for each value, as
        R's ``pmin``.
    small_n : int, default 50
        Threshold below which maximum smoothing is used.
    min_span : float, default 0.3
        Minimum span for large datasets.
    power : float, default 1/3
        Power for scaling between small and large datasets.

    Returns
    -------
    float or ndarray
        Recommended LOWESS span parameter (an array for array ``n``).
    """
    # numpy arithmetic as R's doubles: n = 0 gives Inf, clamped to 1, and a
    # negative n NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        span = np.minimum(min_span + (1 - min_span) * np.divide(small_n, n) ** power, 1.0)
    return float(span) if np.ndim(span) == 0 else span


# Speed of the lowess port (_seq_sum, _r_lowest, _r_clowess, _r_lowess).
# It is about 3x slower than statsmodels' compiled lowess, which it
# replaced: 0.38 s against 0.12 s for 50,000 points at f = 0.7, and cyclic
# loess on 20,000 x 4 takes 1.77 s instead of 0.55 s ("fast"). About half
# of the time is the np.cumsum in _seq_sum, which keeps C's left-to-right
# order; the rest is the Python loop over fitted points and the window
# moving one point per step. It could be made faster, for example by
# compiling _r_lowest / _r_clowess (numba, Cython or a C extension), or by
# finding each window with a search instead of stepping. Any change must
# keep the results bit-identical to lowess.c: check it against R's
# lowess.c compiled with `clang -O2 -ffp-contract=off` on a grid with ties,
# constant x, tiny and huge scales, outliers, f > 1, delta = 0 and small n
# (the port is bit-identical to it on a 1,224-case grid), and not only against R,
# since R on macOS arm64 fuses multiply-adds (known_differences.rst).
def _seq_sum(values: np.ndarray) -> float:
    """Left-to-right sum, as a C loop adds. np.sum adds pairwise, which
    changes the last bits; in lowess fits through two points the
    robustness weights then follow that noise."""
    return float(np.cumsum(values)[-1]) if values.size else 0.0


def _r_lowest(x, y, xs, nleft, nright, userw, rw):
    """Port of lowest() in R's stats/src/lowess.c: the local fit at ``xs``
    over ``x[nleft:nright+1]`` (0-based), widened to the right to take in
    ties. Returns the fitted value, or None when all weights are zero.
    ``rw`` holds the robustness weights, used if ``userw``. Products are
    written out as in C (``t * t * t``, not ``t**3``, which calls pow())."""
    n = len(x)
    x_range = x[n - 1] - x[0]
    h = max(xs - x[nleft], x[nright] - xs)
    h9 = 0.999 * h
    h1 = 0.001 * h

    # C scans j = nleft, nleft+1, ... and stops at the first point right of
    # xs farther than h9; x is sorted, so that point lies at or before the
    # first x > xs + 2h, and the exact test runs on that segment only
    seg = slice(nleft, min(n, int(np.searchsorted(x, xs + 2 * h, side="right")) + 1))
    r = np.abs(x[seg] - xs)
    beyond = (r > h9) & (x[seg] > xs)
    nrt = nleft + (int(np.argmax(beyond)) if beyond.any() else r.size) - 1
    window = slice(nleft, nrt + 1)
    r = r[: nrt - nleft + 1]
    t = r / h
    cube = 1.0 - t * t * t
    tricube = cube * cube * cube
    weights = np.where(r <= h9, np.where(r <= h1, 1.0, tricube), 0.0)
    if userw:
        weights = weights * rw[window]
    a = _seq_sum(weights)
    if a <= 0.0:
        return None

    # weighted least squares, with the weights summing to 1
    weights = weights / a
    xw = x[window]
    if h > 0.0:
        a = _seq_sum(weights * xw)
        b = xs - a
        # C: w[j]*fsquare(x[j]-a), the square first
        c = _seq_sum(weights * ((xw - a) * (xw - a)))
        if np.sqrt(c) > 0.001 * x_range:
            # points are spread out enough to compute a slope
            b /= c
            weights = weights * (b * (xw - a) + 1.0)
    return _seq_sum(weights * y[window])


def _r_clowess(x, y, f, nsteps, delta):
    """Port of clowess() in R's stats/src/lowess.c: lowess on ``x`` sorted
    ascending, fitting every point more than ``delta`` beyond the last
    fitted one and interpolating linearly in between. Returns the fitted
    values in the same order."""
    n = len(x)
    if n < 2:
        return y.copy()
    ys = np.empty(n)
    rw = np.empty(n)
    # at least two, at most n points (f * n can overflow to inf)
    ns = max(2, int(min(n, f * n + 1e-7)))

    # robustness iterations
    for iteration in range(1, nsteps + 2):
        nleft, nright = 0, ns - 1
        last = -1  # index of the previous fitted point
        i = 0  # index of the current point
        while True:
            # move the window right while that brings the radius down
            if nright < n - 1 and x[i] - x[nleft] > x[nright + 1] - x[i]:
                nleft += 1
                nright += 1
                continue

            fitted = _r_lowest(x, y, x[i], nleft, nright, iteration > 1, rw)
            # all weights zero: copy over the value
            ys[i] = y[i] if fitted is None else fitted

            if last < i - 1:
                # skipped points: interpolate
                skipped = np.arange(last + 1, i)
                alpha = (x[skipped] - x[last]) / (x[i] - x[last])
                ys[skipped] = alpha * ys[i] + (1.0 - alpha) * ys[last]

            # ties with the fitted point take its value; the next point fitted
            # is the last one within delta of it
            last = i
            tie_end = int(np.searchsorted(x, x[last], side="right")) - 1
            ys[last + 1 : tie_end + 1] = ys[last]
            last = tie_end
            beyond_cut = int(np.searchsorted(x, x[i] + delta, side="right"))
            i = max(last + 1, beyond_cut - 1)
            if last >= n - 1:
                break

        residuals = y - ys
        # overall scale estimate
        scale = _seq_sum(np.abs(residuals)) / n
        # compute robustness weights, except the last time
        if iteration > nsteps:
            break
        abs_res = np.abs(residuals)
        ordered = np.sort(abs_res)
        m1 = n // 2
        if n % 2 == 0:
            cmad = 3.0 * (ordered[m1] + ordered[n - m1 - 1])
        else:
            cmad = 6.0 * ordered[m1]
        if cmad < 1e-7 * scale:
            # effectively zero
            break
        q = abs_res / cmad
        bisquare = (1.0 - q * q) * (1.0 - q * q)
        rw = np.where(abs_res <= 0.999 * cmad, bisquare, 0.0)
        rw[abs_res <= 0.001 * cmad] = 1.0
    return ys


def _r_lowess(x, y, f=2 / 3, iter=3, delta=None):
    """R's ``stats::lowess(x, y, f, iter, delta)``: returns ``x`` sorted
    (a stable sort, as R's ``order()``) and the fitted values in that order.
    ``delta`` defaults to 1% of the range of ``x``, as in R."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("'x' and 'y' lengths differ")
    if x.size == 0:
        raise ValueError("invalid input")
    if delta is None:
        delta = 0.01 * (np.max(x) - np.min(x))
    if not (np.isfinite(f) and f > 0):
        raise ValueError("'f' must be finite and > 0")
    # R: asInteger(iter) truncates before the check, so -0.5 runs as 0; it
    # is NA beyond the int range
    if not (np.isfinite(iter) and 0 <= int(iter) <= np.iinfo(np.int32).max):
        raise ValueError("'iter' must be finite and >= 0")
    if not (np.isfinite(delta) and delta >= 0):
        raise ValueError("'delta' must be finite and > 0")
    order = np.argsort(x, kind="stable")
    x_sorted = x[order]
    # as C: 0/0 in the weights where h = 0 is never used; NaN in y propagates
    with np.errstate(divide="ignore", invalid="ignore"):
        return x_sorted, _r_clowess(x_sorted, y[order], f, int(iter), delta)


def _r_approx_rule2(x, y, xout):
    """R's ``approx(x, y, xout, rule=2, ties=mean)$y`` (and ``approxfun``):
    linear interpolation, constant beyond the ends. As in R, pairs with NA
    are dropped, y is averaged over tied x with R's mean() (second pass only
    for a finite mean) and each value is approx1()'s
    ``y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]))``."""
    keep = ~(np.isnan(x) | np.isnan(y))
    y = y[keep]
    x_unique, tie_group, ties = np.unique(x[keep], return_inverse=True, return_counts=True)
    if x_unique.size < 2:
        raise ValueError("need at least two non-NA values to interpolate")
    y_mean = np.bincount(tie_group, weights=y) / ties
    with np.errstate(invalid="ignore"):
        correction = np.bincount(tie_group, weights=y - y_mean[tie_group]) / ties
        y_mean = np.where(np.isfinite(y_mean), y_mean + correction, y_mean)

        v = np.asarray(xout, dtype=np.float64)
        i = np.clip(np.searchsorted(x_unique, v, side="right") - 1, 0, x_unique.size - 2)
        j = i + 1
        t = (v - x_unique[i]) / (x_unique[j] - x_unique[i])
        values = y_mean[i] + (y_mean[j] - y_mean[i]) * t
    values = np.where(v == x_unique[i], y_mean[i], values)
    values = np.where(v == x_unique[j], y_mean[j], values)
    values = np.where(v < x_unique[0], y_mean[0], values)
    return np.where(v > x_unique[-1], y_mean[-1], values)


_LOWESS_THRESHOLD = 1e-7


def _lowess_find_seeds(x: np.ndarray, delta: float) -> np.ndarray:
    """Port of find_seeds() in limma's weighted_lowess.c.

    Returns the indices of the anchor points: the first and last points plus
    every point more than ``delta`` beyond the previous anchor.
    """
    npts = len(x)
    seeds = [0]
    last_pt = 0
    for pt in range(1, npts - 1):
        if x[pt] - x[last_pt] > delta:
            seeds.append(pt)
            last_pt = pt
    seeds.append(npts - 1)
    return np.asarray(seeds, dtype=np.int64)


def _lowess_find_limits(seeds, x, w, spanweight):
    """Port of find_limits() in limma's weighted_lowess.c."""
    npts = len(x)
    starts = np.empty(len(seeds), dtype=np.int64)
    ends = np.empty(len(seeds), dtype=np.int64)
    dists = np.empty(len(seeds), dtype=np.float64)
    for k, curpt in enumerate(seeds):
        left = right = int(curpt)
        curw = w[curpt]
        at_end = curpt == npts - 1
        at_start = curpt == 0
        mdist = 0.0
        while curw < spanweight and (not at_end or not at_start):
            if at_end:
                left -= 1
                curw += w[left]
                if left == 0:
                    at_start = True
                mdist = max(mdist, x[curpt] - x[left])
            elif at_start:
                right += 1
                curw += w[right]
                if right == npts - 1:
                    at_end = True
                mdist = max(mdist, x[right] - x[curpt])
            else:
                ldist = x[curpt] - x[left - 1]
                rdist = x[right + 1] - x[curpt]
                if ldist < rdist:
                    left -= 1
                    curw += w[left]
                    if left == 0:
                        at_start = True
                    mdist = max(mdist, ldist)
                else:
                    right += 1
                    curw += w[right]
                    if right == npts - 1:
                        at_end = True
                    mdist = max(mdist, rdist)
        # Extend the window to include ties at either end.
        while left > 0 and x[left] == x[left - 1]:
            left -= 1
        while right < npts - 1 and x[right] == x[right + 1]:
            right += 1
        starts[k], ends[k], dists[k] = left, right, mdist
    return starts, ends, dists


def _lowess_fit_point(x, y, w, rw, curpt, left, right, dist):
    """Port of lowess_fit() in limma's weighted_lowess.c."""
    xs, ys = x[left : right + 1], y[left : right + 1]
    if dist < _LOWESS_THRESHOLD:
        work = w[left : right + 1] * rw[left : right + 1]
        return np.sum(ys * work) / np.sum(work)
    work = (1 - (np.abs(x[curpt] - xs) / dist) ** 3) ** 3 * w[left : right + 1] * rw[left : right + 1]
    allweight = np.sum(work)
    xmean = np.sum(work * xs) / allweight
    ymean = np.sum(work * ys) / allweight
    var = np.sum((xs - xmean) ** 2 * work)
    if var < _LOWESS_THRESHOLD:
        return ymean
    slope = np.sum((xs - xmean) * (ys - ymean) * work) / var
    return slope * x[curpt] + (ymean - slope * xmean)


def _weighted_lowess_sorted(x, y, w, span, iterations, delta):
    """Port of the C entry point weighted_lowess() in limma's weighted_lowess.c.

    ``x`` must be sorted ascending (``y`` and ``w`` in the same order).
    Returns (fitted, robustness_weights), both in sorted order.
    """
    npts = len(x)
    if npts < 2:
        raise ValueError("need at least two points")
    if iterations <= 0:
        raise ValueError("number of robustness iterations should be positive")

    totalweight = np.sum(w)
    spanweight = totalweight * span
    subrange = (x[npts - 1] - x[0]) / npts
    seeds = _lowess_find_seeds(x, delta)
    starts, ends, dists = _lowess_find_limits(seeds, x, w, spanweight)

    fitted = np.empty(npts, dtype=np.float64)
    robustness = np.ones(npts, dtype=np.float64)
    for _ in range(iterations):
        fitted[0] = _lowess_fit_point(x, y, w, robustness, 0, starts[0], ends[0], dists[0])
        last_pt = 0
        for k in range(1, len(seeds)):
            pt = seeds[k]
            fitted[pt] = _lowess_fit_point(x, y, w, robustness, pt, starts[k], ends[k], dists[k])
            if pt - last_pt > 1:
                between = slice(last_pt + 1, pt)
                current = x[pt] - x[last_pt]
                if current > _LOWESS_THRESHOLD * subrange:
                    slope = (fitted[pt] - fitted[last_pt]) / current
                    fitted[between] = slope * x[between] + (fitted[pt] - slope * x[pt])
                else:
                    fitted[between] = 0.5 * (fitted[pt] + fitted[last_pt])
            last_pt = pt

        # Weighted median of the absolute residuals.
        abs_resid = np.abs(y - fitted)
        resid_scale = np.mean(abs_resid)
        order = np.argsort(abs_resid, kind="stable")
        sorted_resid = abs_resid[order]
        cumulative = np.cumsum(w[order])
        halfweight = totalweight / 2
        cmad = 0.0
        for pt in range(npts):
            if cumulative[pt] == halfweight:
                cmad = 3 * (sorted_resid[pt] + sorted_resid[pt + 1])
                break
            if cumulative[pt] > halfweight:
                cmad = 6 * sorted_resid[pt]
                break

        if cmad <= _LOWESS_THRESHOLD * resid_scale:
            break
        robustness = np.where(abs_resid < cmad, (1 - (abs_resid / cmad) ** 2) ** 2, 0.0)

    return fitted, robustness


def loess_fit(
    y: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray | None = None,
    span: float = 0.3,
    iterations: int = 4,
    min_weight: float = 1e-5,
    max_weight: float = 1e5,
    equal_weights_as_null: bool = True,
    method: str = "weightedLowess",
) -> dict:
    """
    Fast lowess fit for univariate x and y allowing for weights.

    Port of R limma's ``loessFit``. Uses base R's ``lowess()`` algorithm when
    ``weights`` is None (or all equal and ``equal_weights_as_null``), and
    :func:`weighted_lowess` otherwise.

    Parameters
    ----------
    y : array_like
        Response values.
    x : array_like
        Predictor values.
    weights : array_like, optional
        Observation weights.
    span : float, default 0.3
        Smoothing parameter (fraction of total weight used for each fit).
    iterations : int, default 4
        Number of fitting iterations (1 = no robustness iterations).
    min_weight, max_weight : float
        Bounds applied to the weights.
    equal_weights_as_null : bool, default True
        Treat weights that are all equal as if ``weights`` were None.
    method : {"weightedLowess"}, default "weightedLowess"
        Only R's default method is ported.

    Returns
    -------
    dict
        fitted : ndarray
        residuals : ndarray

    Notes
    -----
    R's ``method="locfit"`` and ``method="loess"`` delegate to the locfit
    package and ``stats::loess``; they are not ported and raise
    NotImplementedError.
    """
    if method != "weightedLowess":
        if method in ("locfit", "loess"):
            raise NotImplementedError(f"loess_fit method={method!r} is not ported")
        raise ValueError("method must be one of 'weightedLowess', 'locfit', 'loess'")

    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(y)
    if len(x) != n:
        raise ValueError("y and x have different lengths")
    fitted = np.full(n, np.nan)
    residuals = np.full(n, np.nan)

    obs = np.isfinite(y) & np.isfinite(x)
    x_obs = x[obs]
    y_obs = y[obs]
    n_obs = len(y_obs)
    if n_obs == 0:
        return {"fitted": fitted, "residuals": residuals}

    if span < 1 / n_obs:
        fitted[obs] = y_obs
        residuals[obs] = 0
        return {"fitted": fitted, "residuals": residuals}

    min_weight = max(min_weight, 0.0)

    if weights is not None:
        # R: length(weights) != n, and weights[obs] by position, so any shape
        # with n values (an n x 1 matrix) is accepted
        weights = np.asarray(weights, dtype=np.float64).ravel(order="F")
        if weights.size != n:
            raise ValueError("y and weights have different lengths")
        w_obs = np.nan_to_num(weights[obs], nan=0.0)
        w_obs = np.minimum(np.maximum(w_obs, min_weight), max_weight)
        if equal_weights_as_null and w_obs.max() - w_obs.min() < 1e-15:
            weights = None

    if weights is None:
        # R: lowess(x=xobs, y=yobs, f=span, iter=iterations-1)
        order = np.argsort(x_obs, kind="stable")
        fitted_obs = np.empty(n_obs)
        fitted_obs[order] = _r_lowess(x_obs, y_obs, f=span, iter=iterations - 1)[1]
        fitted[obs] = fitted_obs
        residuals[obs] = y_obs - fitted_obs
        return {"fitted": fitted, "residuals": residuals}

    n_weighted_obs = n_obs if min_weight > 0 else int(np.sum(w_obs > 0))
    if n_weighted_obs < 4 + 1 / span:
        if n_weighted_obs == 1:
            fitted_obs = np.full(n_obs, y_obs[w_obs > 0][0])
        else:
            # lm.wfit(cbind(1, x), y, w): the slope is aliased (NA in R) when
            # the positively weighted x values are all equal.
            positive = w_obs > 0
            xw = x_obs[positive]
            if np.all(xw == xw[0]):
                fitted_obs = np.full(n_obs, np.sum(w_obs * y_obs) / np.sum(w_obs))
            else:
                design = np.column_stack([np.ones(n_obs), x_obs])
                sqrt_w = np.sqrt(w_obs)
                coef = np.linalg.lstsq(design * sqrt_w[:, None], y_obs * sqrt_w, rcond=None)[0]
                fitted_obs = design @ coef
        fitted[obs] = fitted_obs
        residuals[obs] = y_obs - fitted_obs
        return {"fitted": fitted, "residuals": residuals}

    fit = weighted_lowess(x_obs, y_obs, weights=w_obs, span=span, iterations=iterations, npts=200)
    fitted[obs] = fit["fitted"]
    residuals[obs] = fit["residuals"]
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
    Weighted lowess smoother.

    Port of R limma's ``weightedLowess`` and its C routine
    ``weighted_lowess.c``. Fits a degree-1 lowess curve at anchor points
    spaced by ``delta`` and interpolates linearly between them. Prior weights
    act as frequency weights in both the span calculation and the local
    regression.

    Parameters
    ----------
    x, y : array_like
        Covariate and response.
    weights : array_like, optional
        Non-negative prior weights. Default all 1.
    delta : float, optional
        Anchor-point spacing. If None, chosen so that about ``npts`` anchors
        span the covariate range.
    npts : int, default 200
        Target number of anchor points when ``delta`` is None.
    span : float, default 0.3
        Proportion of the total weight included in each local fit.
    iterations : int, default 4
        Number of fitting iterations (1 = no robustness iterations).
    output_style : {"loess", "lowess"}, default "loess"
        ``"loess"``: ``{"fitted", "residuals", "weights", "delta"}`` in the
        original point order. ``"lowess"``: ``{"x", "y", "delta"}`` sorted
        by ``x``.

    Returns
    -------
    dict
        Schema depends on ``output_style``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(y) != len(x):
        raise ValueError("x and y should have same length")
    if weights is None:
        weights = np.ones(len(y))
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != len(y):
            raise ValueError("weights should have same length as x and y")
    iterations = int(iterations)
    output_style = output_style.lower()
    if output_style not in ("loess", "lowess"):
        raise ValueError(f"output_style must be 'loess' or 'lowess', got {output_style!r}")

    order = np.argsort(x, kind="stable")
    x_sorted = x[order]
    if delta is None:
        npts = int(npts + 0.5)
        if npts < 1:
            raise ValueError("number of points should be a positive integer")
        if npts >= len(x):
            delta = 0.0
        else:
            dx = np.sort(np.diff(x_sorted))
            cumrange = np.cumsum(dx)
            numclusters = np.arange(npts)
            delta = np.min(cumrange[len(dx) - numclusters - 1] / (npts - numclusters))
    delta = float(delta)

    fitted_sorted, robustness_sorted = _weighted_lowess_sorted(
        x_sorted, y[order], weights[order], span, iterations, delta
    )

    if output_style == "lowess":
        return {"x": x_sorted, "y": fitted_sorted, "delta": delta}
    fitted = np.empty(len(x))
    fitted[order] = fitted_sorted
    robustness = np.empty(len(x))
    robustness[order] = robustness_sorted
    return {"fitted": fitted, "residuals": y - fitted, "weights": robustness, "delta": delta}


def _match_arg(value: str, choices: tuple[str, ...], name: str) -> str:
    """R's ``match.arg``: exact match, otherwise a unique partial match."""
    if value in choices:
        return value
    matches = [c for c in choices if c.startswith(value)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"'{name}' should be one of " + ", ".join(f'"{c}"' for c in choices))


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


_REQUIRED = object()


def _noncentral(central, noncentral):
    """scipy distribution for an R p-function whose ncp may be missing (R
    calls a separate algorithm when it is given)."""
    return lambda *args: central(*args[:-1]) if args[-1] is None else noncentral(*args)


def _beta(shape1, shape2, ncp):
    if ncp is not None:
        raise ValueError("zscore: the noncentral beta distribution (ncp) is not supported")
    return _scipy_stats.beta(shape1, shape2)


def _nbinom(size, prob, mu):
    # R: pnbinom(q, size, prob, mu) takes prob or mu
    if (prob is None) == (mu is None):
        raise ValueError("zscore: specify 'prob' or 'mu' for nbinom, but not both")
    return _scipy_stats.nbinom(size, prob if mu is None else size / (size + np.asarray(mu)))


# R's p<distribution> functions that zscore calls: their parameters in R's
# order with R's defaults (_REQUIRED for none; None for an argument R
# treats as missing), and the scipy distribution they define. gamma goes
# to zscore_gamma, as its parameters are R's pgamma's.
_ZSCORE_DISTRIBUTIONS = {
    "norm": ((("mean", 0.0), ("sd", 1.0)), lambda mean, sd: _scipy_stats.norm(mean, sd)),
    "t": ((("df", _REQUIRED), ("ncp", None)), _noncentral(_scipy_stats.t, _scipy_stats.nct)),
    "f": (
        (("df1", _REQUIRED), ("df2", _REQUIRED), ("ncp", None)),
        _noncentral(_scipy_stats.f, _scipy_stats.ncf),
    ),
    "chisq": (
        (("df", _REQUIRED), ("ncp", None)),
        _noncentral(_scipy_stats.chi2, _scipy_stats.ncx2),
    ),
    "beta": ((("shape1", _REQUIRED), ("shape2", _REQUIRED), ("ncp", None)), _beta),
    "exp": ((("rate", 1.0),), lambda rate: _scipy_stats.expon(scale=1 / np.asarray(rate))),
    "binom": ((("size", _REQUIRED), ("prob", _REQUIRED)), _scipy_stats.binom),
    "pois": ((("lambda", _REQUIRED),), _scipy_stats.poisson),
    # scipy's hypergeom(M, n, N): M objects, n of them white, N drawn
    "hyper": (
        (("m", _REQUIRED), ("n", _REQUIRED), ("k", _REQUIRED)),
        lambda m, n, k: _scipy_stats.hypergeom(np.asarray(m) + n, m, k),
    ),
    "nbinom": ((("size", _REQUIRED), ("prob", None), ("mu", None)), _nbinom),
    # R counts the failures before the first success, scipy's geom the trials
    "geom": ((("prob", _REQUIRED),), lambda prob: _scipy_stats.nbinom(1, prob)),
    "unif": (
        (("min", 0.0), ("max", 1.0)),
        lambda low, high: _scipy_stats.uniform(low, np.asarray(high) - low),
    ),
    "logis": ((("location", 0.0), ("scale", 1.0)), _scipy_stats.logistic),
    "cauchy": ((("location", 0.0), ("scale", 1.0)), _scipy_stats.cauchy),
    "lnorm": (
        (("meanlog", 0.0), ("sdlog", 1.0)),
        lambda meanlog, sdlog: _scipy_stats.lognorm(sdlog, scale=np.exp(meanlog)),
    ),
    "weibull": (
        (("shape", _REQUIRED), ("scale", 1.0)),
        lambda shape, scale: _scipy_stats.weibull_min(shape, scale=scale),
    ),
}


def _bind_r_arguments(params, args, kwargs, function: str) -> list:
    """Match arguments to an R function's parameters as R does for exact
    names: named ones first, then the positional ones in order."""
    names = [name for name, _ in params]
    unknown = [name for name in kwargs if name not in names]
    if unknown:
        raise TypeError(f"unused argument {unknown[0]!r} for {function} (arguments: {names})")
    free = [name for name in names if name not in kwargs]
    if len(args) > len(free):
        raise TypeError(f"too many arguments for {function} (arguments: {names})")
    bound = {**dict(zip(free, args)), **kwargs}
    missing = [name for name, default in params if default is _REQUIRED and name not in bound]
    if missing:
        raise TypeError(f'argument "{missing[0]}" is missing for {function}, with no default')
    return [bound.get(name, default) for name, default in params]


def zscore(
    q: np.ndarray | float,
    distribution: str,
    *args,
    **kwargs,
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
        Distribution name, as R's ``p<distribution>`` function without
        the ``p``: ``"norm"``, ``"t"``, ``"f"``, ``"chisq"``, ``"beta"``,
        ``"exp"``, ``"gamma"``, ``"binom"``, ``"pois"``, ``"hyper"``,
        ``"nbinom"``, ``"geom"``, ``"unif"``, ``"logis"``, ``"cauchy"``,
        ``"lnorm"`` or ``"weibull"``. R's ``signrank``, ``wilcox``,
        ``tukey`` and the noncentral beta are not supported.
    *args, **kwargs
        The distribution's parameters with R's names, order and defaults
        (e.g. ``df1`` and ``df2`` for ``"f"``; ``m``, ``n`` and ``k`` for
        ``"hyper"``). Python reserves ``lambda``, so give ``"pois"`` its
        mean by position or as ``**{"lambda": 2}``.

    Returns
    -------
    ndarray
        Z-scores with the same shape as ``q``.

    Notes
    -----
    As in R, ``"hyper"`` maps the plain hypergeometric tail probabilities;
    :func:`zscore_hyper` is the mid-p version (R's ``zscoreHyper``).
    """
    from scipy.special import ndtri_exp

    if distribution == "gamma":
        return zscore_gamma(q, *args, **kwargs)
    if distribution not in _ZSCORE_DISTRIBUTIONS:
        raise ValueError(
            f"distribution {distribution!r} not supported. Must be one of "
            f"{['gamma', *_ZSCORE_DISTRIBUTIONS]}."
        )
    params, make_distribution = _ZSCORE_DISTRIBUTIONS[distribution]
    dist = make_distribution(*_bind_r_arguments(params, args, kwargs, f"p{distribution}"))

    # R's pdist recycles q and the parameters against each other
    pupper, plower = np.broadcast_arrays(dist.logsf(q), dist.logcdf(q))
    z = np.empty(pupper.shape)
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
        logp = _log_pgamma(q_flat[up] / scale[up], shape[up], upper=True)
        z_flat[up] = -ndtri_exp(logp)
    if np.any(~up):
        logp = _log_pgamma(q_flat[~up] / scale[~up], shape[~up], upper=False)
        z_flat[~up] = ndtri_exp(logp)
    return z


# Below exp(-644) (about 1e-280) a tail probability is near or past
# underflow, so its logarithm is computed directly
_LOG_TAIL_FLOOR = -644.0


def _log_pgamma(x: np.ndarray, a: np.ndarray, upper: bool) -> np.ndarray:
    """log of the gamma(a, 1) tail probability at x, as R's pgamma(log.p=TRUE):
    finite where the probability underflows. scipy's value is kept where it
    is representable; elsewhere the upper tail is exp(-x + a log x -
    lgamma(a)) times the Legendre continued fraction (x > a, evaluated by
    Lentz's method) and the lower tail exp(-x + a log x - lgamma(a + 1))
    times its power series."""
    from scipy.special import gammaln

    logp = _scipy_stats.gamma.logsf(x, a) if upper else _scipy_stats.gamma.logcdf(x, a)
    far = ~(logp > _LOG_TAIL_FLOOR) & (x > 0) & np.isfinite(x) & np.isfinite(a) & (a > 0)
    if not np.any(far):
        return logp
    xf, af = x[far], a[far]
    tiny, eps = 1e-300, 1e-16
    if upper:
        b = xf + 1.0 - af
        c = np.full_like(xf, 1.0 / tiny)
        d = 1.0 / b
        h = d.copy()
        for i in range(1, 10001):
            an = -i * (i - af)
            b = b + 2.0
            d = an * d + b
            d = np.where(np.abs(d) < tiny, tiny, d)
            c = b + an / c
            c = np.where(np.abs(c) < tiny, tiny, c)
            d = 1.0 / d
            step = d * c
            h = h * step
            if np.all(np.abs(step - 1.0) < eps):
                break
        logp[far] = -xf + af * np.log(xf) - gammaln(af) + np.log(h)
    else:
        term = np.ones_like(xf)
        total = term.copy()
        ap = af.copy()
        for _ in range(10000):
            ap = ap + 1.0
            term = term * xf / ap
            total = total + term
            if np.all(term < total * eps):
                break
        logp[far] = -xf + af * np.log(xf) - gammaln(af + 1.0) + np.log(total)
    return logp


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
# t tail probabilities on the log scale (R's pt / qt with log.p = TRUE)
# ---------------------------------------------------------------------------


# Below this log tail probability scipy's log tails lose accuracy and then
# underflow (exp(-745) is the smallest double), so the continued fraction
# is used.
_LOG_TAIL_SCIPY_FLOOR = -700.0


def _log_betainc(a, b, log_x, log1m_x) -> np.ndarray:
    """log I_x(a, b), the regularised incomplete beta function, on the log
    scale: x^a (1 - x)^b / (a B(a, b)) times the continued fraction of
    Numerical Recipes' ``betacf`` (modified Lentz). The fraction converges
    for x < (a + 1) / (a + b + 2), which holds in the far tails it is used
    for. ``log_x`` and ``log1m_x`` are log(x) and log(1 - x)."""
    a, b, log_x, log1m_x = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (a, b, log_x, log1m_x))
    )
    x = np.exp(log_x)
    tiny = 1e-300

    def nonzero(v):
        return np.where(np.abs(v) < tiny, tiny, v)

    c = np.ones_like(x)
    d = 1 / nonzero(1 - (a + b) * x / (a + 1))
    h = d.copy()
    for m in range(1, 10000):
        even = m * (b - m) * x / ((a - 1 + 2 * m) * (a + 2 * m))
        d = 1 / nonzero(1 + even * d)
        c = nonzero(1 + even / c)
        h *= d * c
        odd = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 1 + 2 * m))
        d = 1 / nonzero(1 + odd * d)
        c = nonzero(1 + odd / c)
        h *= d * c
        if np.all(np.abs(d * c - 1) < 1e-15):
            break
    return a * log_x + b * log1m_x - np.log(a) - betaln(a, b) + np.log(h)


def _t_logsf(t: np.ndarray, df: np.ndarray | float) -> np.ndarray:
    """R's ``pt(t, df, lower.tail = FALSE, log.p = TRUE)`` for t >= 0,
    accurate where the tail probability underflows: P(T > t) is
    I_x(df/2, 1/2) / 2 with x = df / (df + t^2), and for df = Inf the
    normal tail, as R."""
    t, df = np.broadcast_arrays(np.asarray(t, dtype=np.float64), np.asarray(df, dtype=np.float64))
    with np.errstate(divide="ignore"):
        out = np.array(_scipy_stats.t.logsf(t, df), dtype=np.float64)
        normal = np.isposinf(df)
        out[normal] = _scipy_stats.norm.logsf(t[normal])
    far = ~(out > _LOG_TAIL_SCIPY_FLOOR) & (t > 0) & np.isfinite(t) & ~normal
    if np.any(far):
        # log(t^2 / df), so that x = 1 / (1 + t^2 / df) and 1 - x do not overflow
        log_ratio = 2 * np.log(t[far]) - np.log(df[far])
        out[far] = np.log(0.5) + _log_betainc(
            df[far] / 2, 0.5, -np.logaddexp(0, log_ratio), -np.logaddexp(0, -log_ratio)
        )
    return out


def _f_logsf(f, df1, df2):
    """R's ``pf(f, df1, df2, lower.tail = FALSE, log.p = TRUE)``, accurate
    where the tail underflows: P(F > f) is I_x(df2/2, df1/2) with
    x = df2 / (df2 + df1 * f). Returns a float for scalar input."""
    f, df1, df2 = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (f, df1, df2)))
    with np.errstate(divide="ignore"):
        out = np.array(_scipy_stats.f.logsf(f, df1, df2), dtype=np.float64)
    far = ~(out > _LOG_TAIL_SCIPY_FLOOR) & (f > 0) & np.isfinite(f) & np.isfinite(df2)
    if np.any(far):
        # log(df1 * f / df2), so that x and 1 - x do not overflow
        log_ratio = np.log(df1[far]) + np.log(f[far]) - np.log(df2[far])
        out[far] = _log_betainc(
            df2[far] / 2, df1[far] / 2, -np.logaddexp(0, log_ratio), -np.logaddexp(0, -log_ratio)
        )
    return out if out.ndim else float(out)


def _inverse_log_tail(log_p, guess, log_tail, upper=True) -> np.ndarray:
    """Quantiles q with ``log_tail(q) == log_p``, R's q*(..., log.p = TRUE).

    ``guess`` is scipy's quantile from ``exp(log_p)``. It is kept where it
    reproduces ``log_p``; scipy's own round trip agrees to ~5e-11
    (relative) and its failures (underflow below exp(-745), quantiles capped
    at 1e100) are off by far more, so 1e-8 is a chosen cut-off between the
    two. Elsewhere q is found by solving for log q; it is Inf or 0, as in R,
    when the quantile is beyond the doubles.
    """
    log_p = np.asarray(log_p, dtype=np.float64)
    quantiles = np.array(guess, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        inexact = ~(np.abs(log_tail(quantiles) - log_p) <= 1e-8 * np.abs(log_p))
    redo = inexact & np.isfinite(log_p) & (log_p < np.log(0.5))
    if np.any(redo):
        bounds = np.log(np.finfo(np.float64).tiny), np.log(np.finfo(np.float64).max)
        # log_tail decreases in q for an upper tail and increases for a lower tail
        sign = 1.0 if upper else -1.0
        for i in np.flatnonzero(redo):
            def gap(log_q, target=log_p[i]):
                return sign * (float(log_tail(np.exp(log_q))) - target)

            if gap(bounds[1]) > 0:
                quantiles[i] = np.inf
            elif gap(bounds[0]) < 0:
                quantiles[i] = 0.0
            else:
                log_q = _scipy_optimize.brentq(gap, *bounds, xtol=1e-14, rtol=1e-15)
                quantiles[i] = np.exp(log_q)
    return quantiles


def _t_isf_log(log_p: np.ndarray, df: float) -> np.ndarray:
    """
    R's ``qt(log_p, df, lower.tail = FALSE, log.p = TRUE)``: the t quantile
    for a log upper-tail probability, accurate where scipy's ``t.isf``
    underflows or caps (see ``_inverse_log_tail``).

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
    log_p = np.asarray(log_p, dtype=np.float64)
    return _inverse_log_tail(
        log_p, _scipy_stats.t.isf(np.exp(log_p), df), lambda q: _t_logsf(q, df)
    )


# ---------------------------------------------------------------------------
# zscore_t (port of R limma's zscoreT and its dot-prefixed helpers)
# ---------------------------------------------------------------------------


def _zscore_t_quantile(x: np.ndarray, df: np.ndarray) -> np.ndarray:
    # qnorm(pt(abs(x), df, lower.tail=FALSE, log.p=TRUE),
    #       lower.tail=FALSE, log.p=TRUE) * sign(x)
    # qnorm(log_p, lower.tail=FALSE, log.p=TRUE) is -ndtri_exp(log_p)
    return -ndtri_exp(_t_logsf(np.abs(x), df)) * np.sign(x)


def _zscore_t_wallace(x: np.ndarray, df: np.ndarray) -> np.ndarray:
    return (df + 0.125) / (df + 0.375) * np.sqrt(df * np.log1p(x / df * x)) * np.sign(x)


def _zscore_t_bailey(x: np.ndarray, df: np.ndarray) -> np.ndarray:
    return (
        (df + 0.125)
        / (df + 1.125)
        * np.sqrt((df + 19.0 / 12.0) * np.log1p(x / (df + 1.0 / 12.0) * x))
        * np.sign(x)
    )


def _zscore_t_hill(x: np.ndarray, df: np.ndarray) -> np.ndarray:
    A = df - 0.5
    B = 48.0 * A * A
    z = A * np.log1p(x / df * x)
    z = (
        ((((-0.4 * z - 3.3) * z - 24.0) * z - 85.5) / (0.8 * z * z + 100.0 + B) + z + 3.0) / B + 1.0
    ) * np.sqrt(z)
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
                f"method '{method}' not recognized. Must be one of 'bailey', 'hill', 'wallace'."
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
    out[:hwidth] = out[:hwidth] / cw[width - hwidth - 1 : width - 1]
    # R: x[(n-hwidth+1):n] / cw[(width-1):(width-hwidth)]  (descending)
    # 0-based: out[n-hwidth:n] /= cw[width-2:width-hwidth-2:-1]
    # Construct the descending slice via index arithmetic.
    idx = np.arange(width - 2, width - hwidth - 2, -1)
    out[n - hwidth : n] = out[n - hwidth : n] / cw[idx]

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
            vals[i] = np.sum((2.0 * (theta - p[mask]) / theta**2) / f_hat_p_cur[mask])
        return 0.01 * (int(np.argmax(vals)) + 1)

    # Initial theta.hat with f_hat_p = 1
    theta_hat = _argmax_theta(np.ones_like(p))
    f_theta_hat = 2.0 * (theta_hat - x_grid) * (x_grid < theta_hat) / theta_hat**2
    f_theta_hat_p = 2.0 * (theta_hat - p) * (p < theta_hat) / theta_hat**2

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
            vals[i] = np.sum((2.0 * (theta - p[mask]) / theta**2) / f_hat_p[mask])
        theta_hat = 0.01 * (int(np.argmax(vals)) + 1)
        f_theta_hat = 2.0 * (theta_hat - x_grid) * (x_grid < theta_hat) / theta_hat**2
        f_theta_hat_p = 2.0 * (theta_hat - p) * (p < theta_hat) / theta_hat**2

        # Check if the Unif[0,1]-density is the new f_theta_hat
        if np.sum(f_theta_hat_p / f_hat_p) < np.sum(1.0 / f_hat_p):
            theta_hat = 0.0
            f_theta_hat = np.ones(101)
            f_theta_hat_p = np.ones(m)

        if theta_hat not in thetas:
            thetas.append(theta_hat)
            thetas.sort()

        if plot:
            from .plotting import _require_matplotlib

            plt = _require_matplotlib()
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
    # tab <- tabulate(cut(p, bin)): cut's bins are (left, right], so a value
    # on an edge counts in the bin below it; values outside the bins (and
    # NaN) are not counted
    bin_index = np.searchsorted(edges, p, side="left") - 1
    in_bins = (bin_index >= 0) & (bin_index < nbins)
    bin_counts = np.bincount(bin_index[in_bins], minlength=nbins)
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
        Observed p-values (a matrix is used as its values, as in R).
    method : {"lfdr", "mean", "hist", "convest"}, default "lfdr"
        Estimator to use.
    nbins : int, default 20
        Number of histogram bins when ``method="hist"``, a whole number of
        at least 1.
    **convest_kwargs
        Extra keyword arguments passed to :func:`convest` when
        ``method="convest"``.

    Returns
    -------
    float
        Estimated proportion of true nulls.

    Notes
    -----
    Missing p-values raise for ``"lfdr"`` and ``"mean"`` (R drops them in
    ``sort()`` and recycles the shorter vector, an arbitrary result) and
    for ``"convest"``, as R; ``"hist"`` leaves them out, as R.
    """
    # R: length(p) and sort(p) use a matrix's values
    p = np.asarray(p, dtype=np.float64).ravel()
    if method not in ("lfdr", "mean", "hist", "convest"):
        raise ValueError(
            f"method '{method}' not recognized. Must be one of 'lfdr', 'mean', 'hist', 'convest'."
        )
    if method in ("lfdr", "mean") and np.isnan(p).any():
        # R's sort() drops NA and its arithmetic then recycles the shorter
        # vector; convest's error instead ("hist" leaves NA out, as R)
        raise ValueError("Missing values in p not allowed")
    if method == "lfdr":
        return _prop_true_null_by_local_fdr(p)
    if method == "mean":
        return _prop_true_null_by_mean_p(p)
    if method == "hist":
        # R's 1:nbins goes down to 0 for nbins = 0 and truncates a fraction
        return _prop_true_null_from_histogram(p, nbins=_whole_count(nbins, "nbins"))
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
    x = _as_dense_array(x, np.float64)
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
    arrays / scalars, and for DataFrames whose columns are all numeric. As
    R's ``is.numeric``, logical, complex and categorical (factor) values
    are not numeric.
    """

    def numeric(dtype) -> bool:
        return (
            pd.api.types.is_numeric_dtype(dtype)
            and not pd.api.types.is_bool_dtype(dtype)
            and not pd.api.types.is_complex_dtype(dtype)
        )

    if isinstance(x, pd.DataFrame):
        return x.shape[1] > 0 and all(numeric(dtype) for dtype in x.dtypes)
    if hasattr(x, "dtype"):
        return numeric(x.dtype)
    try:
        return numeric(np.asarray(x).dtype)
    except (TypeError, ValueError):
        return False


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
    unchanged. Values are compared as R's ``as.character`` gives them, and
    missing values (None, NaN) stay missing (None), as R's NA; R fails
    with "NAs are not allowed in subscripted assignments" when there is
    also a repeated value.
    """
    values = np.asarray(x, dtype=object).ravel()
    missing = pd.isna(values)
    x = _r_as_character(values)
    x[missing] = None
    unique_vals, counts = np.unique(x[~missing].astype(str), return_counts=True)
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
    y = np.where(small, 0.5 * x**2, y)
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
    ``scipy.special.logsumexp`` of an array. ``x`` and ``y`` of different
    lengths raise, unless one is a single value; R recycles the shorter
    (see :doc:`/validation/known_differences`).
    """
    if y is None:
        from scipy.special import logsumexp as _lse

        return _lse(np.asarray(x, dtype=np.float64))

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size and 1 not in (x.size, y.size):
        # R's pmin / pmax recycle the shorter; a clear error instead
        raise ValueError(
            f"x and y must have the same length or one of them a single value, not "
            f"{x.size} and {y.size}"
        )
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
    variances = np.array(
        [np.var(x[group == g], ddof=1) if counts[i] > 1 else np.nan for i, g in enumerate(unique)]
    )
    keep_grp = counts > 0
    means = means[keep_grp]
    variances = variances[keep_grp]
    counts = counts[keep_grp]

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
    # R: as.vector(), a matrix read column by column
    var = np.asarray(var, dtype=np.float64).ravel(order="F")
    if df is None:
        if n is None:
            raise ValueError("Either df or n must be provided")
        df = np.asarray(n, dtype=np.float64).ravel(order="F") - 1.0
    else:
        df = np.asarray(df, dtype=np.float64).ravel(order="F")
    if multiplier is None:
        if n is None:
            raise ValueError("Either multiplier or n must be provided")
        multiplier = 1.0 / np.asarray(n, dtype=np.float64).ravel(order="F")
    else:
        multiplier = np.asarray(multiplier, dtype=np.float64).ravel(order="F")
    if np.min(multiplier) < 0:
        raise ValueError("Multipliers must be non-negative")
    if np.max(multiplier) == 0:
        return {"var": 0.0, "df": 0.0, "multiplier": 0.0}
    sm = float(np.sum(multiplier))
    mnorm = multiplier / sm
    pooled_var = np.sum(mnorm * var)
    # R divides directly: 0 / 0 (every variance 0) is NaN, a missing
    # variance gives NA, and only infinite df give Inf
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled_df = pooled_var**2 / np.sum(mnorm**2 * var**2 / df)
    pooled_var, pooled_df = float(pooled_var), float(pooled_df)
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
    # = P(X > noverlap - 1) under hypergeometric(good=i, bad=ngenes-i, drawn=i).
    # The upper tail directly, as R: 1 - cdf loses everything below ~1e-16.
    p = _stats.hypergeom.sf(np.maximum(noverlap - 1, -1), ngenes, i, i)

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
    x_mat = _as_dense_array(x, np.float64)
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
    ``sum(y / (offset + x)) - n = 0``. As in R, up to ``maxit + 1`` updates
    run and a warning is issued when the limit is reached; inputs with no
    root (e.g. all-zero ``y``) run to the limit and return ``-inf``.

    An empty ``y`` gives ``nan``, as in R. Unlike R, missing values and an
    undefined (0 / 0) Newton step raise a ``ValueError``, and ``offset`` must
    be a scalar or have the same length as ``y``.
    """
    y = np.asarray(y, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    if y.size == 0:
        # R: mean(numeric(0)) - offset through the constant-offset shortcut
        return float("nan")
    if np.isnan(y).any() or np.isnan(offset).any():
        raise ValueError("y and offset must not contain missing values")
    if offset.size not in (1, y.size):
        raise ValueError("offset must be a scalar or have the same length as y")
    if np.any(y < 0):
        raise ValueError("negative y not permitted")
    if np.any(offset < 0):
        raise ValueError("offsets must be positive")

    r0, r1 = float(np.min(offset)), float(np.max(offset))
    if r0 + 1e-14 > r1:
        return float(np.mean(y) - r0)

    n = y.size
    x = 0.0
    iteration = 0
    # IEEE arithmetic as in R: with no root, dif becomes -Inf and x stays there.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        while True:
            iteration += 1
            Q = np.sum(y / (offset + x))
            dQ = np.sum(y / (offset + x) ** 2)
            dif = (Q - n) / dQ
            if np.isnan(dif):
                # R stops here with "missing value where TRUE/FALSE needed"
                raise ValueError(
                    "Newton step is undefined (0/0): an offset of zero where y is zero"
                )
            x = x + dif
            if abs(dif) < 1e-8:
                break
            if iteration > maxit:
                warnings.warn("iteration limit maxit exceeded")
                break
    return float(x)


def bwss_matrix(x) -> dict:
    """
    Between- and within-column sums of squares of a matrix.

    Port of R limma's ``bwss.matrix(x)`` (``bwss.R``).
    """
    x = _as_dense_array(x, np.float64)
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
    variances = np.array(
        [np.nanvar(x[:, j], ddof=1) if counts[j] > 1 else np.nan for j in range(x.shape[1])]
    )
    total_n = counts.sum()
    grand_mean = np.sum(counts * means) / total_n
    wss = float(np.nansum((counts - 1) * variances))
    bss = float(np.sum(counts * (means - grand_mean) ** 2))
    wdf = int(np.sum(counts - 1))
    bdf = int(x.shape[1] - 1)
    return {"bss": bss, "wss": wss, "bdf": bdf, "wdf": wdf}

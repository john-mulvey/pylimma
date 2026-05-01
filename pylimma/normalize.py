# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   norm.R (normalizeBetweenArrays, normalizeQuantiles,
#          normalizeMedianValues)        Copyright (C) 2002-2016 Gordon Smyth
#   norm.R (normalizeCyclicLoess)        Copyright (C) 2010-2026 Yunshun Chen,
#                                                                Gordon Smyth
#   norm.R (normalizeVSN.default)        Copyright (C) 2010      Gordon Smyth
#   background-normexp.R                 Copyright (C) 2002-2015 Gordon Smyth,
#                                                                Jeremy Silver
#   src/normexp.c (saddle, m2loglik,
#          gm2loglik, hm2loglik)         Copyright (C) 2007-2024 Jeremy Silver,
#                                                                Gordon Smyth,
#                                                                Lizhong Chen
#   background.R                         Copyright (C) 2003-2024 Gordon Smyth
#   avearrays.R                          Copyright (C) 2010-2015 Gordon Smyth
#
# normexp_fit(method="rma") additionally ports the algorithm from the
# Bioconductor affy package (which limma's normexp.fit calls per array):
#   affy::bg.parameters                  Copyright (C) Rafael A. Irizarry et al.;
#                                        LGPL (>= 2.0)
#
# normalize_vsn() additionally ports the algorithm from the Bioconductor
# vsn package (which limma's normalizeVSN.default calls per matrix):
#   vsn/R/vsn2.R, vsn/src/vsn2.c         Copyright (C) 2002-2009 Wolfgang Huber;
#                                        contributions from Markus Ruschhaupt,
#                                        Dennis Kostka, David Kreil;
#                                        Artistic-2.0
#
# Python port: Copyright (C) 2026 John Mulvey
"""
Between-array normalization for pylimma.

Faithful port of the matrix-applicable methods in R limma's
``normalizeBetweenArrays`` (``limma/R/norm.R``):

- ``"none"``:        identity.
- ``"scale"``:       median-centered scaling (``normalizeMedianValues``).
- ``"quantile"``:    quantile normalization (``normalizeQuantiles``).
- ``"cyclicloess"``: cyclic LOESS normalization (``normalizeCyclicLoess``).

Two-channel-only methods (``Aquantile``, ``Gquantile``, ``Rquantile``,
``Tquantile``) are not ported - they require RGList/MAList input which is
out of scope under pylimma's AnnData / flat-array design (see the
``policy_data_class_wrappers`` memory entry).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm as _scipy_norm, rankdata
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

from .utils import choose_lowess_span
from .classes import get_eawp, put_eawp, EList, _is_anndata


_VALID_METHODS = ("none", "scale", "quantile", "cyclicloess")


def normalize_between_arrays(
    object,
    method: str | None = None,
    targets=None,
    cyclic_method: str = "fast",
    *,
    out_layer: str = "normalized",
    uns_key: str = "normalize_between_arrays",
    layer: str | None = None,
    **kwargs,
):
    """
    Normalize columns of an expression matrix between arrays.

    Parameters
    ----------
    object : ndarray
        Expression matrix, shape (n_genes, n_samples).
    method : {"none", "scale", "quantile", "cyclicloess"}, optional
        Normalization method. Defaults to ``"quantile"`` for matrix input
        (matches R's default for matrices).
    targets :
        Ignored for single-channel matrix input. Accepted for R signature
        parity.
    cyclic_method : str, default "fast"
        Sub-method for cyclic LOESS. Currently only ``"fast"`` is
        implemented.
    **kwargs
        Forwarded to the underlying method (``ties`` for quantile; ``span``,
        ``iterations``, ``adaptive_span`` for cyclic LOESS).

    Returns
    -------
    ndarray
        Normalized matrix, same shape as input.

    Notes
    -----
    Two-channel methods (``"Aquantile"``, ``"Gquantile"``, ``"Rquantile"``,
    ``"Tquantile"``) are not supported; pylimma operates on single-channel
    expression matrices only.
    """
    # Polymorphic input: ndarray / dict / EList / AnnData.
    original_input = object
    eawp = get_eawp(object, layer=layer)
    arr = np.asarray(eawp["exprs"], dtype=np.float64)

    if method is None:
        method = "quantile"

    if method not in _VALID_METHODS:
        raise ValueError(
            f"method '{method}' not applicable to single-channel data. "
            f"Choose one of {_VALID_METHODS}. Two-channel methods "
            "(Aquantile/Gquantile/Rquantile/Tquantile) are out of scope."
        )

    if method == "none":
        normalized = arr
    elif method == "scale":
        normalized = normalize_median_values(arr)
    elif method == "quantile":
        normalized = normalize_quantiles(arr, **kwargs)
    elif method == "cyclicloess":
        normalized = normalize_cyclic_loess(
            arr, method=cyclic_method, **kwargs,
        )
    else:
        raise AssertionError("unreachable")  # pragma: no cover

    return put_eawp(
        {"E": normalized},
        original_input,
        out_layer=out_layer,
        weights_layer=None,
        uns_key=uns_key,
        single_matrix=True,
    )


def normalize_median_values(x: np.ndarray) -> np.ndarray:
    """
    Scale columns so they have the same median.

    Faithful port of R limma's ``normalizeMedianValues``
    (``limma/R/norm.R:523-533``).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1 or x.shape[1] == 1:
        return x
    cmed = np.log(np.nanmedian(x, axis=0))
    cmed = np.exp(cmed - np.mean(cmed))
    return x / cmed[np.newaxis, :]


def normalize_quantiles(A: np.ndarray, ties: bool = True) -> np.ndarray:
    """
    Quantile-normalize columns of a matrix.

    Faithful port of R limma's ``normalizeQuantiles``
    (``limma/R/norm.R:468-509``). Handles missing values via
    interpolation.
    """
    A = np.asarray(A, dtype=np.float64).copy()
    if A.ndim == 1:
        return A
    n_rows, n_cols = A.shape
    if n_cols == 1:
        return A

    S = np.full((n_rows, n_cols), np.nan)
    O = np.full((n_rows, n_cols), -1, dtype=np.int64)
    nobs = np.full(n_cols, n_rows, dtype=np.int64)
    i_grid = np.arange(n_rows) / (n_rows - 1)

    for j in range(n_cols):
        col = A[:, j]
        isna = np.isnan(col)
        nobsj = int((~isna).sum())
        if nobsj < n_rows:
            nobs[j] = nobsj
            if nobsj <= 1:
                S[:, j] = col[~isna][0] if nobsj == 1 else np.nan
            else:
                obs_vals = col[~isna]
                obs_idx = np.where(~isna)[0]
                ix = np.argsort(obs_vals)
                src_pos = np.arange(nobsj) / (nobsj - 1)
                # R's approx default is rule=1: NA outside range. Since
                # i_grid[0]=0 == src_pos[0] and i_grid[-1]=1 == src_pos[-1],
                # no extrapolation occurs in practice.
                S[:, j] = np.interp(i_grid, src_pos, obs_vals[ix])
                O[obs_idx, j] = obs_idx[ix]
        else:
            ix = np.argsort(col)
            S[:, j] = col[ix]
            O[:, j] = ix

    # rowMeans without na.rm - rows with any NaN propagate
    m = np.mean(S, axis=1)

    out = A.copy()
    for j in range(n_cols):
        col = A[:, j]
        isna = np.isnan(col)
        if ties:
            # rankdata does not handle NaN; mask manually
            if np.any(isna):
                r = np.full(n_rows, np.nan)
                obs_idx = np.where(~isna)[0]
                r[obs_idx] = rankdata(col[obs_idx], method="average")
            else:
                r = rankdata(col, method="average")
        if nobs[j] < n_rows:
            obs_idx = np.where(~isna)[0]
            if ties:
                positions = (r[obs_idx] - 1) / (nobs[j] - 1)
                out[obs_idx, j] = np.interp(positions, i_grid, m)
            else:
                src_pos = np.arange(nobs[j]) / (nobs[j] - 1)
                out[O[obs_idx, j], j] = np.interp(src_pos, i_grid, m)
        else:
            if ties:
                positions = (r - 1) / (n_rows - 1)
                out[:, j] = np.interp(positions, i_grid, m)
            else:
                out[O[:, j], j] = m
    return out


def normalize_cyclic_loess(
    x: np.ndarray,
    weights: np.ndarray | None = None,
    span: float = 0.7,
    adaptive_span: bool = False,
    iterations: int = 3,
    method: str = "fast",
) -> np.ndarray:
    """
    Cyclic LOESS normalisation of columns of a matrix.

    Faithful port of R limma's ``normalizeCyclicLoess``
    (``limma/R/norm.R:535-580``). All three R methods are
    implemented:

    - ``"fast"`` (default): each column vs the row mean.
    - ``"pairs"``: pairwise loess over every column pair.
    - ``"affy"``: like ``"pairs"`` but accumulates all pair-wise
      adjustments then divides by ``n`` per iteration.

    Probe-vector weights (length = number of rows in ``x``) are
    supported for all three methods and are threaded through to
    :func:`loess_fit` in the weighted path. The unweighted ``"fast"``
    path keeps the legacy statsmodels lowess route (with ``it=3`` and
    ``delta=0.01*range(x)``) so existing pre-computed R-parity
    fixtures continue to pass bit-for-bit.

    Per-observation (matrix-shaped) weights are accepted by R's
    signature but fail inside R's own ``loessFit`` call with
    ``"y and weights have different lengths"`` - this is a latent
    bug in R limma. pylimma does not attempt to fix it; pass a
    probe-vector instead, or collapse your ``(n_genes, n_samples)``
    weight matrix with ``row_means = np.nanmean(w, axis=1)`` before
    calling.

    The ``adaptive_span`` default of ``False`` matches the installed R
    limma (3.66.0) used to generate the parity fixtures. The current
    upstream limma source (modified 25 Feb 2026) flips this default to
    ``True``; pylimma will follow that change once the bundled limma
    version is upgraded.
    """
    from .utils import loess_fit

    x = np.asarray(x, dtype=np.float64).copy()
    if x.ndim != 2:
        raise ValueError("normalize_cyclic_loess requires a 2D matrix")

    method = method.lower()
    if method not in ("fast", "pairs", "affy"):
        raise ValueError(
            f"normalize_cyclic_loess(method={method!r}) unknown; "
            "choose one of 'fast', 'pairs', 'affy'"
        )

    if adaptive_span:
        span = choose_lowess_span(
            x.shape[0], small_n=50, min_span=0.3, power=1/3,
        )

    n = x.shape[1]

    # Shared helper: fit loess of ``m`` vs ``a`` and return fitted
    # values in the original ordering. Uses statsmodels' lowess for
    # the unweighted path (legacy fixture parity) and routes through
    # loess_fit when weights are supplied.
    def _loess_adjust(m, a, w):
        obs = np.isfinite(m) & np.isfinite(a)
        if w is not None:
            obs &= np.isfinite(w) & (w > 0)
        if int(obs.sum()) < 2:
            return np.zeros_like(m)
        if w is None:
            xobs = a[obs]
            yobs = m[obs]
            delta = 0.01 * float(xobs.max() - xobs.min())
            f_obs = sm_lowess(
                yobs, xobs, frac=span, it=3, delta=delta,
                return_sorted=False,
            )
        else:
            out = loess_fit(m, a, weights=w, span=span)
            f_obs = out["fitted"][obs]
        f = np.zeros_like(m)
        f[obs] = f_obs
        return f

    if method == "fast":
        for _ in range(iterations):
            a = np.nanmean(x, axis=1)
            for i in range(n):
                m = x[:, i] - a
                f = _loess_adjust(m, a, weights)
                x[:, i] = x[:, i] - f
        return x

    if method == "pairs":
        # Pairwise loess (R norm.R:545-554).
        for _ in range(iterations):
            for i in range(n - 1):
                for j in range(i + 1, n):
                    m = x[:, j] - x[:, i]
                    a = 0.5 * (x[:, j] + x[:, i])
                    f = _loess_adjust(m, a, weights)
                    x[:, i] = x[:, i] + f / 2.0
                    x[:, j] = x[:, j] - f / 2.0
        return x

    # method == "affy": accumulate all pair adjustments, divide by n
    # per iteration (R norm.R:565-577).
    for _ in range(iterations):
        adjustment = np.zeros_like(x)
        for i in range(n - 1):
            for j in range(i + 1, n):
                m = x[:, j] - x[:, i]
                a = 0.5 * (x[:, j] + x[:, i])
                f = _loess_adjust(m, a, weights)
                adjustment[:, j] = adjustment[:, j] + f
                adjustment[:, i] = adjustment[:, i] - f
        x = x - adjustment / n
    return x


# -----------------------------------------------------------------------------
# normexp background correction
# -----------------------------------------------------------------------------


def _r_bw_nrd0(x: np.ndarray) -> float:
    """Port of ``stats::bw.nrd0`` - Silverman's rule-of-thumb bandwidth."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        raise ValueError("need at least 2 data points")
    hi = float(np.std(x, ddof=1))
    q25, q75 = np.percentile(x, [25, 75])
    lo = min(hi, float(q75 - q25) / 1.34)
    if lo == 0:
        lo = hi
    if lo == 0:
        lo = abs(float(x[0]))
    if lo == 0:
        lo = 1.0
    return 0.9 * lo * len(x) ** (-0.2)


def _r_density_epanechnikov(
    x: np.ndarray,
    n_pts: int = 512,
    cut: float = 3.0,
    ext: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``stats::density.default`` with ``kernel='epanechnikov'``.

    Mirrors R's FFT-based binned KDE: bandwidth via ``bw.nrd0``, linear
    binning via ``massdist``, FFT convolution with the canonical
    epanechnikov kernel, linear interpolation back onto the ``from``-to-
    ``to`` grid. ``n_pts`` is the user grid size; internally the FFT uses
    ``max(n_pts, 512)`` rounded up to a power of 2.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    N = len(x)

    bw = _r_bw_nrd0(x)

    n_user = int(n_pts)
    n = max(n_user, 512)
    if n > 512:
        n = 2 ** int(np.ceil(np.log2(n)))

    from_ = float(x.min()) - cut * bw
    to = float(x.max()) + cut * bw
    lo = from_ - ext * bw
    up = to + ext * bw

    # Linear binning (R's stats::massdist): length-2n output, first n active,
    # final n are zero-padding for FFT.
    y = np.zeros(2 * n)
    xdelta = (up - lo) / (n - 1)
    w = 1.0 / N
    xpos = (x - lo) / xdelta
    ix = np.floor(xpos).astype(np.int64)
    fx = xpos - ix
    interior = (ix >= 0) & (ix <= n - 2)
    np.add.at(y, ix[interior],     (1 - fx[interior]) * w)
    np.add.at(y, ix[interior] + 1,      fx[interior]  * w)
    left  = (ix == -1)
    right = (ix == n - 1)
    if left.any():
        np.add.at(y, np.zeros(int(left.sum()), dtype=np.int64),
                  fx[left] * w)
    if right.any():
        np.add.at(y, np.full(int(right.sum()), n - 1, dtype=np.int64),
                  (1 - fx[right]) * w)

    # kords: FFT-ordered kernel grid. Positions: 0, dx, 2dx, ..., n*dx,
    # -(n-1)*dx, ..., -dx  (length 2n).
    dx = (up - lo) / (n - 1)
    kords = np.empty(2 * n)
    kords[:n + 1] = np.arange(n + 1) * dx
    kords[n + 1:] = -np.arange(n - 1, 0, -1) * dx

    a = bw * np.sqrt(5.0)
    ax = np.abs(kords)
    kv = np.where(ax < a, 0.75 * (1.0 - (ax / a) ** 2) / a, 0.0)

    Y = np.fft.fft(y)
    K = np.fft.fft(kv)
    dens_ext = np.maximum(0.0, np.real(np.fft.ifft(Y * np.conj(K)))[:n] / (2 * n))

    xords = np.linspace(lo, up, n)
    x_out = np.linspace(from_, to, n_user)
    y_out = np.interp(x_out, xords, dens_ext)
    return x_out, y_out


def _r_density_mode(x: np.ndarray, n_pts: int) -> float:
    """Grid point with the largest density estimate, matching R's
    ``aux$x[order(-aux$y)[1]]`` idiom."""
    x_grid, y_grid = _r_density_epanechnikov(x, n_pts=n_pts)
    # R's order(-y)[1] is stable-sort ascending on -y -> index of max y.
    return float(x_grid[int(np.argsort(-y_grid, kind="stable")[0])])


def _bg_parameters(pm: np.ndarray, n_pts: int = 2 ** 14) -> dict:
    """Port of ``affy::bg.parameters`` (Bioconductor affy, LGPL >= 2.0).

    Estimates normexp (mu, sigma, alpha) by locating the mode of the
    foreground density, using the mode as the background centre, deriving
    sigma from below-mode dispersion and alpha from the mode of the
    above-mode residuals.
    """
    pm = np.asarray(pm, dtype=np.float64)
    pmbg = _r_density_mode(pm, n_pts)
    bg_data = pm[pm < pmbg]
    pmbg = _r_density_mode(bg_data, n_pts)
    bg_data = pm[pm < pmbg] - pmbg
    bgsd = np.sqrt(np.sum(bg_data ** 2) / (len(bg_data) - 1)) * np.sqrt(2)
    sig_data = pm[pm > pmbg] - pmbg
    expmean = _r_density_mode(sig_data, n_pts)
    alpha = 1.0 / expmean
    return {"alpha": alpha, "mu": pmbg, "sigma": bgsd}


def _bg_parameters_rma75(pm: np.ndarray, n_pts: int = 2 ** 14) -> dict:
    """Port of limma's ``.bg.parameters.rma75`` (``background-normexp.R``).

    Closed-form RMA-75 estimator of McGee & Chen (2006).
    """
    pm = np.asarray(pm, dtype=np.float64)

    def _mu_correct(m: float, s: float, a: float) -> float:
        sa = s * a

        def f(u):
            return (_scipy_norm.pdf(u - sa)
                    - sa * (_scipy_norm.cdf(u - sa)
                            + _scipy_norm.cdf(m / s + sa) - 1))

        from scipy.optimize import brentq
        t = brentq(f, -5, 10, xtol=1e-12)
        return m - s * t

    pmbg = _r_density_mode(pm, n_pts)
    bg_data = pm[pm < pmbg]
    pmbg = _r_density_mode(bg_data, n_pts)
    mubg = pmbg
    bg_data = pm[pm < pmbg] - pmbg
    bgsd = np.sqrt(np.sum(bg_data ** 2) / (len(bg_data) - 1)) * np.sqrt(2)

    q75 = 0.75
    alpha3 = -(float(np.quantile(pm, q75)) - pmbg) / np.log(1 - q75)

    mu3 = _mu_correct(mubg, bgsd, 1.0 / alpha3)
    mu3 = (mu3 + mubg) / 2

    bg_data3 = pm[pm < mu3] - mu3
    bgsd3 = np.sqrt(np.sum(bg_data3 ** 2) / (len(bg_data3) - 1)) * np.sqrt(2)
    alpha3 = -(float(np.quantile(pm, q75)) - mu3) / np.log(1 - q75)
    return {"alpha": 1.0 / alpha3, "mu": mu3, "sigma": bgsd3}


def _normexp_saddle_m2loglik(par: np.ndarray, x: np.ndarray) -> float:
    """Port of ``normexp_m2loglik_saddle`` in ``src/normexp.c`` (lines 20-131).

    Saddle-point approximation to normexp minus-twice log-likelihood.
    Parameterised as ``par = (mu, log(sigma), log(alpha))``.
    """
    x = np.asarray(x, dtype=np.float64)
    mu = float(par[0])
    sigma = float(np.exp(par[1]))
    sigma2 = sigma * sigma
    alpha = float(np.exp(par[2]))
    alpha2 = alpha * alpha
    alpha3 = alpha * alpha2
    alpha4 = alpha2 * alpha2
    c2 = sigma2 * alpha

    err = x - mu
    # Newton-iteration upper bound on theta (keeps 1 - alpha*theta > 0)
    ub1 = np.maximum(0.0, (err - alpha) / (alpha * np.abs(err)))
    ub2 = err / sigma2
    upperbound = np.minimum(ub1, ub2)
    c1 = -sigma2 - err * alpha
    c0 = -alpha + err
    disc = np.sqrt(c1 * c1 - 4.0 * c0 * c2)
    theta_quad = (-c1 - disc) / (2.0 * c2)
    theta = np.minimum(theta_quad, upperbound)

    has_converged = np.zeros_like(theta, dtype=bool)
    # Match C loop: `while(...) {j++; ...; if(j>50) break;}` permits exactly
    # 50 iterations of the Newton body (j reaches 51 only to test the break).
    for j in range(50):
        if has_converged.all():
            break
        active = ~has_converged
        omat = 1.0 - alpha * theta[active]
        dK = mu + sigma2 * theta[active] + alpha / omat
        ddK = sigma2 + alpha2 / (omat * omat)
        delta = (x[active] - dK) / ddK
        theta[active] += delta
        if j == 0:
            theta[active] = np.minimum(theta[active], upperbound[active])
        converged_now = np.abs(delta) < 1e-10
        has_converged[np.flatnonzero(active)[converged_now]] = True

    omat = 1.0 - alpha * theta
    omat2 = omat * omat
    k1 = mu * theta + 0.5 * sigma2 * theta * theta - np.log(omat)
    k2 = sigma2 + alpha2 / omat2
    logf = -0.5 * np.log(2.0 * np.pi * k2) - x * theta + k1
    k3 = 2.0 * alpha3 / (omat * omat2)
    k4 = 6.0 * alpha4 / (omat2 * omat2)
    logf = logf + k4 / (8.0 * k2 * k2) - 5.0 * k3 * k3 / (24.0 * k2 ** 3)
    return -2.0 * float(logf.sum())


def _normexp_m2loglik(par: np.ndarray, x: np.ndarray) -> float:
    """Port of ``normexp_m2loglik`` in ``src/normexp.c`` (lines 174-196).

    Exact normexp minus-twice log-likelihood. Parameterised as
    ``par = (mu, log(sigma^2), log(alpha))`` - note the ``log(sigma^2)``
    parametrisation here differs from the saddle function's ``log(sigma)``.
    """
    x = np.asarray(x, dtype=np.float64)
    mu = float(par[0])
    s2 = float(np.exp(par[1]))
    al = float(np.exp(par[2]))
    s = np.sqrt(s2)

    e = x - mu
    mu_sf = e - s2 / al
    # pnorm(0, mu_sf, s, lower.tail=FALSE, log.p=TRUE) = log(1 - Phi((0-mu_sf)/s))
    # = scipy.stats.norm.logsf(-mu_sf / s) = logsf((0 - mu_sf)/s)
    log_upper = _scipy_norm.logsf(-mu_sf / s)
    return -2.0 * float(np.sum(-np.log(al) - e / al + 0.5 * s2 / (al * al)
                               + log_upper))


def _normexp_gm2loglik(par: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Port of ``normexp_gm2loglik`` in ``src/normexp.c`` (lines 198-235).

    Gradient of ``_normexp_m2loglik`` wrt ``(mu, log(sigma^2), log(alpha))``.
    """
    x = np.asarray(x, dtype=np.float64)
    mu = float(par[0])
    s2 = float(np.exp(par[1]))
    al = float(np.exp(par[2]))
    s = np.sqrt(s2)
    al2 = al * al

    e = x - mu
    mu_sf = e - s2 / al
    # psionPsi = dnorm(0, mu_sf, s) / pnorm(0, mu_sf, s, lower.tail=FALSE)
    #          = exp(logpdf(mu_sf/s) - log(s) - logsf(-mu_sf/s))
    # dnorm(0, mu_sf, s) = (1/s) * phi(-mu_sf/s) = (1/s) * phi(mu_sf/s)
    log_dnorm = _scipy_norm.logpdf(-mu_sf / s) - np.log(s)
    log_upper = _scipy_norm.logsf(-mu_sf / s)
    psionPsi = np.exp(log_dnorm - log_upper)

    d_mu   = np.sum(1.0 / al - psionPsi)
    d_s2   = np.sum(0.5 / al2 - (1.0 / al + 0.5 / s2 * mu_sf) * psionPsi)
    d_al   = np.sum(e / al2 - 1.0 / al - s2 / (al2 * al) + psionPsi * s2 / al2)

    g = np.array([d_mu, d_s2, d_al], dtype=np.float64)
    g *= -2.0
    g[1] *= s2  # chain rule for log(s2)
    g[2] *= al  # chain rule for log(al)
    return g


def _normexp_hm2loglik(par: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Port of ``normexp_hm2loglik`` in ``src/normexp.c`` (lines 237-326).

    Hessian of ``_normexp_m2loglik`` wrt ``(mu, log(sigma^2), log(alpha))``.
    """
    x = np.asarray(x, dtype=np.float64)
    mu = float(par[0])
    s2 = float(np.exp(par[1]))
    al = float(np.exp(par[2]))
    s = np.sqrt(s2)
    al2 = al * al
    al3 = al2 * al
    al4 = al3 * al
    s2onal = s2 / al
    s2onal2 = s2onal * s2onal
    s2onal3 = s2onal2 * s2onal

    e = x - mu
    mu_sf = e - s2onal
    eps2 = e + s2onal
    log_dnorm = _scipy_norm.logpdf(-mu_sf / s) - np.log(s)
    log_upper = _scipy_norm.logsf(-mu_sf / s)
    psionPsi = np.exp(log_dnorm - log_upper)
    psionPsi2 = psionPsi * psionPsi

    dL_dal = np.sum(0.5 / al2 - (1.0 / al + 0.5 / s2 * mu_sf) * psionPsi)
    dL_ds2 = np.sum(e / al2 - 1.0 / al - s2 / al3 + psionPsi * s2 / al2)

    d2L_dbtdbt = np.sum(-psionPsi2 - psionPsi * mu_sf / s2)
    d2L_dbtds2 = np.sum(-0.5 * eps2 * psionPsi2 / s2
                        + (-eps2 ** 2 + 2.0 * s2onal * eps2 + s2)
                          * psionPsi * (0.5 / (s2 * s2)))
    d2L_dbtdal = np.sum(-1.0 / al2 + (s2 / al2) * psionPsi2
                        + mu_sf * psionPsi / al2)
    d2L_ds2ds2 = np.sum(
        -(0.25 / (s2 * s2)) * (eps2 ** 2) * psionPsi2
        + psionPsi * (-e ** 3 + e * (3.0 * al - e) * s2onal
                      + (e + al) * s2onal2 + s2onal3)
          / (4.0 * s2 ** 3)
    )
    d2L_dalds2 = np.sum(
        -1.0 / al3 + (0.5 / al2) * (psionPsi2 * eps2
                                    + (e ** 2 + s2 - s2onal2) * psionPsi / s2)
    )
    d2L_daldal = np.sum(
        1.0 / al2 - 2.0 * e / al3 + 3.0 * s2 / al4
        - psionPsi2 * (s2 * s2 / al4)
        - psionPsi * (mu_sf + 2.0 * al) * (s2 / al4)
    )

    # Assemble: see C lines 309-317. Chain-rule factors for log(s2), log(al).
    H = np.empty((3, 3))
    H[0, 0] = -2.0 * d2L_dbtdbt
    H[0, 1] = H[1, 0] = -2.0 * s2 * d2L_dbtds2
    H[0, 2] = H[2, 0] = -2.0 * al * d2L_dbtdal
    H[1, 1] = -2.0 * (s2 * s2 * d2L_ds2ds2 + s2 * dL_ds2)
    H[1, 2] = H[2, 1] = -2.0 * al * s2 * d2L_dalds2
    H[2, 2] = -2.0 * (al2 * d2L_daldal + al * dL_dal)
    return H


def normexp_fit(
    x: np.ndarray,
    method: str = "saddle",
    n_pts: int | None = None,
    trace: bool = False,
) -> dict:
    """
    Estimate parameters of the normal + exponential convolution model.

    Faithful port of R limma's ``normexp.fit``
    (``limma/R/background-normexp.R``). The compiled C
    routines (``fit_saddle_nelder_mead``, ``normexp_m2loglik``,
    ``normexp_gm2loglik``, ``normexp_hm2loglik``) are ported to pure
    NumPy/SciPy in the ``_normexp_*`` helpers.

    Parameters
    ----------
    x : ndarray
        One array's worth of foreground intensities.
    method : {"saddle", "mle", "rma", "rma75", "mcgee", "nlminb", "nlminblog"}
        ``"saddle"`` (default): Nelder-Mead on the saddle-point
        approximation to the log-likelihood. ``"mle"``: refine with a
        trust-region Newton search using the exact log-likelihood.
        ``"rma"``: closed-form estimator via ``affy::bg.parameters``
        (ported from the affy Bioconductor package). ``"rma75"``:
        closed-form variant of ``rma`` from McGee & Chen (2006).
        ``"mcgee"`` aliases ``"rma75"``; ``"nlminb"`` and ``"nlminblog"``
        alias ``"mle"`` - matches R's backward-compatibility mapping.
    n_pts : int, optional
        Downsample ``x`` to ``n_pts`` quantile points before fitting.
    trace : bool
        Ignored (matches R's stub behaviour).

    Returns
    -------
    dict with keys:
        ``par`` : ndarray of length 3 = ``(mu, log(sigma), log(alpha))``
        ``m2loglik`` : float, only present for the saddle / mle paths
        ``convergence`` : int, only present for the saddle / mle paths
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) < 4:
        raise ValueError(
            "Not enough data: need at least 4 non-missing corrected intensities"
        )
    if trace:
        print("trace not currently implemented")

    valid = ("mle", "saddle", "rma", "rma75", "mcgee", "nlminb", "nlminblog")
    if method not in valid:
        raise ValueError(f"'method' should be one of {valid}")
    # Backward-compatibility aliases (background-normexp.R:38-40)
    if method == "mcgee":
        method = "rma75"
    if method in ("nlminb", "nlminblog"):
        method = "mle"

    if method == "rma":
        out = _bg_parameters(x)
        return {"par": np.array(
            [out["mu"], np.log(out["sigma"]), -np.log(out["alpha"])]
        )}

    if method == "rma75":
        out = _bg_parameters_rma75(x)
        return {"par": np.array(
            [out["mu"], np.log(out["sigma"]), -np.log(out["alpha"])]
        )}

    # Starting values (background-normexp.R:53-68)
    q = np.quantile(x, [0.0, 0.05, 0.1, 1.0])
    if q[0] == q[3]:
        return {"par": np.array([q[0], -np.inf, -np.inf]),
                "m2loglik": np.nan, "convergence": 0}
    if q[1] > q[0]:
        mu = q[1]
    elif q[2] > q[0]:
        mu = q[2]
    else:
        mu = q[0] + 0.05 * (q[3] - q[0])
    below = x[x < mu]
    sigma2 = float(np.mean((below - mu) ** 2)) if len(below) else 0.0
    alpha0 = float(np.mean(x)) - mu
    if alpha0 <= 0:
        alpha0 = 1e-6
    par0 = np.array([mu, 0.5 * np.log(sigma2), np.log(alpha0)])

    if n_pts is not None and 4 <= n_pts < len(x):
        # R: x <- quantile(x, ((1:n.pts) - 0.5)/n.pts, type=5)
        # Type-5 quantile: linear interpolation at h = p*N + 0.5.
        probs = (np.arange(1, n_pts + 1) - 0.5) / n_pts
        n_x = len(x)
        sorted_x = np.sort(x)
        h = probs * n_x + 0.5
        h_floor = np.clip(np.floor(h).astype(int) - 1, 0, n_x - 1)
        h_ceil = np.clip(h_floor + 1, 0, n_x - 1)
        frac = np.clip(h - np.floor(h), 0.0, 1.0)
        x = sorted_x[h_floor] * (1 - frac) + sorted_x[h_ceil] * frac

    # Nelder-Mead on the saddle-point log-likelihood. R's nmmin defaults:
    # abstol=-Inf, intol=sqrt(machine.eps), alpha=1, beta=0.5, gamma=2,
    # maxit=500.
    nm_result = minimize(
        _normexp_saddle_m2loglik,
        par0,
        args=(x,),
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1.490116e-08,
                 "fatol": 1.490116e-08, "adaptive": False},
    )
    saddle_out = {
        "par": np.asarray(nm_result.x, dtype=np.float64).copy(),
        "convergence": 0 if nm_result.success else 1,
        "fncount": int(nm_result.nfev),
        "m2loglik": float(nm_result.fun),
    }

    if method == "saddle":
        return saddle_out

    # MLE refinement: convert log(sigma) -> log(sigma^2) then Newton-CG.
    par1 = saddle_out["par"].copy()
    par1[1] = 2.0 * par1[1]
    LL1 = _normexp_m2loglik(par1, x)

    # R's nlminb call uses scale = median(abs(Par1))/abs(Par1) to equalise
    # per-parameter magnitudes. scipy's Newton-CG has no direct `scale`
    # argument; implement it by working in rescaled coordinates
    # y = par / sc (so sc is a *divisor* and the search space is normalised)
    # and passing the scaled objective/gradient/Hessian to minimize.
    abs_par1 = np.abs(par1)
    if np.all(abs_par1 > 0):
        sc = np.median(abs_par1) / abs_par1
    else:
        sc = np.ones_like(par1)
    y0 = par1 * sc

    def _obj_scaled(y, x_):
        return _normexp_m2loglik(y / sc, x_)

    def _jac_scaled(y, x_):
        return _normexp_gm2loglik(y / sc, x_) / sc

    def _hess_scaled(y, x_):
        # Hessian transforms as diag(1/sc) H diag(1/sc).
        H = _normexp_hm2loglik(y / sc, x_)
        inv_sc = 1.0 / sc
        return inv_sc[:, None] * H * inv_sc[None, :]

    mle_result = minimize(
        _obj_scaled,
        y0,
        args=(x,),
        method="Newton-CG",
        jac=_jac_scaled,
        hess=_hess_scaled,
    )
    par_raw = np.asarray(mle_result.x, dtype=np.float64) / sc
    par_out = par_raw.copy()
    par_out[1] = 0.5 * par_out[1]
    mle_out = {
        "par": par_out,
        "convergence": 0 if mle_result.success else 1,
        "m2loglik": float(mle_result.fun),
        "iterations": int(getattr(mle_result, "nit", 0)),
        "evaluations": {
            "function": int(getattr(mle_result, "nfev", 0)),
            "gradient": int(getattr(mle_result, "njev", 0)),
        },
        "message": str(getattr(mle_result, "message", "")),
    }

    if mle_out["m2loglik"] >= LL1:
        return saddle_out
    return mle_out


def normexp_signal(par: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Expected value of signal given foreground under the normal + exponential
    convolution model.

    Faithful port of R limma's ``normexp.signal``
    (``limma/R/background-normexp.R:3-23``).
    """
    par = np.asarray(par, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    mu = float(par[0])
    sigma = float(np.exp(par[1]))
    sigma2 = sigma * sigma
    alpha = float(np.exp(par[2]))
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    mu_sf = x - mu - sigma2 / alpha
    # dnorm(0, mu_sf, sigma, log=TRUE) - pnorm(0, mu_sf, sigma, lower.tail=FALSE, log.p=TRUE)
    # = logpdf((0-mu_sf)/sigma) - log(sigma) - logsf((0-mu_sf)/sigma)
    z = -mu_sf / sigma
    signal = mu_sf + sigma2 * np.exp(
        _scipy_norm.logpdf(z) - np.log(sigma) - _scipy_norm.logsf(z)
    )
    obs = ~np.isnan(signal)
    if np.any(signal[obs] < 0):
        warnings.warn(
            "Limit of numerical accuracy reached with very low intensity or "
            "very high background:\nsetting adjusted intensities to small value"
        )
        signal[obs] = np.maximum(signal[obs], 1e-6)
    return signal


# -----------------------------------------------------------------------------
# background correction (single-channel / matrix path only)
# -----------------------------------------------------------------------------


_BACKGROUND_METHODS = (
    "none", "subtract", "half", "minimum", "movingmin", "edwards", "normexp",
)


def background_correct(
    object,
    background: np.ndarray | None = None,
    method: str = "auto",
    offset: float = 0.0,
    printer=None,
    normexp_method: str = "saddle",
    verbose: bool = True,
    *,
    out_layer: str = "bg_corrected",
    uns_key: str = "background_correct",
    layer: str | None = None,
):
    """
    Background-correct a single-channel intensity matrix or EList.

    Faithful port of R limma's ``backgroundCorrect.matrix``
    (``limma/R/background.R:40-108``). Accepts a matrix,
    dict / EList / AnnData carrying the foreground, with the background
    passed separately as ``background`` (matches ``backgroundCorrect.matrix``
    ``Eb`` parameter).

    Two-colour / RGList / MAList dispatch is out of scope (see
    ``memory/policy_data_class_wrappers.md``). ``movingmin`` and
    ``edwards`` both require a printer / spotted-array layout and raise
    ``NotImplementedError``; the ``printer`` parameter is accepted for R
    signature compatibility but is only used by those out-of-scope paths.
    """
    if printer is not None and method not in ("movingmin", "edwards"):
        # R silently ignores printer for every non-movingmin / non-edwards
        # method; matching that behaviour would be surprising here since
        # Python users typically pass printer only by mistake. Warn but
        # proceed.
        warnings.warn(
            "'printer' is only used by 'movingmin' / 'edwards' background "
            "methods, both of which are out of scope for the single-channel "
            "port; argument ignored.",
            UserWarning,
        )
    original_input = object
    eawp = get_eawp(object, layer=layer)
    E = np.asarray(eawp["exprs"], dtype=np.float64).copy()
    Eb = None if background is None else np.asarray(background, dtype=np.float64)

    if method == "auto":
        method = "normexp" if Eb is None else "subtract"
    if method not in _BACKGROUND_METHODS:
        raise ValueError(
            f"'method' should be one of {_BACKGROUND_METHODS}"
        )

    # R's matrix path: if Eb is NULL and method needs background, downgrade
    # to 'none' (background.R:51-52).
    if Eb is None and method in ("subtract", "half", "minimum",
                                 "movingmin", "edwards"):
        method = "none"

    if method == "none":
        pass
    elif method == "subtract":
        E = E - Eb
    elif method == "half":
        E = np.maximum(E - Eb, 0.5)
    elif method == "minimum":
        E = E - Eb
        for j in range(E.shape[1]):
            col = E[:, j]
            below = col < 1e-18
            if np.any(below & ~np.isnan(col)):
                m = np.nanmin(col[~below])
                col[below] = m / 2
                E[:, j] = col
    elif method == "movingmin":
        raise NotImplementedError(
            "method='movingmin' requires a printer / spotted-array layout "
            "and is out of scope for the single-channel port"
        )
    elif method == "edwards":
        raise NotImplementedError(
            "method='edwards' requires a printer / spotted-array layout "
            "and is out of scope for the single-channel port"
        )
    elif method == "normexp":
        if Eb is not None:
            E = E - Eb
        for j in range(E.shape[1]):
            if verbose:
                print(f"Array {j + 1}", end="")
            col = E[:, j]
            out = normexp_fit(col, method=normexp_method)
            E[:, j] = normexp_signal(out["par"], col)
            if verbose:
                print(" corrected")

    if offset:
        E = E + offset

    return put_eawp(
        {"E": E},
        original_input,
        out_layer=out_layer,
        weights_layer=None,
        uns_key=uns_key,
        single_matrix=True,
    )


# -----------------------------------------------------------------------------
# avearrays: average over technical-replicate columns
# -----------------------------------------------------------------------------


def aver_arrays(
    x,
    id=None,
    weights: np.ndarray | None = None,
):
    """
    Average over technical-replicate columns.

    Faithful port of R limma's ``avearrays.default`` (matrix) and
    ``avearrays.EList`` (``limma/R/avearrays.R:6-64``). The
    unweighted path mirrors ``rowsum(..., reorder=FALSE)``: groups are
    ordered by first appearance (``unique(id)``), not sorted. The
    weighted path fits ``lm_fit(x, design=one_hot(id), weights=weights)``
    and returns the coefficients.

    ``avearrays.MAList`` is out of scope (two-colour data only).

    For ``AnnData`` input, ``id`` defaults to ``adata.obs_names`` and a
    new ``AnnData`` is returned (obs axis collapsed to the unique ids).
    Because the output has a different number of samples than the input,
    in-place mutation via a layer is not possible, so this is the one
    AnnData-in path in pylimma that returns a value rather than ``None``.
    """
    if x is None:
        return None  # R: if(is.null(x)) return(NULL)

    if isinstance(x, EList):
        return _aver_arrays_elist(x, id=id, weights=weights)

    if _is_anndata(x):
        return _aver_arrays_anndata(x, id=id, weights=weights)

    arr = np.asarray(x)
    if arr.ndim < 2:
        raise ValueError("x must be a matrix")

    if id is None:
        # R defaults to colnames(x). Pull from whatever the source exposes:
        # pandas columns or structured-array field names.
        if isinstance(x, pd.DataFrame):
            id = list(x.columns)
        elif arr.dtype.names is not None:  # numpy structured array
            id = list(arr.dtype.names)
        else:
            raise ValueError("No sample IDs")
    id_arr = np.asarray([str(v) for v in id])

    # String matrix short-circuit (avearrays.R:16-21): drop duplicate columns
    # without averaging.
    if arr.dtype.kind in ("U", "S", "O") and not np.issubdtype(arr.dtype, np.number):
        _, first_idx = np.unique(id_arr, return_index=True)
        keep = np.sort(first_idx)
        return arr[:, keep]

    arr = np.asarray(arr, dtype=np.float64)

    # Unique IDs preserving first-appearance order (R: factor(..., levels=unique(ID)))
    _, first_idx = np.unique(id_arr, return_index=True)
    unique_order = id_arr[np.sort(first_idx)]

    if weights is None:
        # rowsum(t(x), ID, reorder=FALSE, na.rm=TRUE) / rowsum(t(1-is.na(x)), ...)
        df_vals = pd.DataFrame(arr.T, index=id_arr)
        # groupby with sort=False keeps first-appearance order
        sums = df_vals.groupby(level=0, sort=False).sum(min_count=0).reindex(unique_order)
        counts = (~pd.DataFrame(np.isnan(arr.T), index=id_arr)).groupby(
            level=0, sort=False).sum().reindex(unique_order)
        y = sums.values / counts.values
        return y.T

    # Weighted path: lm_fit on one-hot design.
    from .lmfit import lm_fit
    one_hot = np.zeros((len(id_arr), len(unique_order)), dtype=np.float64)
    for j, level in enumerate(unique_order):
        one_hot[id_arr == level, j] = 1.0
    fit = lm_fit(arr, design=one_hot, weights=np.asarray(weights, dtype=np.float64))
    coef = np.asarray(fit["coefficients"], dtype=np.float64)
    return coef


def _aver_arrays_elist(x: EList, id=None, weights=None) -> EList:
    """EList dispatch for aver_arrays (``avearrays.EList``)."""
    E = np.asarray(x["E"], dtype=np.float64)
    n_cols = E.shape[1]
    w_default = x.get("weights", None) if weights is None else weights
    if id is None:
        targets = x.get("targets", None)
        if targets is None:
            raise ValueError("No sample IDs")
        id = list(targets.index) if hasattr(targets, "index") else list(targets)
    id_arr = np.asarray([str(v) for v in id])

    dup = np.zeros(n_cols, dtype=bool)
    seen = set()
    for i, v in enumerate(id_arr):
        if v in seen:
            dup[i] = True
        else:
            seen.add(v)
    if not dup.any():
        return x  # R: return(x) when no duplicates

    out = EList(x)
    out["E"] = aver_arrays(E, id=id_arr, weights=w_default)
    if x.get("weights", None) is not None:
        out["weights"] = aver_arrays(np.asarray(x["weights"], dtype=np.float64),
                                     id=id_arr, weights=w_default)
    other = x.get("other", None)
    if other is not None:
        out["other"] = {
            k: aver_arrays(np.asarray(v, dtype=np.float64), id=id_arr,
                           weights=w_default)
            for k, v in other.items()
        }
    # R: y <- x[,!d] first, then overwrites slots. So drop duplicate targets.
    keep = ~dup
    targets = x.get("targets", None)
    if targets is not None and hasattr(targets, "iloc"):
        out["targets"] = targets.iloc[keep].reset_index(drop=True)
    return out


def _aver_arrays_anndata(adata, id=None, weights=None):
    """AnnData dispatch for aver_arrays.

    Averages over duplicate samples (obs rows) identified by ``id``
    (defaulting to ``adata.obs_names``) and returns a new AnnData with
    the obs axis collapsed to the unique ids in order of first
    appearance. Gene (var) axis is preserved.
    """
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError(
            "anndata is required for aver_arrays(AnnData) but is not installed"
        ) from exc

    # get_eawp densifies and transposes to limma's (n_genes, n_samples).
    eawp = get_eawp(adata)
    E = np.asarray(eawp["exprs"], dtype=np.float64)

    if id is None:
        id = np.asarray(adata.obs_names)
    id_arr = np.asarray([str(v) for v in id])
    if id_arr.shape[0] != E.shape[1]:
        raise ValueError(
            f"length of id ({id_arr.shape[0]}) must match number of "
            f"samples ({E.shape[1]})"
        )

    # Delegate the averaging to the matrix path; output is (n_genes, n_unique).
    averaged = aver_arrays(E, id=id_arr, weights=weights)

    # First-occurrence ordering of the unique ids, to pick representative
    # obs rows for the collapsed AnnData.
    _, first_idx = np.unique(id_arr, return_index=True)
    keep_order = np.sort(first_idx)
    new_obs_names = id_arr[keep_order]

    if adata.obs is not None and len(adata.obs.columns):
        obs_new = adata.obs.iloc[keep_order].copy()
    else:
        obs_new = pd.DataFrame(index=pd.Index(new_obs_names))
    obs_new.index = pd.Index(new_obs_names)

    var_new = adata.var.copy() if adata.var is not None else None

    # averaged is (n_genes, n_unique_ids); AnnData wants (n_samples, n_genes).
    return ad.AnnData(X=np.asarray(averaged).T, obs=obs_new, var=var_new)


# ============================================================================
# normalize_vsn (port of vsn::vsnMatrix; called by R limma's normalizeVSN)
# ============================================================================
#
# Algorithm reference: vsn/R/vsn2.R + vsn/src/vsn2.c (Wolfgang Huber 2002-2009).
# We port the matrix path that R limma's normalizeVSN.default invokes:
# single stratum, no reference, default options, calib="affine". Strata,
# reference, sample, "calib=none" branches and the AffyBatch / RGList /
# EListRaw S4 dispatchers are out of scope (RGList / EListRaw are not
# ported per policy_data_class_wrappers).
#
# Model. Per-column parameters (a_j, b_j) for j = 1..ncol. The variance-
# stabilising transform is h(y_ij) = arsinh(exp(b_j) * y_ij + a_j). After
# fitting, the row means mu_i are profiled out and the (a, b) are chosen
# to maximise the log-likelihood under the assumption that the h(y_ij)
# are jointly normal with row-mean mu_i and a single column-shared
# variance sigsq.
#
# Optimiser. R/vsn calls L-BFGS-B via the LINPACK-era lbfgsb wrapper. We
# call scipy.optimize.minimize(method="L-BFGS-B"). Same algorithm,
# different implementation; converged parameters typically agree to
# rtol ~ 1e-4. The resulting transformation matches R to a tighter
# tolerance because small parameter wiggles compress through the asinh.
#
# LTS robustification. After each ML fit, residuals are computed for ALL
# rows; rows are split into 5 hmean-rank quintiles and within each (slice,
# stratum) cell, only rows with squared-residual below the 90th percentile
# are kept for the next iteration (slice 1, the lowest-intensity rows,
# is exempt and always kept whole). Up to 7 iterations.


def _vsn_neg_loglik_and_grad(par, y_full, valid_full):
    """
    Negative profile log-likelihood and gradient for vsn affine
    calibration, single stratum.

    par layout: [a_1, ..., a_C, b_1, ..., b_C] for C columns.
    The transform is h(y) = arsinh(exp(b_j)*y + a_j).
    """
    n_rows, n_cols = y_full.shape
    a = par[:n_cols]
    b = par[n_cols:]
    fb = np.exp(b)

    # z_ij = exp(b_j)*y_ij + a_j
    z = fb[np.newaxis, :] * y_full + a[np.newaxis, :]
    asly = np.where(valid_full, np.arcsinh(z), 0.0)
    one_plus_z2 = 1.0 + z * z
    ma = np.where(valid_full, 1.0 / np.sqrt(one_plus_z2), 0.0)

    # Profile mu: per-row mean of asly over valid columns.
    n_per_row = valid_full.sum(axis=1)
    row_sum = (asly * valid_full).sum(axis=1)
    with np.errstate(invalid="ignore"):
        mu = np.where(n_per_row > 0, row_sum / n_per_row, np.nan)

    # Residuals (zero on invalid cells; mu of all-NaN rows is NaN).
    valid_resid = valid_full & ~np.isnan(mu)[:, np.newaxis]
    resid = np.where(valid_resid, asly - np.where(np.isnan(mu), 0.0, mu)[:, np.newaxis], 0.0)
    ssq = (resid * resid).sum()

    # n_total counts only cells whose row has at least one valid entry,
    # matching the C code: rows with all-NaN are skipped from ssq.
    n_total = int(valid_resid.sum())
    sigsq = ssq / n_total if n_total > 0 else np.inf

    # Jacobian terms (only over rows with mu defined, mirroring the
    # 1st-sweep loop in vsn2.c which counts every non-NA y_ij).
    jac1 = (np.log(one_plus_z2) * valid_full).sum()
    n_per_col = valid_full.sum(axis=0).astype(float)
    jac2 = (n_per_col * b).sum()
    jacobian = 0.5 * jac1 - jac2

    nll = n_total / 2.0 * np.log(2.0 * np.pi * sigsq) + n_total / 2.0 + jacobian

    # Gradient. From vsn2.c grad_loglik:
    #   z_grad = resid / sigsq + ma * z
    #   grad_a_j = sum_i z_grad[i,j] * ma[i,j]
    #   grad_b_j = exp(b_j) * sum_i z_grad[i,j] * ma[i,j] * y[i,j] - n_j
    # (the n_j term comes from sb-(double)nj/FUN(b[j]), then DFDB(b)*FUN(b)=exp(2b),
    #  pulled through to give exp(b)*sum - n_j.)
    rfac = 1.0 / sigsq if sigsq > 0 else 0.0
    z_grad = resid * rfac + ma * z
    grad_a = (z_grad * ma * valid_resid).sum(axis=0)
    grad_b = fb * (z_grad * ma * y_full * valid_resid).sum(axis=0) - n_per_col
    grad = np.concatenate([grad_a, grad_b])
    return nll, grad


def _vsn_ml_fit(y, pstart, factr=5e7, pgtol=2e-4, maxit=60000):
    """Profile-likelihood ML fit via L-BFGS-B (single stratum, affine)."""
    from scipy.optimize import minimize

    n_rows, n_cols = y.shape
    valid = ~np.isnan(y)
    y_filled = np.where(valid, y, 0.0)

    bounds = [(None, None)] * n_cols + [(-100.0, 100.0)] * n_cols
    eps = np.finfo(float).eps
    result = minimize(
        _vsn_neg_loglik_and_grad,
        pstart,
        args=(y_filled, valid),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": factr * eps, "gtol": pgtol, "maxiter": maxit},
    )
    return result.x, int(not result.success)


def _vsn_trsf(y, par):
    """Apply h(y) = arsinh(exp(b)*y + a), single stratum, affine."""
    n_rows, n_cols = y.shape
    a = par[:n_cols]
    b = par[n_cols:]
    fb = np.exp(b)
    return np.where(np.isnan(y), np.nan,
                    np.arcsinh(fb[np.newaxis, :] * y + a[np.newaxis, :]))


def _vsn_lts_select(hy, lts_quantile):
    """
    Robust trimming after an ML fit: split rows into 5 quintile slices by
    hmean rank, drop rows whose squared-residual is above the per-slice
    quantile (slice 1 is exempt). Single-stratum version.
    """
    hmean = np.nanmean(hy, axis=1)
    rvar = np.nansum((hy - hmean[:, np.newaxis]) ** 2, axis=1)
    n_rows = hy.shape[0]

    # rank(hmean, na.last=TRUE): NaNs go to the highest ranks.
    finite = ~np.isnan(hmean)
    order = np.empty(n_rows, dtype=np.int64)
    finite_idx = np.where(finite)[0]
    nan_idx = np.where(~finite)[0]
    finite_sort = finite_idx[np.argsort(hmean[finite_idx], kind="stable")]
    sorted_all = np.concatenate([finite_sort, nan_idx])
    rank = np.empty(n_rows, dtype=np.float64)
    rank[sorted_all] = np.arange(1, n_rows + 1, dtype=np.float64)

    # cut(rank, breaks=5) -> 5 equal-width bins on the rank scale [1..n].
    n_slices = 5
    edges = np.linspace(1.0, float(n_rows), n_slices + 1)
    # cut(..., right=TRUE, include.lowest=FALSE by default for cut):
    # bin k contains rank in (edges[k-1], edges[k]], with the lowest
    # value getting the lowest bin (cut's default attaches the lower
    # boundary to bin 1).
    slice_ = np.digitize(rank, edges[1:-1], right=False)
    slice_ = np.clip(slice_, 0, n_slices - 1)

    # Per-slice quantile of rvar.
    keep = np.zeros(n_rows, dtype=bool)
    for s in range(n_slices):
        in_slice = slice_ == s
        if not np.any(in_slice):
            continue
        if s == 0:
            keep |= in_slice  # slice 1 always kept (exempt)
            continue
        slice_rvar = rvar[in_slice]
        finite_rvar = slice_rvar[~np.isnan(slice_rvar)]
        if finite_rvar.size == 0:
            keep |= in_slice
            continue
        threshold = np.quantile(finite_rvar, lts_quantile)
        keep |= in_slice & (rvar <= threshold)
    return np.where(keep)[0]


def _scaling_factor_transform(b):
    """vsn's f(b) = exp(b) (the 'affine' scale-factor parameterisation)."""
    return np.exp(b)


def normalize_vsn(
    x: np.ndarray,
    lts_quantile: float = 0.9,
    cvg_niter: int = 7,
    factr: float = 5e7,
    pgtol: float = 2e-4,
    maxit: int = 60000,
) -> np.ndarray:
    """
    Variance-stabilising normalisation (vsn).

    Port of the matrix-path used by R limma's ``normalizeVSN.default``,
    which delegates to ``vsn::vsnMatrix``. Fits per-column offset and
    scale parameters under the model
    ``h(y_ij) = arsinh(exp(b_j)*y_ij + a_j)``, then returns the
    transformed matrix on a base-2-log-like scale (offset to align
    array means).

    This is the **single-stratum, no-reference, ``calib="affine"``**
    code path - the one limma calls. Stratified fits, reference-based
    fits, sub-sampling and ``calib="none"`` are not ported.

    Parameters
    ----------
    x : ndarray
        Expression matrix, shape ``(n_features, n_arrays)``. Raw
        intensities (not log-transformed). NaN allowed.
    lts_quantile : float, default 0.9
        Quantile used by the LTS robust iteration. ``1.0`` disables
        the robust iteration (single ML fit).
    cvg_niter : int, default 7
        Maximum number of LTS-ML iterations.
    factr, pgtol, maxit :
        L-BFGS-B options (passed through to
        ``scipy.optimize.minimize(method="L-BFGS-B")``). Defaults
        match R/vsn's ``defaultpar``.

    Returns
    -------
    ndarray
        Transformed matrix, same shape as ``x``. Values are on a
        base-2-log-like scale; for typical microarray-style inputs,
        differences between columns are calibrated out.

    Notes
    -----
    **Loglik is flat in a uniform-shift-in-b direction.** For typical
    microarray-style intensities ``exp(b_j) * y_ij`` dominates ``a_j``
    and ``arsinh(z) ~ log(2 z) = log(2) + b_j + log(y_ij)``, so a
    uniform shift ``b -> b + delta`` adds the same constant to every
    transformed cell. The profile-likelihood row-mean centring and
    the per-stratum ``hoffset`` rescaling both absorb this shift, so
    the log-likelihood is asymptotically flat under it. Different
    L-BFGS-B implementations land at different points along this
    flat valley despite reporting near-identical nll.

    **End-to-end parity is rtol ~ 2e-4 against R/vsn 3.66.0**, not the
    rtol ~ 1e-6 typical of pylimma's other ports, for that reason.
    The output IS still invariant under the b shift in the asymptotic
    limit; the residual disagreement comes from the finite-``y``
    correction. The transform itself (``_vsn_trsf`` plus the
    ``hoffset`` formula) is bit-faithful to ``vsn::vsn2trsf`` and is
    verified in
    ``TestNormalizeVSNRParity::test_transform_at_r_params_matches_r_to_machine_precision``
    (rtol=1e-6).

    **Deliberate divergence from R's pstart heuristic.** R/vsn's
    ``pstartHeuristic`` initialises ``b = 1``; with R's LINPACK
    ``lbfgsb`` that walks down to ``b ~ 0``. With scipy's L-BFGS-B
    starting from ``b = 1`` walks up to ``b ~ 1.1``, landing in a
    different region of the flat valley and giving rtol ~ 4e-3.
    pylimma starts from ``b = 0`` (unit scale factor, the natural
    "no scaling" initial guess) which keeps scipy's converged point
    in the same valley region as R's, reducing end-to-end rel-diff
    to rtol ~ 2e-4. ``pstart`` is documented as a heuristic in
    ``R/vsn2.R`` and is not surfaced through limma's
    ``normalizeVSN.default`` interface, so this change is invisible
    to users translating limma scripts. See
    ``notes_during_implementation.md`` 2026-05-01 entry for the full
    discussion.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be a 2-D matrix")
    n_rows, n_cols = x.shape
    if n_cols < 2:
        raise ValueError(
            "x must have 2 or more columns when no reference is supplied"
        )

    # Initial parameters: a=0, b=0 per column (unit scale factor, the
    # natural "no scaling" starting point). R/vsn's pstartHeuristic
    # uses b=1; with R's LINPACK lbfgsb that walks DOWN toward b=0,
    # whereas scipy's L-BFGS-B with the same loose tolerances walks
    # UP and lands in a different region of the loglik's flat valley
    # (the loglik is asymptotically flat under a uniform shift in b
    # for large y, so multiple optima exist). Starting at b=0 keeps
    # scipy's converged params close to R's; output rel-diff drops
    # from ~4e-3 (b=1 start) to ~2e-4 (b=0 start).
    pstart = np.zeros(2 * n_cols)

    # vsnLTS loop: ML on the active rows, then trim to the best 90% by
    # squared row residual within hmean-rank quintile.
    whsel = np.arange(n_rows)
    cur = pstart
    rsv_par = pstart
    iter_par = pstart
    for it in range(cvg_niter):
        sub = x if it == 0 else x[whsel]
        rsv_par, _fail = _vsn_ml_fit(sub, iter_par, factr=factr,
                                     pgtol=pgtol, maxit=maxit)
        iter_par = rsv_par  # warm-start the next iteration (matches R)
        if abs(lts_quantile - 1.0) < np.sqrt(np.finfo(float).eps):
            break
        # Apply current params to the FULL data and re-trim.
        hy = _vsn_trsf(x, rsv_par)
        whsel = _vsn_lts_select(hy, lts_quantile)

    # hoffset: per-stratum offset applied so that arrays sit on a common
    # log2 scale. R/vsn: hoffset = log2(2 * scalingFactorTransformation
    # (rowMeans(coef[,,2]))). For single stratum, coef[,,2] reshapes to
    # the b vector (length n_cols), so:
    b_coefs = rsv_par[n_cols:]
    hoffset = float(np.log2(2.0 * _scaling_factor_transform(np.mean(b_coefs))))

    hx = _vsn_trsf(x, rsv_par)  # natural-log asinh values
    return hx / np.log(2.0) - hoffset

# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   genas.R   Copyright (C) 2009-2015 Belinda Phipson, Gordon Smyth
# Python port: Copyright (C) 2026 John Mulvey
"""
``genas`` - genuine association of gene expression profiles.

Port of R limma's ``genas`` from ``limma/R/genas.R``.
Implements the multivariate-t maximum-likelihood fit (null and
alternative) and every ``subset`` branch in R's ``.whichGenes``.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, optimize, stats

from .classes import _resolve_fit_input

# ---------------------------------------------------------------------------
# Log-likelihoods ported verbatim from genas.R:108-159
# ---------------------------------------------------------------------------


def _mult_t_loglik_null(x, B, V, s, df_total, m=2) -> float:
    """Port of R ``.multTLogLikNull``: multivariate-t log-likelihood
    under the null hypothesis of no biological correlation.

    ``B`` is the per-gene coefficient matrix (n_genes, m).
    ``V`` is the technical coefficient covariance (m, m).
    ``s`` is the per-gene posterior residual variance (n_genes,).
    ``df_total`` is the per-gene total df (n_genes,) or a scalar.
    """
    a1, a2 = float(x[0]), float(x[1])
    # V0 = t(chol_mat) %*% chol_mat with chol_mat = diag(exp(a1), exp(a2))
    # → V0 = diag(exp(2*a1), exp(2*a2))
    V0 = np.array([[np.exp(2.0 * a1), 0.0], [0.0, np.exp(2.0 * a2)]])
    try:
        R = linalg.cholesky(V0 + V, lower=False)  # upper-triangular
    except linalg.LinAlgError:
        return np.inf
    second = float(np.sum(np.log(np.diag(R))))

    # backsolve(R, t(B), transpose=TRUE) ⇔ solve R' W = B' ⇔ W = (R')^-1 B'.
    # solve_triangular with lower=True treats R.T as lower-triangular.
    W = linalg.solve_triangular(R.T, B.T, lower=True)  # shape (m, n_genes)
    Q = np.sum(W**2, axis=0)  # per-gene quadratic form

    df_total = np.asarray(df_total, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    third = 0.5 * (m + df_total) * np.log1p(Q / s / df_total)

    # R sums across genes.
    return float(np.sum(second + third))


def _mult_t_loglik(x, B, V, s, df_total, m=2) -> float:
    """Port of R ``.multTLogLik``: multivariate-t log-likelihood
    allowing biological correlation via a Cholesky off-diagonal.
    """
    a1, a2, b = float(x[0]), float(x[1]), float(x[2])
    # R: L <- matrix(c(1,b,0,1),2,2) fills column-major, giving
    # column 1 = [1, b], column 2 = [0, 1] → L = [[1, 0], [b, 1]]
    L = np.array([[1.0, 0.0], [b, 1.0]])
    D = np.array([[np.exp(a1), 0.0], [0.0, np.exp(a2)]])
    V0 = L @ D @ L.T
    try:
        R = linalg.cholesky(V0 + V, lower=False)
    except linalg.LinAlgError:
        return np.inf
    second = float(np.sum(np.log(np.diag(R))))

    W = linalg.solve_triangular(R.T, B.T, lower=True)
    Q = np.sum(W**2, axis=0)

    df_total = np.asarray(df_total, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    third = 0.5 * (m + df_total) * np.log1p(Q / s / df_total)
    return float(np.sum(second + third))


# ---------------------------------------------------------------------------
# Subset rules (port of R .whichGenes, genas.R:162-218)
# ---------------------------------------------------------------------------


def _which_genes(fit_subset, subset: str):
    """Return (keep_mask, modified_coefficients_or_None).

    For "logFC" and "predFC" the coefficients are re-centred after the
    kept mask is chosen - callers should apply the returned
    coefficients to the fit before optimisation.
    """
    from .ebayes import pred_fcm
    from .utils import prop_true_null

    n_genes = fit_subset["coefficients"].shape[0]
    coef1 = np.asarray(fit_subset["coefficients"])[:, 0]
    coef2 = np.asarray(fit_subset["coefficients"])[:, 1]
    new_coef = None

    if subset == "Fpval":
        fp = np.asarray(fit_subset["F_p_value"], dtype=np.float64)
        p = 1.0 - prop_true_null(fp)
        r = stats.rankdata(fp, method="average")
        cut = p * n_genes
        keep = r <= cut
        return keep, None

    if subset == "p.union":
        p1 = 1.0 - prop_true_null(np.asarray(fit_subset["p_value"])[:, 0])
        p2 = 1.0 - prop_true_null(np.asarray(fit_subset["p_value"])[:, 1])
        cut1 = p1 * n_genes
        cut2 = p2 * n_genes
        if p1 == 0 and p2 == 0:
            return np.zeros(n_genes, dtype=bool), None
        r1 = stats.rankdata(fit_subset["p_value"][:, 0], method="average")
        r2 = stats.rankdata(fit_subset["p_value"][:, 1], method="average")
        return (r1 <= cut1) | (r2 <= cut2), None

    if subset == "p.int":
        p1 = 1.0 - prop_true_null(np.asarray(fit_subset["p_value"])[:, 0])
        p2 = 1.0 - prop_true_null(np.asarray(fit_subset["p_value"])[:, 1])
        r1 = stats.rankdata(fit_subset["p_value"][:, 0], method="average")
        r2 = stats.rankdata(fit_subset["p_value"][:, 1], method="average")
        cut1 = p1 * n_genes
        cut2 = p2 * n_genes
        return (r1 <= cut1) & (r2 <= cut2), None

    # DELIBERATE DIVERGENCE FROM R LIMMA (2026-04-20).
    # R genas.R:200-202 contains
    #
    #     fit$coeff[,1] <- sign(fit$coeff[,1]) * (abs(fit$coeff[,1]) - q1)
    #     fit$coeff[,2] <- sign(fit$coeff[,2]) * (abs(fit$coeff[,2]) - q2)
    #
    # which **appears** to re-centre the coefficients above the 90th-
    # percentile threshold before the downstream MLE. R's ``$`` does
    # partial matching on READ but not on WRITE, so ``fit$coeff[,1] <- x``
    # creates a new ``coeff`` slot instead of updating ``coefficients``.
    # The downstream likelihood (.multTLogLik at genas.R:115, 141)
    # reads ``fit$coefficients``, so R's "logFC" and "predFC" subsets
    # effectively ignore the re-centring and run the MLE on the raw
    # coefficients of the kept genes.
    #
    # pylimma applies the re-centring as the author clearly intended
    # (reading the author's intent from the surrounding code + help
    # page). Numerical output therefore diverges from R for these two
    # subsets; see docstring note and known_diff_genas_recentring.md.
    if subset == "logFC":
        q1 = float(np.quantile(np.abs(coef1), 0.9))
        q2 = float(np.quantile(np.abs(coef2), 0.9))
        keep = (np.abs(coef1) > q1) | (np.abs(coef2) > q2)
        new_coef = np.column_stack(
            [
                np.sign(coef1) * (np.abs(coef1) - q1),
                np.sign(coef2) * (np.abs(coef2) - q2),
            ]
        )
        return keep, new_coef

    if subset == "predFC":
        pfc1 = pred_fcm(fit_subset, coef=0)
        pfc2 = pred_fcm(fit_subset, coef=1)
        q1 = float(np.quantile(np.abs(pfc1), 0.9))
        q2 = float(np.quantile(np.abs(pfc2), 0.9))
        keep = (np.abs(pfc1) > q1) | (np.abs(pfc2) > q2)
        new_coef = np.column_stack(
            [
                np.sign(pfc1) * (np.abs(pfc1) - q1),
                np.sign(pfc2) * (np.abs(pfc2) - q2),
            ]
        )
        return keep, new_coef

    raise ValueError(f"Unknown subset rule: {subset!r}")


def _subset_fit_to_two_coefs(fit, coef):
    """Return a shallow MArrayLM-like dict keeping only the two chosen
    coefficient columns, with ``F`` / ``F_p_value`` recomputed from
    the subset. Matches R's ``fit <- fit[, coef]`` slicing which
    re-derives the F statistic from only the selected contrasts.
    """
    from scipy import stats as _stats

    from .classes import MArrayLM
    from .decide_tests import classify_tests_f

    new_fit = MArrayLM(fit)
    idx = list(coef)
    for slot in ("coefficients", "stdev_unscaled", "t", "p_value", "lods"):
        if new_fit.get(slot) is not None:
            arr = np.asarray(new_fit[slot])
            if arr.ndim >= 2 and arr.shape[1] >= max(idx) + 1:
                new_fit[slot] = arr[:, idx]
    cov = new_fit.get("cov_coefficients")
    if cov is not None:
        cov_arr = np.asarray(cov)
        new_fit["cov_coefficients"] = cov_arr[np.ix_(idx, idx)]

    # Re-derive F / F_p_value from the 2-col subset, matching limma's
    # ``[.MArrayLM`` behaviour (it discards and recomputes F so the
    # downstream .whichGenes "Fpval" branch sees the subset-scoped F).
    if new_fit.get("t") is not None and new_fit.get("cov_coefficients") is not None:
        f_stat, df1, df2 = classify_tests_f(new_fit, fstat_only=True)
        df2_arr = np.asarray(df2, dtype=np.float64)
        if df2_arr.ndim == 0:
            if np.isinf(df2_arr):
                fp = _stats.chi2.sf(f_stat * df1, df1)
            else:
                fp = _stats.f.sf(f_stat, df1, df2_arr)
        else:
            mask_inf = np.isinf(df2_arr)
            df2_safe = np.where(mask_inf, 1.0, df2_arr)
            fp = _stats.f.sf(f_stat, df1, df2_safe)
            if mask_inf.any():
                fp = np.where(mask_inf, _stats.chi2.sf(f_stat * df1, df1), fp)
        new_fit["F"] = f_stat
        new_fit["F_p_value"] = fp
    return new_fit


def _filter_fit_by_mask(fit, mask):
    """Return a fit subset to rows where mask is True.  Per-gene slots
    (coefs, stdev, t, p, Amean, sigma, df_residual, s2_post, df_total,
    df_prior if per-gene, s2_prior if per-gene, lods, F, F_p_value,
    genes) are sliced; scalar slots are untouched."""
    from .classes import MArrayLM

    out = MArrayLM(fit)
    for key, value in list(out.items()):
        if value is None:
            continue
        arr = np.asarray(value) if not hasattr(value, "iloc") else None
        if arr is not None and arr.ndim >= 1 and arr.shape[0] == mask.shape[0]:
            out[key] = value[mask] if arr.ndim == 1 else value[mask, ...]
        elif hasattr(value, "iloc") and len(value) == mask.shape[0]:
            out[key] = value.iloc[mask].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Public entry point (port of R genas.R:3-106)
# ---------------------------------------------------------------------------


def genas(
    fit,
    coef=(0, 1),
    subset: str = "all",
    plot: bool = False,
    alpha: float = 0.4,
    *,
    key: str = "pylimma",
) -> dict:
    """
    Estimate the biological correlation between two contrasts.

    Port of R limma's ``genas(fit, coef=c(1,2), subset="all",
    plot=FALSE, alpha=0.4)``. Fits a multivariate-t model and returns
    the technical correlation, biological covariance matrix, biological
    correlation, LR deviance, p-value, and the number of genes used.

    Parameters
    ----------
    fit : MArrayLM / dict
        Fit from :func:`lm_fit` followed by :func:`e_bayes`.
    coef : sequence of two ints, default ``(0, 1)``
        Zero-based coefficient indices (R uses 1-based).
    subset : {"all", "Fpval", "p.union", "p.int", "logFC", "predFC"}
        Gene-subset rule used to pre-filter before fitting. Matches
        R's ``.whichGenes``; ``"n"`` is accepted as a backwards-compat
        alias for ``"all"``.
    plot : bool, default False
        Ignored; pylimma does not emit the R genas plot.
    alpha : float, default 0.4
        Ignored in the non-plotting port.

    Returns
    -------
    dict with keys ``technical_correlation``, ``covariance_matrix``
    (the biological covariance ``V0 = L D L'``), ``biological_correlation``,
    ``deviance``, ``p_value``, ``n``.

    Notes
    -----
    **Deliberate divergence from R for subset="logFC" and
    subset="predFC".** R's ``genas.R`` writes its re-centred
    coefficients to a ``fit$coeff`` slot (partial match on the
    read-side only - R's ``$<-`` does not partial-match), so the
    ``fit$coefficients`` slot that feeds the downstream MLE is never
    actually re-centred. This is a latent bug in R limma; pylimma
    applies the re-centring as the author's code clearly intended.
    For ``subset in {"all", "Fpval", "p.union", "p.int"}`` pylimma
    matches R's numerical output to within optimiser tolerance.
    See ``known_diff_genas_recentring.md`` in the memory index.
    """
    from .ebayes import e_bayes
    from .utils import fit_gamma_intercept

    # Accept AnnData-stored fits: unwrap adata.uns[key] into a plain dict.
    # For MArrayLM / dict input, _resolve_fit_input is a no-op.
    fit, _adata, _adata_key = _resolve_fit_input(fit, key)

    if fit.get("cov_coefficients") is None:
        raise ValueError("fit$cov_coefficients is missing; genas needs a full-rank fit")

    out = {
        "technical_correlation": float("nan"),
        "covariance_matrix": np.full((2, 2), np.nan),
        "biological_correlation": float("nan"),
        "deviance": 0.0,
        "p_value": 1.0,
        "n": 0,
    }

    if subset == "n":
        subset = "all"  # R back-compat
    if subset not in ("all", "Fpval", "p.union", "p.int", "logFC", "predFC"):
        raise ValueError(f"Unknown subset rule: {subset!r}")

    coef = tuple(int(c) for c in coef)
    if len(coef) != 2:
        raise ValueError("coef must specify exactly two coefficients")

    if fit.get("s2_post") is None:
        fit = e_bayes(fit)

    trend = np.atleast_1d(np.asarray(fit["s2_prior"])).size > 1
    robust = np.atleast_1d(np.asarray(fit["df_prior"])).size > 1

    # Subset to the two coefs being correlated (R: fit <- fit[,coef])
    fit2 = _subset_fit_to_two_coefs(fit, coef)

    coefs = np.asarray(fit2["coefficients"], dtype=np.float64)
    cov = np.asarray(fit2["cov_coefficients"], dtype=np.float64)
    s2_post = np.asarray(fit2["s2_post"], dtype=np.float64)

    if coefs.shape[0] < 1:
        return out

    # Starting values for (a1, a2) via fitGammaIntercept on scaled
    # squared coefs, R genas.R:44-51.
    x1 = fit_gamma_intercept(coefs[:, 0] ** 2 / s2_post, offset=cov[0, 0])
    x2 = fit_gamma_intercept(coefs[:, 1] ** 2 / s2_post, offset=cov[1, 1])
    if x1 > 0 and x2 > 0:
        V0null = np.diag([x1, x2])
        C = linalg.cholesky(V0null, lower=False)
        x_start = np.log(np.diag(C))
    else:
        x_start = np.array([0.0, 0.0])
    m = 2

    # Apply subset filter to the coef-subset fit (R .whichGenes path)
    if subset != "all":
        keep_mask, new_coef = _which_genes(fit2, subset)
        if not keep_mask.any():
            return out
        fit2 = _filter_fit_by_mask(fit2, keep_mask)
        if new_coef is not None:
            fit2["coefficients"] = new_coef[keep_mask, :]
        # R re-runs eBayes on the filtered subset
        fit2 = e_bayes(fit2, trend=trend, robust=robust)

    B = np.asarray(fit2["coefficients"], dtype=np.float64)
    V = np.asarray(fit2["cov_coefficients"], dtype=np.float64)
    s = np.asarray(fit2["s2_post"], dtype=np.float64)
    df_total = np.asarray(fit2["df_total"], dtype=np.float64)
    if df_total.ndim == 0:
        df_total = np.full(B.shape[0], float(df_total))

    # Null fit (2 parameters: log-diagonal of V0)
    Q2 = optimize.minimize(
        _mult_t_loglik_null,
        x_start,
        args=(B, V, s, df_total, m),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000},
    )
    # Alternative fit (3 parameters: log-diagonals + off-diagonal)
    x_alt_start = np.array([Q2.x[0], Q2.x[1], 0.0])
    Q1 = optimize.minimize(
        _mult_t_loglik,
        x_alt_start,
        args=(B, V, s, df_total, m),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000},
    )

    a1, a2, b = float(Q1.x[0]), float(Q1.x[1]), float(Q1.x[2])
    L = np.array([[1.0, 0.0], [b, 1.0]])
    D = np.array([[np.exp(a1), 0.0], [0.0, np.exp(a2)]])
    V0 = L @ D @ L.T
    rho_biol = float(V0[1, 0] / np.sqrt(V0[0, 0] * V0[1, 1]))

    rho_tech = float(V[1, 0] / np.sqrt(V[0, 0] * V[1, 1]))

    deviance = float(abs(2.0 * (Q2.fun - Q1.fun)))
    p_val = float(stats.chi2.sf(deviance, df=1))

    out["technical_correlation"] = rho_tech
    out["covariance_matrix"] = V0
    out["biological_correlation"] = rho_biol
    out["deviance"] = deviance
    out["p_value"] = p_val
    out["n"] = int(B.shape[0])
    return out

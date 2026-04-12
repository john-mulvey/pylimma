# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   lmfit.R (lmFit, lm.series, gls.series)
#                              Copyright (C) 2003-2023 Gordon Smyth
#   lmfit.R (mrlm)             Copyright (C) 2002-2020 Gordon Smyth
#   lmEffects.R                Copyright (C) 2016-2020 Gordon Smyth
#
# mrlm() additionally ports the IRLS / Huber-M-estimation algorithm from
# the R MASS package (which limma's mrlm calls per gene):
#   MASS::rlm                  Copyright (C) Brian Ripley, Bill Venables;
#                              GPL-2 | GPL-3
# Python port: Copyright (C) 2026 John Mulvey
"""
Linear model fitting for pylimma.

Implements the core linear model fitting from limma:
- lm_fit(): main entry point for fitting linear models to expression data
- lm_series(): OLS/WLS fitting for each gene
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np
import pandas as pd
from scipy import linalg

from .dups import unwrap_dups, duplicate_correlation
from .classes import MArrayLM, get_eawp, _is_anndata, _as_matrix_weights

if TYPE_CHECKING:
    from anndata import AnnData


def _qr_r_style(x: np.ndarray, tol: float = 1e-7) -> tuple:
    """
    QR decomposition matching R's ``qr(x, LAPACK=FALSE)`` semantics.

    Unlike ``scipy.linalg.qr(pivoting=True)``, which uses LAPACK column-norm
    pivoting and can reorder columns by magnitude at every step, R's
    Linpack-based ``qr`` keeps columns in the order supplied and only
    swaps a column to the end of the active set when its residual norm
    drops below ``tol`` -- i.e. when it is found collinear with the
    preceding columns. This makes the "later of two collinear columns is
    the redundant one" rule deterministic and reproducible.

    Parameters
    ----------
    x : ndarray
        Matrix of shape (n, p).
    tol : float
        Relative tolerance on column norms. R's default is ``1e-7``.

    Returns
    -------
    q, r : ndarray
        QR factorisation of ``x[:, pivot]`` in economy mode
        (``q`` is ``(n, p)``, ``r`` is ``(p, p)``).
    pivot : ndarray of int
        Permutation of column indices. ``pivot[:rank]`` are the
        original indices of estimable columns in their original order;
        ``pivot[rank:]`` are the original indices of columns flagged
        as collinear with an earlier column.
    rank : int
        Number of estimable columns.
    """
    x = np.asarray(x, dtype=np.float64)
    n, p = x.shape
    if p == 0:
        return (np.eye(n), np.zeros((n, 0)),
                np.zeros(0, dtype=int), 0)

    # Plain (non-pivoted) economy-mode QR on the original column order
    # so that redundant columns leave |R[i, i]| at zero in their
    # original slot. Scipy's economic QR returns the n x p R, which
    # is all we need for the rank check.
    _, r_probe = linalg.qr(x, mode="economic")
    diag_abs = np.abs(np.diag(r_probe))
    scale = diag_abs.max() if diag_abs.size else 0.0
    threshold = max(tol * scale, tol)
    estimable = diag_abs > threshold

    if estimable.all():
        pivot = np.arange(p, dtype=int)
        # Full-mode QR so q is n x n and q.T @ y includes the
        # (n - p) residual rows used by downstream sigma computation.
        q, r = linalg.qr(x, mode="full")
        return q, r, pivot, p

    est_idx = np.where(estimable)[0]
    nonest_idx = np.where(~estimable)[0]
    pivot = np.concatenate([est_idx, nonest_idx]).astype(int)
    rank = int(est_idx.size)
    # Redo QR on the reordered matrix so the top-left rank x rank
    # block of R is non-singular. Full mode ensures q is n x n.
    q, r = linalg.qr(x[:, pivot], mode="full")
    return q, r, pivot, rank


def is_fullrank(x: np.ndarray) -> bool:
    """
    Check whether a matrix has full column rank.

    Parameters
    ----------
    x : ndarray
        Matrix to check.

    Returns
    -------
    bool
        True if x has full column rank.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    # eigvalsh returns eigenvalues in ascending order (smallest first)
    # R's eigen() returns them in descending order (largest first)
    # For full rank: largest eigenvalue > 0 and smallest/largest > 1e-13
    eigvals = np.linalg.eigvalsh(x.T @ x)
    return eigvals[-1] > 0 and abs(eigvals[0] / eigvals[-1]) > 1e-13


def non_estimable(
    x: np.ndarray,
    coef_names: list[str] | None = None,
) -> list[str] | None:
    """
    Check for non-estimable coefficients in a design matrix.

    Mirrors R's ``limma::nonEstimable``: runs a QR decomposition in
    original column order (``qr(x, LAPACK=FALSE)``), and if the design
    is rank-deficient returns the names of the redundant columns (the
    tail of ``qr$pivot``). When no column names are available, falls
    back to 1-based string indices to match R's
    ``colnames(x) <- as.character(1:p)`` default.

    Parameters
    ----------
    x : ndarray, DataFrame, or patsy DesignMatrix
        Design matrix. Column names are extracted when present.
    coef_names : list of str, optional
        Explicit column names. Overrides any names carried on ``x``.
        Use this when the caller has already converted the design to
        an anonymous ndarray but still has the original names.

    Returns
    -------
    list of str or None
        Names (or 1-based indices) of the non-estimable columns, or
        ``None`` when the design is full rank.
    """
    # Extract column names if not supplied explicitly.
    if coef_names is None:
        if isinstance(x, pd.DataFrame):
            coef_names = [str(c) for c in x.columns]
        else:
            di = getattr(x, "design_info", None)
            if di is not None:
                cn = getattr(di, "column_names", None)
                if cn is not None:
                    coef_names = [str(c) for c in cn]

    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    p = arr.shape[1]
    q, r, pivot, rank = _qr_r_style(arr)

    if rank >= p:
        return None

    # R's nonEstimable: colnames(x) or as.character(1:p); then
    # replace any empty name with its 1-based index.
    if coef_names is None:
        names = [str(i + 1) for i in range(p)]
    else:
        names = [
            coef_names[i] if coef_names[i] else str(i + 1)
            for i in range(p)
        ]
    return [names[pivot[i]] for i in range(rank, p)]


def lm_series(
    M: np.ndarray,
    design: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict:
    """
    Fit linear model for each gene to a series of arrays.

    Low-level function implementing OLS or WLS fitting.

    Parameters
    ----------
    M : ndarray
        Expression matrix, shape (n_genes, n_samples).
    design : ndarray
        Design matrix, shape (n_samples, n_coefficients).
    weights : ndarray, optional
        Weights matrix, shape (n_genes, n_samples) or (n_samples,).

    Returns
    -------
    dict
        coefficients : ndarray, shape (n_genes, n_coefs)
        stdev_unscaled : ndarray, shape (n_genes, n_coefs)
        sigma : ndarray, shape (n_genes,)
        df_residual : ndarray, shape (n_genes,)
        cov_coefficients : ndarray, shape (n_coefs, n_coefs)
        rank : int
        pivot : ndarray
    """
    n_genes, n_samples = M.shape
    n_coefs = design.shape[1]

    # Get coefficient names
    coef_names = [f"x{i}" for i in range(n_coefs)]

    # Normalise weight shape via R limma's asMatrixWeights logic;
    # _as_matrix_weights always returns a fresh copy, so the
    # subsequent ``weights[weights <= 0] = np.nan`` is safe
    # (see known_diff_weights_mutation.md).
    if weights is not None:
        weights = _as_matrix_weights(weights, (n_genes, n_samples))
        weights[weights <= 0] = np.nan
        M = M.copy()
        M[~np.isfinite(weights)] = np.nan

    # Check if we can do fast computation (no missing values, no probe-specific weights)
    has_missing = np.any(~np.isfinite(M))
    has_probe_weights = weights is not None and not np.allclose(
        weights, weights[0:1, :], equal_nan=True
    )

    if not has_missing and not has_probe_weights:
        # Fast path: fit all genes at once
        return _lm_series_fast(M, design, weights, coef_names)
    else:
        # Slow path: iterate over genes
        return _lm_series_slow(M, design, weights, coef_names)


def _lm_series_fast(
    M: np.ndarray,
    design: np.ndarray,
    weights: np.ndarray | None,
    coef_names: list[str],
) -> dict:
    """Fast path for lm_series when no missing values or probe weights."""
    n_genes, n_samples = M.shape
    n_coefs = design.shape[1]

    if weights is None:
        # OLS
        q, r, pivot, rank = _qr_r_style(design)

        # Solve for coefficients: design @ coef = M.T
        # Using QR: coef = R^-1 @ Q.T @ M.T
        qty = q.T @ M.T  # (n_coefs, n_genes)
        coefficients = linalg.solve_triangular(r[:rank, :rank], qty[:rank, :])

        # Reorder coefficients according to pivot
        coef_full = np.full((n_coefs, n_genes), np.nan)
        coef_full[pivot[:rank], :] = coefficients
        coefficients = coef_full.T  # (n_genes, n_coefs)

        # Residual standard deviation
        df_residual = n_samples - rank
        if df_residual > 0:
            # effects = Q.T @ M.T, residuals are effects[rank:]
            residual_effects = qty[rank:, :]
            sigma = np.sqrt(np.mean(residual_effects**2, axis=0))
        else:
            sigma = np.full(n_genes, np.nan)

        # Covariance of coefficients
        r_inv = linalg.solve_triangular(r[:rank, :rank], np.eye(rank))
        cov_coef_core = r_inv @ r_inv.T

        # Full covariance matrix with NaN for non-estimable
        cov_coefficients = np.full((n_coefs, n_coefs), np.nan)
        est_idx = pivot[:rank]
        cov_coefficients[np.ix_(est_idx, est_idx)] = cov_coef_core

        # Standard errors (unscaled by sigma)
        stdev_unscaled = np.full((n_genes, n_coefs), np.nan)
        stdev_unscaled[:, est_idx] = np.sqrt(np.diag(cov_coef_core))

    else:
        # WLS with array weights (same weights for all genes)
        w = weights[0, :]
        sqrt_w = np.sqrt(w)

        # Weight the design and data
        design_w = design * sqrt_w[:, np.newaxis]
        M_w = M * sqrt_w

        q, r, pivot, rank = _qr_r_style(design_w)

        qty = q.T @ M_w.T
        coefficients = linalg.solve_triangular(r[:rank, :rank], qty[:rank, :])

        coef_full = np.full((n_coefs, n_genes), np.nan)
        coef_full[pivot[:rank], :] = coefficients
        coefficients = coef_full.T

        df_residual = n_samples - rank
        if df_residual > 0:
            residual_effects = qty[rank:, :]
            sigma = np.sqrt(np.mean(residual_effects**2, axis=0))
        else:
            sigma = np.full(n_genes, np.nan)

        r_inv = linalg.solve_triangular(r[:rank, :rank], np.eye(rank))
        cov_coef_core = r_inv @ r_inv.T

        cov_coefficients = np.full((n_coefs, n_coefs), np.nan)
        est_idx = pivot[:rank]
        cov_coefficients[np.ix_(est_idx, est_idx)] = cov_coef_core

        stdev_unscaled = np.full((n_genes, n_coefs), np.nan)
        stdev_unscaled[:, est_idx] = np.sqrt(np.diag(cov_coef_core))

    return {
        "coefficients": coefficients,
        "stdev_unscaled": stdev_unscaled,
        "sigma": sigma,
        "df_residual": np.full(n_genes, df_residual),
        "cov_coefficients": cov_coefficients,
        "pivot": pivot,
        "rank": rank,
    }


def _lm_series_slow(
    M: np.ndarray,
    design: np.ndarray,
    weights: np.ndarray | None,
    coef_names: list[str],
) -> dict:
    """Slow path for lm_series with missing values or probe-specific weights."""
    n_genes, n_samples = M.shape
    n_coefs = design.shape[1]

    coefficients = np.full((n_genes, n_coefs), np.nan)
    stdev_unscaled = np.full((n_genes, n_coefs), np.nan)
    sigma = np.full(n_genes, np.nan)
    df_residual = np.zeros(n_genes)

    for i in range(n_genes):
        y = M[i, :]
        obs = np.isfinite(y)

        if np.sum(obs) == 0:
            continue

        X = design[obs, :]
        y_obs = y[obs]

        if weights is None:
            # OLS
            q, r, pivot, rank = _qr_r_style(X)

            if rank == 0:
                continue

            qty = q.T @ y_obs
            coef = linalg.solve_triangular(r[:rank, :rank], qty[:rank])

            coefficients[i, pivot[:rank]] = coef

            r_inv = linalg.solve_triangular(r[:rank, :rank], np.eye(rank))
            stdev_unscaled[i, pivot[:rank]] = np.sqrt(np.diag(r_inv @ r_inv.T))

            df_residual[i] = np.sum(obs) - rank
            if df_residual[i] > 0:
                resid_effects = qty[rank:]
                sigma[i] = np.sqrt(np.mean(resid_effects**2))
        else:
            # WLS
            w = weights[i, obs]
            sqrt_w = np.sqrt(w)

            X_w = X * sqrt_w[:, np.newaxis]
            y_w = y_obs * sqrt_w

            q, r, pivot, rank = _qr_r_style(X_w)

            if rank == 0:
                continue

            qty = q.T @ y_w
            coef = linalg.solve_triangular(r[:rank, :rank], qty[:rank])

            coefficients[i, pivot[:rank]] = coef

            r_inv = linalg.solve_triangular(r[:rank, :rank], np.eye(rank))
            stdev_unscaled[i, pivot[:rank]] = np.sqrt(np.diag(r_inv @ r_inv.T))

            df_residual[i] = np.sum(obs) - rank
            if df_residual[i] > 0:
                resid_effects = qty[rank:]
                sigma[i] = np.sqrt(np.mean(resid_effects**2))

    # Compute covariance matrix from full design
    q, r, pivot, rank = _qr_r_style(design)
    r_inv = linalg.solve_triangular(r[:rank, :rank], np.eye(rank))
    cov_coef_core = r_inv @ r_inv.T

    cov_coefficients = np.full((n_coefs, n_coefs), np.nan)
    est_idx = pivot[:rank]
    cov_coefficients[np.ix_(est_idx, est_idx)] = cov_coef_core

    return {
        "coefficients": coefficients,
        "stdev_unscaled": stdev_unscaled,
        "sigma": sigma,
        "df_residual": df_residual,
        "cov_coefficients": cov_coefficients,
        "pivot": pivot,
        "rank": rank,
    }


def _huber_weights(u: np.ndarray, k: float = 1.345) -> np.ndarray:
    """Compute Huber weights: psi(u)/u = 1 if |u| <= k, else k/|u|."""
    return np.where(np.abs(u) <= k, 1.0, k / np.abs(u))


def _bisquare_weights(u: np.ndarray, c: float = 4.685) -> np.ndarray:
    """Compute Tukey bisquare weights: (1 - (u/c)^2)^2 if |u| <= c, else 0."""
    return np.where(np.abs(u) <= c, (1 - (u / c) ** 2) ** 2, 0.0)


def mrlm(
    M: np.ndarray,
    design: np.ndarray | None = None,
    ndups: int = 1,
    spacing: int = 1,
    weights: np.ndarray | None = None,
    method: str = "huber",
    maxit: int = 20,
    acc: float = 1e-4,
    k: float = 1.345,
) -> dict:
    """
    Robustly fit linear model for each gene using M-estimation.

    Uses iteratively reweighted least squares with a robust loss function
    (Huber's T by default) to downweight outliers. Matches R's MASS::rlm.

    Parameters
    ----------
    M : ndarray
        Expression matrix, shape (n_genes, n_samples).
    design : ndarray, optional
        Design matrix, shape (n_samples, n_coefficients).
        If None, uses intercept-only model.
    ndups : int, default 1
        Number of within-array duplicate spots.
    spacing : int, default 1
        Spacing between duplicate spots.
    weights : ndarray, optional
        Prior weights for observations.
    method : str, default "huber"
        Robust estimation method. Options: "huber", "bisquare".
    maxit : int, default 20
        Maximum iterations for IRLS (matches R's default).
    acc : float, default 1e-4
        Convergence tolerance (matches R's default).
    k : float, default 1.345
        Tuning constant for Huber estimator (matches R's default).

    Returns
    -------
    dict
        coefficients : ndarray, shape (n_genes, n_coefs)
        stdev_unscaled : ndarray, shape (n_genes, n_coefs)
        sigma : ndarray, shape (n_genes,)
        df_residual : ndarray, shape (n_genes,)
        cov_coefficients : ndarray, shape (n_coefs, n_coefs)

    Notes
    -----
    This function implements R's MASS::rlm algorithm exactly:
    - MAD scale estimation: median(|resid|) / 0.6745
    - Huber weights: psi(u)/u where psi is the Huber function
    - Convergence on residuals: sqrt(sum((old - new)^2) / sum(old^2))

    References
    ----------
    Huber, P. J. (1981). Robust Statistics. Wiley.
    Venables, W. N. & Ripley, B. D. (2002). Modern Applied Statistics with S.
    """
    M = np.asarray(M, dtype=np.float64)
    n_genes, n_samples = M.shape

    # Check design
    if design is None:
        design = np.ones((n_samples, 1))
    design = np.asarray(design, dtype=np.float64)
    n_coefs = design.shape[1]

    # Normalise weight shape through asMatrixWeights and copy-guard
    # the subsequent NaN write (see lm_series header comment).
    if weights is not None:
        weights = _as_matrix_weights(weights, (n_genes, n_samples))
        weights[weights <= 0] = np.nan
        M = M.copy()
        M[~np.isfinite(weights)] = np.nan

    # Unwrap duplicates if needed
    if ndups > 1:
        M = unwrap_dups(M, ndups=ndups, spacing=spacing)
        design = np.kron(design, np.ones((ndups, 1)))
        if weights is not None:
            weights = unwrap_dups(weights, ndups=ndups, spacing=spacing)
        n_genes, n_samples = M.shape

    # Select weight function
    if method.lower() == "huber":
        weight_fn = lambda u: _huber_weights(u, k)
    elif method.lower() == "bisquare":
        weight_fn = lambda u: _bisquare_weights(u, c=4.685)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'huber' or 'bisquare'")

    # Initialize output
    coefficients = np.full((n_genes, n_coefs), np.nan)
    stdev_unscaled = np.full((n_genes, n_coefs), np.nan)
    sigma = np.full(n_genes, np.nan)
    df_residual = np.zeros(n_genes)

    # Fit each gene using R's exact IRLS algorithm
    for i in range(n_genes):
        y = M[i, :]
        obs = np.isfinite(y)

        if np.sum(obs) <= n_coefs:
            continue

        y_obs = y[obs]
        X = design[obs, :]

        # Apply prior weights if provided
        if weights is not None:
            prior_w = weights[i, obs]
            sqrt_prior_w = np.sqrt(prior_w)
            y_work = y_obs * sqrt_prior_w
            X_work = X * sqrt_prior_w[:, np.newaxis]
        else:
            prior_w = None
            y_work = y_obs
            X_work = X

        # OLS initial fit
        coef, _, rank_fit, _ = np.linalg.lstsq(X_work, y_work, rcond=None)
        resid = y_work - X_work @ coef

        # IRLS iterations (matching R's MASS::rlm)
        # R computes scale at START of each iteration and returns that value
        scale = np.median(np.abs(resid)) / 0.6745  # Initial scale
        for _ in range(maxit):
            # MAD scale estimate (computed at start of iteration)
            scale = np.median(np.abs(resid)) / 0.6745
            if scale == 0:
                break

            # Compute robust weights
            u = resid / scale
            w = weight_fn(u)

            # Weighted least squares
            sqrt_w = np.sqrt(w)
            Xw = X_work * sqrt_w[:, np.newaxis]
            yw = y_work * sqrt_w
            new_coef = np.linalg.lstsq(Xw, yw, rcond=None)[0]
            new_resid = y_work - X_work @ new_coef

            # Convergence on residuals (R's irls.delta)
            conv = np.sqrt(
                np.sum((resid - new_resid) ** 2) / max(1e-20, np.sum(resid**2))
            )

            coef = new_coef
            resid = new_resid

            if conv <= acc:
                break

        coefficients[i, :] = coef

        # Unscaled standard errors from QR decomposition of final weighted design
        # R's MASS::rlm uses the QR from final IRLS iteration (with robust weights)
        Xw_final = X_work * sqrt_w[:, np.newaxis]
        q, r, piv, rank = _qr_r_style(Xw_final)
        if rank > 0:
            r_inv = linalg.solve_triangular(r[:rank, :rank], np.eye(rank))
            stdev_unscaled[i, piv[:rank]] = np.sqrt(np.diag(r_inv @ r_inv.T))

        df_residual[i] = len(y_obs) - rank
        if df_residual[i] > 0:
            # Use the scale from the last iteration (R's behaviour)
            # R returns the scale computed at the START of the final iteration
            sigma[i] = scale

    # Compute covariance from full design
    q, r, pivot, rank = _qr_r_style(design)
    r_inv = linalg.solve_triangular(r[:rank, :rank], np.eye(rank))
    cov_coef_core = r_inv @ r_inv.T

    cov_coefficients = np.full((n_coefs, n_coefs), np.nan)
    est_idx = pivot[:rank]
    cov_coefficients[np.ix_(est_idx, est_idx)] = cov_coef_core

    return {
        "coefficients": coefficients,
        "stdev_unscaled": stdev_unscaled,
        "sigma": sigma,
        "df_residual": df_residual,
        "cov_coefficients": cov_coefficients,
        "rank": rank,
    }


def gls_series(
    M: np.ndarray,
    design: np.ndarray | None = None,
    ndups: int = 2,
    spacing: int = 1,
    block: np.ndarray | None = None,
    correlation: float | None = None,
    weights: np.ndarray | None = None,
) -> dict:
    """
    Fit linear model for each gene using generalized least squares.

    Allows for correlation between samples, either through duplicate spots
    within arrays or through blocking factors.

    Parameters
    ----------
    M : ndarray
        Expression matrix, shape (n_genes, n_samples).
    design : ndarray, optional
        Design matrix, shape (n_samples, n_coefficients).
        If None, uses intercept-only model.
    ndups : int, default 1
        Number of within-array duplicate spots.
    spacing : int, default 1
        Spacing between duplicate spots.
    block : array_like, optional
        Block indicator for correlated samples. If provided, ndups and
        spacing are ignored.
    correlation : float, optional
        Intra-block correlation. If None, will need to be estimated
        externally (e.g., via duplicate_correlation()).
    weights : ndarray, optional
        Observation weights.

    Returns
    -------
    dict
        coefficients : ndarray, shape (n_genes, n_coefs)
        stdev_unscaled : ndarray, shape (n_genes, n_coefs)
        sigma : ndarray, shape (n_genes,)
        df_residual : ndarray, shape (n_genes,)
        cov_coefficients : ndarray, shape (n_coefs, n_coefs)
        correlation : float
        block : ndarray or None
        ndups : int
        spacing : int

    Notes
    -----
    This function uses Cholesky decomposition to transform the GLS problem
    to an equivalent OLS problem. The correlation structure is either:

    - Within-array duplicates (ndups > 1): spots within the same array
      are correlated with the specified correlation.
    - Between-sample blocks (block != None): samples within the same
      block are correlated.

    References
    ----------
    Smyth, G. K., Michaud, J. and Scott, H. S. (2005). Use of within-array
    replicate spots for assessing differential expression in microarray
    experiments. Bioinformatics, 21, 2067-2075.
    """
    M = np.asarray(M, dtype=np.float64)
    n_genes, n_arrays = M.shape

    # Check design
    if design is None:
        design = np.ones((n_arrays, 1))
    design = np.asarray(design, dtype=np.float64)
    if design.shape[0] != n_arrays:
        raise ValueError("Number of rows of design matrix does not match number of arrays")
    n_coefs = design.shape[1]
    coef_names = [f"x{i}" for i in range(n_coefs)]

    # Check correlation - auto-estimate if not provided (R parity)
    if correlation is None:
        dc_result = duplicate_correlation(
            M, design=design, ndups=ndups, spacing=spacing,
            block=block, weights=weights
        )
        correlation = dc_result["consensus_correlation"]
    if abs(correlation) >= 1:
        raise ValueError("correlation is 1 or -1, so the model is degenerate")

    # Normalise weight shape through asMatrixWeights; helper returns
    # a fresh copy so the subsequent NaN/zero writes are safe.
    if weights is not None:
        weights = _as_matrix_weights(weights, (n_genes, n_arrays))
        weights[np.isnan(weights)] = 0
        M = M.copy()
        M[weights < 1e-15] = np.nan
        weights[weights < 1e-15] = np.nan

    # Build correlation matrix
    if block is None:
        # Within-array duplicates
        if ndups < 2:
            warnings.warn("No duplicates (ndups < 2)")
            ndups = 1
            correlation = 0

        # Correlation matrix: arrays x arrays with duplicates
        # Each array has ndups spots that are correlated
        corr_block = np.full((ndups, ndups), correlation)
        np.fill_diagonal(corr_block, 1.0)
        cormatrix = linalg.block_diag(*[corr_block for _ in range(n_arrays)])

        # Unwrap duplicates
        M = unwrap_dups(M, ndups=ndups, spacing=spacing)
        if weights is not None:
            weights = unwrap_dups(weights, ndups=ndups, spacing=spacing)

        # Expand design for duplicates
        design = np.kron(design, np.ones((ndups, 1)))
        n_genes, n_samples = M.shape
    else:
        # Correlated samples (blocking)
        ndups = 1
        spacing = 1
        block = np.asarray(block)
        if len(block) != n_arrays:
            raise ValueError("Length of block does not match number of arrays")

        # Build correlation matrix from block structure
        unique_blocks = np.unique(block)
        n_blocks = len(unique_blocks)
        Z = np.array([[block[i] == ub for ub in unique_blocks] for i in range(n_arrays)], dtype=float)
        cormatrix = Z @ (correlation * Z.T)
        np.fill_diagonal(cormatrix, 1.0)
        n_samples = n_arrays

    # Initialize output
    stdev_unscaled = np.full((n_genes, n_coefs), np.nan)

    # Check if fast computation is possible
    has_missing = np.any(~np.isfinite(M))
    has_probe_weights = weights is not None and not np.allclose(
        weights, weights[0:1, :], equal_nan=True
    )

    if not has_missing and not has_probe_weights:
        # Fast path: fit all genes at once
        V = cormatrix.copy()
        if weights is not None:
            # Array weights
            wrs = 1.0 / np.sqrt(weights[0, :])
            V = wrs[:, np.newaxis] * V * wrs[np.newaxis, :]

        # Cholesky decomposition
        chol_V = linalg.cholesky(V, lower=False)

        # Transform data: y* = L^-T @ y
        y = linalg.solve_triangular(chol_V, M.T, trans='T')  # (n_samples, n_genes)

        # Transform design: X* = L^-T @ X
        X = linalg.solve_triangular(chol_V, design, trans='T')

        # OLS on transformed data
        q, r, pivot, rank = _qr_r_style(X)

        # Coefficients
        qty = q.T @ y  # (n_coefs, n_genes)
        coefficients = linalg.solve_triangular(r[:rank, :rank], qty[:rank, :])

        coef_full = np.full((n_coefs, n_genes), np.nan)
        coef_full[pivot[:rank], :] = coefficients
        coefficients = coef_full.T

        # Residual standard deviation
        df_residual = n_samples - rank
        if df_residual > 0:
            sigma = np.sqrt(np.mean(qty[rank:, :]**2, axis=0))
        else:
            sigma = np.full(n_genes, np.nan)

        # Covariance of coefficients
        r_inv = linalg.solve_triangular(r[:rank, :rank], np.eye(rank))
        cov_coef_core = r_inv @ r_inv.T

        cov_coefficients = np.full((n_coefs, n_coefs), np.nan)
        est_idx = pivot[:rank]
        cov_coefficients[np.ix_(est_idx, est_idx)] = cov_coef_core

        stdev_unscaled[:, est_idx] = np.sqrt(np.diag(cov_coef_core))

        return {
            "coefficients": coefficients,
            "stdev_unscaled": stdev_unscaled,
            "sigma": sigma,
            "df_residual": np.full(n_genes, df_residual),
            "cov_coefficients": cov_coefficients,
            "pivot": pivot,
            "rank": rank,
            "ndups": ndups,
            "spacing": spacing,
            "block": block,
            "correlation": correlation,
        }

    # Slow path: iterate over genes
    coefficients = np.full((n_genes, n_coefs), np.nan)
    sigma = np.full(n_genes, np.nan)
    df_residual = np.zeros(n_genes)

    for i in range(n_genes):
        y = M[i, :]
        obs = np.isfinite(y)

        if np.sum(obs) == 0:
            continue

        y_obs = y[obs]
        X = design[obs, :]
        V = cormatrix[np.ix_(obs, obs)]

        if weights is not None:
            wrs = 1.0 / np.sqrt(weights[i, obs])
            V = wrs[:, np.newaxis] * V * wrs[np.newaxis, :]

        # Cholesky decomposition
        try:
            chol_V = linalg.cholesky(V, lower=False)
        except linalg.LinAlgError:
            # Matrix not positive definite
            continue

        # Transform
        y_t = linalg.solve_triangular(chol_V, y_obs, trans='T')

        if np.all(X == 0):
            n = len(y_obs)
            df_residual[i] = n
            sigma[i] = np.sqrt(np.mean(y_t**2))
        else:
            X_t = linalg.solve_triangular(chol_V, X, trans='T')

            # OLS on transformed
            q, r, pivot, rank = _qr_r_style(X_t)

            if rank == 0:
                continue

            qty = q.T @ y_t
            coef = linalg.solve_triangular(r[:rank, :rank], qty[:rank])
            coefficients[i, pivot[:rank]] = coef

            r_inv = linalg.solve_triangular(r[:rank, :rank], np.eye(rank))
            stdev_unscaled[i, pivot[:rank]] = np.sqrt(np.diag(r_inv @ r_inv.T))

            df_residual[i] = len(y_obs) - rank
            if df_residual[i] > 0:
                # Residual SS from QR decomposition: qty[rank:] are residual effects
                # This avoids dimension issues when rank < n_coefs
                sigma[i] = np.sqrt(np.sum(qty[rank:]**2) / df_residual[i])

    # Compute covariance from full correlation matrix
    chol_V = linalg.cholesky(cormatrix, lower=False)
    X_t = linalg.solve_triangular(chol_V, design, trans='T')
    q, r, pivot, rank = _qr_r_style(X_t)
    r_inv = linalg.solve_triangular(r[:rank, :rank], np.eye(rank))
    cov_coef_core = r_inv @ r_inv.T

    cov_coefficients = np.full((n_coefs, n_coefs), np.nan)
    est_idx = pivot[:rank]
    cov_coefficients[np.ix_(est_idx, est_idx)] = cov_coef_core

    return {
        "coefficients": coefficients,
        "stdev_unscaled": stdev_unscaled,
        "sigma": sigma,
        "df_residual": df_residual,
        "cov_coefficients": cov_coefficients,
        "pivot": pivot,
        "rank": rank,
        "ndups": ndups,
        "spacing": spacing,
        "block": block,
        "correlation": correlation,
    }


def _parse_design(
    design,
    data: pd.DataFrame | None = None,
    n_samples: int | None = None,
) -> tuple[np.ndarray, list[str] | None]:
    """
    Parse design matrix from various input types.

    Parameters
    ----------
    design : array_like or str or None
        Design matrix as numpy array, DataFrame, patsy ``DesignMatrix``,
        or formula string like ``"~ group + batch"``.
    data : DataFrame, optional
        Sample metadata for formula parsing.
    n_samples : int, optional
        Number of samples (for default intercept-only design).

    Returns
    -------
    design : ndarray
        Design matrix.
    names : list of str or None
        Column names if the input carried them (DataFrame columns,
        patsy ``design_info.column_names``, or formula output). None
        otherwise; callers should fall back to 1-based string indices
        matching R's ``colnames(x) <- as.character(1:p)``.
    """
    names: list[str] | None = None

    if design is None:
        if n_samples is None:
            raise ValueError("Must provide design or n_samples")
        return np.ones((n_samples, 1)), None

    if isinstance(design, str):
        # Formula string - use patsy
        try:
            import patsy
        except ImportError:
            raise ImportError(
                "patsy is required for formula-based design matrices. "
                "Install with: pip install patsy"
            )
        if data is None:
            raise ValueError("data must be provided when design is a formula string")
        dm = patsy.dmatrix(design, data)
        try:
            names = [str(c) for c in dm.design_info.column_names]
        except AttributeError:
            names = None
        return np.asarray(dm), names

    if isinstance(design, pd.DataFrame):
        names = [str(c) for c in design.columns]
    else:
        di = getattr(design, "design_info", None)
        if di is not None:
            cn = getattr(di, "column_names", None)
            if cn is not None:
                names = [str(c) for c in cn]

    arr = np.asarray(design, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr, names


def lm_fit(
    data,
    design=None,
    ndups: int | None = None,
    spacing: int | None = None,
    block: np.ndarray | None = None,
    correlation: float | None = None,
    weights: np.ndarray | None = None,
    method: str = "ls",
    key: str = "pylimma",
    layer: str | None = None,
    weights_layer: str | None = None,
) -> dict | None:
    """
    Fit linear models to expression data.

    This is the main entry point for fitting gene-wise linear models.
    Accepts either an AnnData object or a numpy array/DataFrame.

    Parameters
    ----------
    data : AnnData, ndarray, or DataFrame
        Expression data. If AnnData, reads from adata.X (samples x genes)
        or specified layer, and stores results in adata.uns[key].
        If ndarray or DataFrame, expects (n_genes, n_samples) and returns
        results as a dict.

        **Important:** Expression values must be normalised and log-transformed
        before calling this function.
    design : ndarray or str, optional
        Design matrix (n_samples, n_coefficients) or formula string
        like "~ group + batch". If None, uses intercept-only model.
    ndups : int, default 1
        Number of within-array duplicate spots.
    spacing : int, default 1
        Spacing between duplicate spots in the expression matrix.
    block : array_like, optional
        Block indicator for correlated samples. When provided, samples
        within the same block are assumed to be correlated.
    correlation : float, optional
        Intra-block or intra-duplicate correlation. Required when
        ndups > 1 or block is provided. Use duplicate_correlation()
        to estimate this value.
    weights : ndarray, optional
        Observation weights. Can be:
        - 1D array of length n_samples (array weights)
        - 2D array of shape (n_genes, n_samples) (gene-specific weights)
    method : str, default "ls"
        Fitting method. Options:
        - "ls": least squares (default)
        - "robust": robust regression using M-estimation
    key : str, default "pylimma"
        Key for storing results in adata.uns (AnnData input only).
    layer : str, optional
        Layer to use for expression data (AnnData input only).
        If None, uses adata.X.
    weights_layer : str, optional
        AnnData-only. Layer to read as observation weights. When
        ``None`` (default) and ``layer`` ends in ``"_E"``, the companion
        layer ``{layer[:-2]}_weights`` is auto-loaded if present (voom /
        vooma convention). Set this when ``voom``/``vooma`` were called
        with a non-default ``weights_layer=`` so the read side matches.

    Returns
    -------
    dict or None
        If input is ndarray/DataFrame, returns dict with fit results.
        If input is AnnData, stores results in adata.uns[key] and returns None.

    Notes
    -----
    The function dispatches to different fitting algorithms based on parameters:

    - If method="robust", uses mrlm() for robust M-estimation
    - If ndups < 2 and block is None, uses lm_series() for simple OLS/WLS
    - If ndups >= 2 or block is provided, uses gls_series() for GLS

    The fit results include:
    - coefficients: estimated coefficients (n_genes, n_coefs)
    - stdev_unscaled: unscaled standard errors (n_genes, n_coefs)
    - sigma: residual standard deviation (n_genes,)
    - df_residual: residual degrees of freedom (n_genes,)
    - cov_coefficients: coefficient covariance matrix (n_coefs, n_coefs)
    - Amean: mean expression per gene (n_genes,)
    - design: the design matrix used

    References
    ----------
    Smyth, G. K. (2004). Linear models and empirical Bayes methods for
    assessing differential expression in microarray experiments.
    Statistical Applications in Genetics and Molecular Biology, 3(1), Article 3.
    """
    # Polymorphic input dispatch
    is_anndata = _is_anndata(data)
    adata = data if is_anndata else None

    eawp = get_eawp(data, layer=layer, weights_layer=weights_layer)
    expr = eawp["exprs"]

    # Pick up weights from the input object if not passed explicitly.
    # For EList this surfaces ``obj["weights"]``; for AnnData it surfaces
    # a companion ``{stem}_weights`` layer auto-loaded by get_eawp.
    if weights is None and eawp.get("weights") is not None:
        weights = np.asarray(eawp["weights"], dtype=np.float64)

    # Resolve ndups/spacing from printer metadata when not explicitly
    # supplied, matching R's lmFit (lmfit.R:47-50):
    #   if(is.null(ndups)) ndups <- y$printer$ndups
    #   if(is.null(spacing)) spacing <- y$printer$spacing
    def _printer_attr(obj, key, default=1):
        printer = None
        if hasattr(obj, "printer"):
            printer = getattr(obj, "printer")
        elif isinstance(obj, dict):
            printer = obj.get("printer")
        if isinstance(printer, dict):
            value = printer.get(key)
            if value is not None:
                return int(value)
        return default

    if ndups is None:
        ndups = _printer_attr(data, "ndups", 1)
    if spacing is None:
        spacing = _printer_attr(data, "spacing", 1)

    # Extract metadata needed downstream (gene names and sample data). get_eawp
    # gives us probes/targets as DataFrames but lm_fit historically extracted
    # gene_names from DataFrame.index and sample_data from adata.obs - preserve
    # both conventions here.
    if is_anndata:
        sample_data = adata.obs
        gene_names = list(adata.var_names) if adata.var_names is not None else None
    elif isinstance(data, pd.DataFrame):
        sample_data = None
        gene_names = list(data.index)
    else:
        sample_data = None
        gene_names = None

    if expr.ndim != 2:
        raise ValueError("Expression data must be 2-dimensional")

    n_genes, n_samples = expr.shape

    # Fall back to the design carried on the input object before
    # defaulting to an intercept, matching R's lmFit one-liner:
    #   if(is.null(design)) design <- y$design
    # EList populates eawp["design"] via _eawp_from_elist_like; the
    # AnnData branch of get_eawp populates it from
    # adata.uns[<stem>]["design"] when called with layer="<stem>_E".
    if design is None and eawp.get("design") is not None:
        design = eawp["design"]

    # Parse design matrix. Column names (if any) flow through to
    # non_estimable so the "Coefficients not estimable" warning
    # identifies columns by name rather than by index when possible.
    design, design_names = _parse_design(
        design, data=sample_data, n_samples=n_samples
    )

    if design.shape[0] != n_samples:
        raise ValueError(
            f"Design matrix has {design.shape[0]} rows but expression data has {n_samples} samples"
        )

    # Check for non-estimable coefficients
    ne = non_estimable(design, coef_names=design_names)
    if ne is not None:
        warnings.warn(f"Coefficients not estimable: {', '.join(ne)}")

    # Validate method (matches R's match.arg)
    if method not in ("ls", "robust"):
        raise ValueError(
            f"method '{method}' not recognized. Must be 'ls' or 'robust'."
        )

    # Validate correlation requirement. mrlm (robust) does not use
    # correlation - duplicates are unwrapped and fit per-gene via M-estimation
    # - so correlation is only required for the GLS (ls) path.
    if method == "ls" and (ndups >= 2 or block is not None) and correlation is None:
        raise ValueError(
            "correlation must be provided when ndups >= 2 or block is specified. "
            "Use duplicate_correlation() to estimate it."
        )

    # Dispatch to appropriate fitting function
    if method == "robust":
        # R warns when robust is combined with blocking or within-array
        # duplicates (correlation cannot be combined with robust regression)
        # but still calls mrlm. Match that behaviour.
        if block is not None or ndups > 1:
            warnings.warn(
                "Correlation cannot be combined with robust regression. "
                "If you wish to use blocking or duplicate correlation, "
                "then use least squares regression."
            )
        fit = mrlm(expr, design, ndups=ndups, spacing=spacing, weights=weights)
    elif ndups < 2 and block is None:
        # Simple OLS or WLS
        fit = lm_series(expr, design, weights=weights)
    else:
        # GLS for correlated samples (duplicates or blocking)
        fit = gls_series(
            expr,
            design,
            ndups=ndups,
            spacing=spacing,
            block=block,
            correlation=correlation,
            weights=weights,
        )

    # Promote to MArrayLM and add post-fit slots
    fit = MArrayLM(fit)
    fit["Amean"] = np.nanmean(expr, axis=1)
    fit["design"] = design
    fit["genes"] = gene_names
    # Propagate sample metadata (y$targets on an EList, adata.obs on an
    # AnnData). R's lmFit does the same via fit$targets <- y$targets.
    # Downstream diagnostic plots (plotSA / plotMDS) expect it.
    if eawp.get("targets") is not None:
        fit["targets"] = eawp["targets"]

    if is_anndata:
        # Store as plain dict so adata.write_h5ad() works. anndata's
        # IO registry dispatches on exact type, not isinstance, so the
        # MArrayLM subclass trips `IORegistryError`. Users can rewrap
        # via pylimma.MArrayLM(adata.uns[key]) if they need the
        # class's method API back.
        adata.uns[key] = dict(fit)
        return None
    else:
        return fit

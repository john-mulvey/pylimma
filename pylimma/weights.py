# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   arrayWeights.R            Copyright (C) 2005-2019 Matt Ritchie, Cynthia Liu,
#                                                     Gordon Smyth
#   arrayWeightsREML.R        Copyright (C) 2005-2019 Gordon Smyth
#   arrayWeightsPrWtsREML.R   Copyright (C) 2019      Gordon Smyth
#   arrayWeightsGeneByGene.R  Copyright (C) 2005-2020 Matt Ritchie, Cynthia Liu,
#                                                     Gordon Smyth
#   arrayWeightsQuick.R       Copyright (C) 2004      Gordon Smyth
#   weights.R (modifyWeights) Copyright (C) 2003-2020 Gordon Smyth
# Python port: Copyright (C) 2026 John Mulvey
"""
Array weights for limma.

Implements functions to estimate sample-specific quality weights:
- array_weights(): Main entry point for estimating array quality weights
- array_weights_quick(): Quick approximation to array weights
- modify_weights(): Multiply rows of a weights matrix by status-specific
  scalars
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from .classes import get_eawp


def _contr_sum(n: int) -> np.ndarray:
    """
    Create sum-to-zero contrast matrix.

    Equivalent to R's contr.sum(n).

    Parameters
    ----------
    n : int
        Number of levels.

    Returns
    -------
    ndarray
        Contrast matrix of shape (n, n-1).
    """
    if n < 2:
        raise ValueError("Need at least 2 levels for contrast coding")
    # Identity matrix minus last row
    mat = np.eye(n, n - 1)
    mat[-1, :] = -1
    return mat


def _hat_values(qr_result: tuple) -> np.ndarray:
    """
    Compute hat values (diagonal of hat matrix) from QR decomposition.

    Parameters
    ----------
    qr_result : tuple
        Result from np.linalg.qr or scipy.linalg.qr.

    Returns
    -------
    ndarray
        Hat values.
    """
    Q = qr_result[0]
    # Hat values are row sums of squares of Q
    return np.sum(Q**2, axis=1)


def _array_weights_gene_by_gene(
    E: np.ndarray,
    design: np.ndarray,
    weights: np.ndarray | None,
    var_design: np.ndarray,
    prior_n: float = 10,
    trace: bool = False,
) -> np.ndarray:
    """
    Estimate array variances via gene-by-gene update algorithm.

    Internal function implementing the gene-by-gene method for array weights.

    Parameters
    ----------
    E : ndarray
        Expression matrix, shape (n_genes, n_samples).
    design : ndarray
        Design matrix, shape (n_samples, n_coefficients).
    weights : ndarray or None
        Prior observation weights, shape (n_genes, n_samples).
    var_design : ndarray
        Variance design matrix (columns sum to zero).
    prior_n : float
        Prior sample size for regularization.

    Returns
    -------
    ndarray
        Array weights, shape (n_samples,).
    """
    n_genes, n_samples = E.shape

    # Z matrix: intercept + var_design
    Z = np.column_stack([np.ones(n_samples), var_design])
    n_gam = var_design.shape[1]

    # Initialize array gammas to zero (with prior weight)
    gam = np.zeros(n_gam)
    aw = np.ones(n_samples)

    # Prior information matrix
    info2 = prior_n * var_design.T @ var_design

    # trace: report weight range roughly 10 times across genes (matches R)
    if trace:
        print("gene range(w)")
        report_interval = max(n_genes // 10, 1)

    # Step through genes
    for i in range(n_genes):
        # Combine array weights with observation weights
        if weights is None:
            w = aw.copy()
        else:
            w = aw * weights[i, :]

        y = E[i, :]

        # Handle missing values
        if np.any(np.isnan(y)):
            obs = np.isfinite(y)
            n_obs = np.sum(obs)
            if n_obs <= 2:
                continue

            X = design[obs, :]
            y_obs = y[obs]
            w_obs = w[obs]

            # Weighted least squares fit
            sqrt_w = np.sqrt(w_obs)
            X_w = X * sqrt_w[:, np.newaxis]
            y_w = y_obs * sqrt_w

            try:
                # Use full QR decomposition to get residual effects
                q, r = linalg.qr(X_w, mode="full")
                r_diag = np.diag(r[: X_w.shape[1], :])
                rank = np.sum(np.abs(r_diag) > 1e-10)
                df_resid = n_obs - rank
                if df_resid < 2:
                    continue

                qty = q.T @ y_w

                # Hat values from economic Q
                q_econ = q[:, : X_w.shape[1]]
                h1 = np.zeros(n_samples)
                h1[obs] = 1 - np.sum(q_econ**2, axis=1)

                # Weighted squared residuals
                resid_effects = qty[:rank]
                fitted = q[:, :rank] @ resid_effects
                resid = (y_w - fitted) / sqrt_w

                d = np.zeros(n_samples)
                d[obs] = w_obs * resid**2

                # Residual variance from effects beyond the fitted
                effects = qty[rank:]
                s2 = np.mean(effects**2)
            except (np.linalg.LinAlgError, ValueError):
                continue
        else:
            # No missing values - fit all at once
            sqrt_w = np.sqrt(w)
            X_w = design * sqrt_w[:, np.newaxis]
            y_w = y * sqrt_w

            try:
                # Use full QR decomposition
                q, r = linalg.qr(X_w, mode="full")
                r_diag = np.diag(r[: design.shape[1], :])
                rank = np.sum(np.abs(r_diag) > 1e-10)
                df_resid = n_samples - rank
                if df_resid < 2:
                    continue

                qty = q.T @ y_w

                # Hat values from economic Q
                q_econ = q[:, : design.shape[1]]
                h1 = 1 - np.sum(q_econ**2, axis=1)

                # Residuals
                fitted = q[:, :rank] @ qty[:rank]
                resid = (y_w - fitted) / sqrt_w
                d = w * resid**2

                # Residual variance
                effects = qty[rank:]
                s2 = np.mean(effects**2)
            except (np.linalg.LinAlgError, ValueError):
                continue

        if s2 < 1e-15:
            continue

        # Update information matrix
        # info = t(Z) %*% (h1 * Z)
        info = Z.T @ (h1[:, np.newaxis] * Z)

        # Schur complement update
        info2 = info2 + info[1:, 1:] - np.outer(info[1:, 0], info[0, 1:]) / info[0, 0]

        # Score
        z = d / s2 - h1
        dl = var_design.T @ z

        # Update gamma
        try:
            gam = gam + np.linalg.solve(info2, dl)
        except np.linalg.LinAlgError:
            continue

        # Update array weights
        aw = np.exp(var_design @ (-gam))

        if trace and (i + 1) % report_interval == 0:
            print(f"{i + 1} {aw.min():.6g} {aw.max():.6g}")

    return aw


def _array_weights_reml(
    E: np.ndarray,
    design: np.ndarray,
    var_design: np.ndarray,
    prior_n: float = 10,
    maxiter: int = 50,
    tol: float = 1e-5,
    trace: bool = False,
) -> np.ndarray:
    """
    Estimate array weights by REML.

    Faithful port of limma's .arrayWeightsREML (limma/R/arrayWeightsREML.R).
    Uses an exact Fisher scoring algorithm similar to statmod::remlscor:

    - Initial unweighted fit filters genes with zero residual variance.
    - Fisher information built from pairwise products of Q = QR-fit basis columns.
    - Score includes prior.n support toward w=1.
    - Convergence on score-step inner product, rescaled by ngam*(ngenes+prior.n).

    Parameters
    ----------
    E : ndarray
        Expression matrix, shape (n_genes, n_samples). Assumes no NAs or
        infinite values (caller handles filtering).
    design : ndarray
        Design matrix, shape (n_samples, p).
    var_design : ndarray
        Variance design matrix Z2 (columns sum to zero, excludes intercept).
    prior_n : float
        Prior sample size pulling weights toward 1.
    maxiter : int
        Maximum Fisher scoring iterations.
    tol : float
        Convergence tolerance on rescaled score-step inner product.
    """
    n_genes, n_samples = E.shape
    Z2 = var_design
    n_gam = Z2.shape[1]
    Z = np.column_stack([np.ones(n_samples), Z2])
    p = design.shape[1]
    p2 = (p * (p + 1)) // 2

    # Starting values
    gam = np.zeros(n_gam)
    w = np.ones(n_samples)
    convcrit_last = np.inf

    if trace:
        print("iter convcrit range(w)")

    # Initial unweighted fit to detect zero-variance genes and seed residuals.
    # Use full Q (narrays x narrays) so we can access the residual effects rows.
    q_u_full, r_u = linalg.qr(design, mode="full")
    rank_u = int(np.sum(np.abs(np.diag(r_u[:p, :])) > 1e-10))
    q_u = q_u_full[:, :p]  # first p cols = column space basis
    effects_u = q_u_full.T @ E.T  # (narrays, ngenes)
    effects_null = effects_u[rank_u:, :]  # residual effects
    s2 = np.mean(effects_null**2, axis=0)

    zero_var_filter = None
    if np.min(s2) < 1e-15:
        zero_var_filter = s2 >= 1e-15
        E = E[zero_var_filter, :]
        n_genes = E.shape[0]
        if n_genes < 2:
            return w
        s2 = s2[zero_var_filter]
        effects_u = effects_u[:, zero_var_filter]

    # Residuals (original scale): y - X beta = y - Q[:,:rank] @ Q[:,:rank].T @ y
    residuals = E - (q_u[:, :rank_u] @ effects_u[:rank_u, :]).T
    qr_factors = (q_u, rank_u)

    for iteration in range(1, maxiter + 1):
        if iteration > 1:
            sw = np.sqrt(w)
            design_w = design * sw[:, np.newaxis]
            E_w = E * sw
            q_full, r_it = linalg.qr(design_w, mode="full")
            rank_it = int(np.sum(np.abs(np.diag(r_it[:p, :])) > 1e-10))
            q_it = q_full[:, :p]
            effects = q_full.T @ E_w.T
            s2 = np.mean(effects[rank_it:, :] ** 2, axis=0)
            # R's lm.wfit$residuals returns residuals on the original (unweighted)
            # scale: y - X beta.
            beta = linalg.solve_triangular(
                r_it[:rank_it, :rank_it], effects[:rank_it, :], lower=False
            )
            residuals = (E.T - design @ beta).T
            qr_factors = (q_it, rank_it)

        q, rank = qr_factors

        # Q (narrays x p) = first p cols of full orthogonal matrix
        # qr.qy(qr, diag(n,p)) returns the first p cols of the full Q.
        # With economic QR we already have q of shape (narrays, p).
        Q = q  # shape (narrays, p)

        # Build Q2: p2 columns of pairwise Q products.
        # Columns grouped by offset k=0..p-1: cols (j0+1):(j0+p-k) = Q[,1:(p-k)] * Q[,(k+1):p]
        Q2 = np.zeros((n_samples, p2))
        j0 = 0
        for k in range(p):
            span = p - k
            Q2[:, j0 : j0 + span] = Q[:, :span] * Q[:, k : k + span]
            j0 += span
        if p > 1:
            Q2[:, p:p2] *= np.sqrt(2.0)

        # Hat values: row sums of diagonal product block (first p cols)
        h = np.sum(Q2[:, :p], axis=1)

        # Fisher info (including intercept for log-variance model)
        info = Z.T @ ((1.0 - 2.0 * h)[:, np.newaxis] * Z) + (Q2.T @ Z).T @ (Q2.T @ Z)

        # Schur complement removing intercept row/col
        info2 = info[1:, 1:] - np.outer(info[1:, 0], info[0, 1:]) / info[0, 0]

        # Score: colMeans over genes of w * residuals^2 / s2, then - (1 - h)
        # residuals is shape (ngenes, narrays). w * resid^2 / s2 gives per-gene
        # per-array contribution; colMeans = mean across genes, shape (narrays,).
        score_per = (w[np.newaxis, :] * residuals**2) / s2[:, np.newaxis]
        z = np.mean(score_per, axis=0) - (1.0 - h)

        # Add prior support
        info2 = n_genes * info2 + prior_n * (Z2.T @ Z2)
        z = n_genes * z + prior_n * (w - 1.0)

        dl = Z2.T @ z
        try:
            gamstep = linalg.solve(info2, dl)
        except linalg.LinAlgError:
            break

        convcrit = float(dl @ gamstep) / n_gam / (n_genes + prior_n)
        if not np.isfinite(convcrit) or convcrit >= convcrit_last:
            break
        convcrit_last = convcrit

        gam = gam + gamstep
        w = np.exp(Z2 @ (-gam))

        if trace:
            print(f"{iteration} {convcrit:.6g} {w.min():.6g} {w.max():.6g}")

        if convcrit < tol:
            break

    return w


def _array_weights_pr_wts_reml(
    E: np.ndarray,
    design: np.ndarray,
    weights: np.ndarray,
    var_design: np.ndarray,
    prior_n: float = 10,
    maxiter: int = 50,
    tol: float = 1e-5,
    trace: bool = False,
) -> np.ndarray:
    """
    Estimate array weights by REML allowing for prior observation weights.

    Faithful per-gene Fisher-scoring port of limma's
    .arrayWeightsPrWtsREML (limma/R/arrayWeightsPrWtsREML.R).

    Algorithmic structure mirrors :func:`_array_weights_reml` but accumulates
    Fisher information and the score *inside* the gene loop using
    `lm.wfit(design, y[g,], w * weights[g,])` semantics. Information and
    score are then averaged by `(ngenes + prior.n)` per
    `arrayWeightsPrWtsREML.R:65-66`.

    Parameters
    ----------
    E : ndarray
        Expression matrix, shape (n_genes, n_samples). Assumes no NA / Inf
        (caller filters rows).
    design : ndarray
        Design matrix, shape (n_samples, p).
    weights : ndarray
        Prior observation weights, shape (n_genes, n_samples). Non-NaN.
    var_design : ndarray
        Variance design matrix Z2 (intercept already excluded, columns sum
        to zero).
    prior_n : float
        Prior sample size pulling weights toward 1.
    maxiter : int
        Maximum Fisher-scoring iterations.
    tol : float
        Convergence tolerance on the rescaled score-step inner product.
    trace : bool
        If True, print iteration progress.
    """
    n_genes, n_samples = E.shape
    Z2 = var_design
    n_gam = Z2.shape[1]
    Z = np.column_stack([np.ones(n_samples), Z2])
    p = design.shape[1]
    p2 = (p * (p + 1)) // 2

    gam = np.zeros(n_gam)
    w = np.ones(n_samples)

    if trace:
        print("iter convcrit range(w)")

    for iteration in range(1, maxiter + 1):
        info2 = prior_n * (Z2.T @ Z2)
        z = prior_n * (w - 1.0)

        for g in range(n_genes):
            wg = w * weights[g, :]
            if np.any(wg <= 0):
                continue

            sw = np.sqrt(wg)
            X_w = design * sw[:, np.newaxis]
            y_w = E[g, :] * sw

            try:
                q_full, r_it = linalg.qr(X_w, mode="full")
            except linalg.LinAlgError:
                continue
            rank = int(np.sum(np.abs(np.diag(r_it[:p, :])) > 1e-10))
            if rank < p:
                # rank-deficient gene; skip rather than risk incompatible Q2
                continue

            Q = q_full[:, :p]
            effects = q_full.T @ y_w
            s2 = float(np.mean(effects[rank:] ** 2))

            # R: residuals are y - X beta on the original (unweighted) scale
            beta = linalg.solve_triangular(r_it[:rank, :rank], effects[:rank], lower=False)
            resid = E[g, :] - design @ beta

            Q2 = np.zeros((n_samples, p2))
            j0 = 0
            for k in range(p):
                span = p - k
                Q2[:, j0 : j0 + span] = Q[:, :span] * Q[:, k : k + span]
                j0 += span
            if p > 1:
                Q2[:, p:p2] *= np.sqrt(2.0)

            h = np.sum(Q2[:, :p], axis=1)
            info = Z.T @ ((1.0 - 2.0 * h)[:, np.newaxis] * Z) + (Q2.T @ Z).T @ (Q2.T @ Z)
            info2 = info2 + info[1:, 1:] - np.outer(info[1:, 0], info[0, 1:]) / info[0, 0]

            if s2 > 1e-15:
                z = z + wg * resid**2 / s2 - (1.0 - h)

        info2_avg = info2 / (n_genes + prior_n)
        z_avg = z / (n_genes + prior_n)

        dl = Z2.T @ z_avg
        try:
            gamstep = linalg.solve(info2_avg, dl)
        except linalg.LinAlgError:
            break

        gam = gam + gamstep
        w = np.exp(Z2 @ (-gam))

        convcrit = float(dl @ gamstep) / (n_genes + prior_n) / n_gam
        if trace:
            print(f"{iteration} {convcrit:.6g} {w.min():.6g} {w.max():.6g}")

        if not np.isfinite(convcrit):
            break
        if convcrit < tol:
            break

    return w


def array_weights(
    object,
    design: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    var_design: np.ndarray | None = None,
    var_group: np.ndarray | None = None,
    prior_n: float = 10,
    method: str = "auto",
    maxiter: int = 50,
    tol: float = 1e-5,
    trace: bool = False,
    *,
    layer: str | None = None,
    weights_layer: str | None = None,
) -> np.ndarray:
    """
    Estimate relative quality weights for each array/sample.

    Estimates the relative reliability of each sample in a gene expression
    experiment. Samples with higher variability get lower weights.

    Parameters
    ----------
    object : dict
        Dict with 'E' (expression matrix) and optionally 'weights'.
        Typically the output from voom().
    design : ndarray, optional
        Design matrix for the linear model. If None, taken from object
        or defaults to intercept-only.
    weights : ndarray, optional
        Prior observation weights. If None, taken from object.
    var_design : ndarray, optional
        Design matrix for the variance model. Columns should sum to zero.
    var_group : ndarray, optional
        Factor defining variance groups. Takes precedence over var_design.
    prior_n : float, default 10
        Prior sample size for regularization. Higher values give more stable
        but less responsive estimates.
    method : str, default "auto"
        Estimation method:
        - "auto": Choose automatically (genebygene if weights or NAs present)
        - "genebygene": Gene-by-gene update algorithm
        - "reml": REML estimation (faster when no weights or NAs)
    maxiter : int, default 50
        Maximum iterations for REML method.
    tol : float, default 1e-5
        Convergence tolerance for REML method.
    trace : bool, default False
        If True, print iteration progress (array weight range) to stdout.

    Returns
    -------
    ndarray
        Quality weights for each sample, shape (n_samples,).
        Higher weights indicate more reliable samples.

    Notes
    -----
    Array weights are useful when some samples have higher technical
    variability than others. By downweighting noisy samples, the analysis
    gains power while remaining valid.

    The weights are relative (their geometric mean is approximately 1)
    and can be incorporated into downstream analysis via lm_fit().
    """
    # Polymorphic input: ndarray / dict / EList / AnnData.
    eawp = get_eawp(object, layer=layer, weights_layer=weights_layer)
    E = np.asarray(eawp["exprs"], dtype=np.float64)
    n_genes, n_samples = E.shape

    # Pick up design/weights from the input object if not passed explicitly.
    if design is None and eawp.get("design") is not None:
        design = np.asarray(eawp["design"], dtype=np.float64)
    if weights is None and eawp.get("weights") is not None:
        weights = np.asarray(eawp["weights"], dtype=np.float64)

    # Initial weights
    w = np.ones(n_samples)

    # Require at least 2 genes
    if n_genes < 2:
        return w

    # Default design
    if design is None:
        design = np.ones((n_samples, 1))
    design = np.asarray(design, dtype=np.float64)

    # Reduce rank if not full rank
    q, r, pivot = linalg.qr(design, pivoting=True)
    p = np.sum(np.abs(np.diag(r)) > 1e-10)
    if p < design.shape[1]:
        design = design[:, pivot[:p]]

    # Require at least 2 residual df
    if n_samples - p < 2:
        return w

    # Check weights (already extracted from input object above if not passed)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != E.shape:
            raise ValueError("weights must have same shape as expression matrix")
        if np.any(np.isnan(weights)):
            raise ValueError("NA weights not allowed")
        if np.any(weights < 0):
            raise ValueError("Negative weights not allowed")
        if np.any(np.isinf(weights)):
            raise ValueError("Infinite weights not allowed")
        # Handle zero weights by setting corresponding E to NA
        if np.any(weights == 0):
            E = E.copy()
            E[weights == 0] = np.nan
            weights = weights.copy()
            weights[weights == 0] = 1

    # Handle var_group (takes precedence over var_design)
    if var_group is not None:
        var_group = np.asarray(var_group)
        if len(var_group) != n_samples:
            raise ValueError("var_group has wrong length")

        # Get unique levels
        unique_levels = np.unique(var_group)
        n_levels = len(unique_levels)
        if n_levels < 2:
            raise ValueError("Need at least two variance groups")

        # Create contrast-coded design matrix (sum to zero)
        # Similar to R's contr.sum
        var_design = np.zeros((n_samples, n_levels - 1))
        for i, level in enumerate(unique_levels[:-1]):
            var_design[var_group == level, i] = 1
        var_design[var_group == unique_levels[-1], :] = -1

    # Setup variance design matrix
    if var_design is None:
        var_design = _contr_sum(n_samples)
    else:
        # Center columns (make them sum to zero)
        var_design = var_design - np.mean(var_design, axis=0)
        # Remove rank-deficient columns
        q_v, r_v, pivot_v = linalg.qr(var_design, pivoting=True)
        rank_v = np.sum(np.abs(np.diag(r_v)) > 1e-10)
        var_design = var_design[:, pivot_v[:rank_v]]

    # Detect NA values
    has_na = np.any(~np.isfinite(E))
    if has_na:
        E = E.copy()
        E[~np.isfinite(E)] = np.nan

    # Choose method
    method = method.lower()
    if method == "auto":
        if has_na or weights is not None:
            method = "genebygene"
        else:
            method = "reml"

    if method == "genebygene":
        return _array_weights_gene_by_gene(E, design, weights, var_design, prior_n, trace=trace)
    elif method == "reml":
        if has_na:
            # Remove rows with any NA
            na_rows = np.any(np.isnan(E), axis=1)
            E = E[~na_rows, :]
            if weights is not None:
                weights = weights[~na_rows, :]
            if E.shape[0] < 2:
                return w

        if weights is None:
            return _array_weights_reml(E, design, var_design, prior_n, maxiter, tol, trace=trace)
        else:
            return _array_weights_pr_wts_reml(
                E,
                design,
                weights,
                var_design,
                prior_n,
                maxiter,
                tol,
                trace=trace,
            )
    else:
        raise ValueError(f"Unknown method: {method}")


def array_weights_quick(
    y,
    fit: dict,
    *,
    layer: str | None = None,
) -> np.ndarray:
    """
    Compute approximate array quality weights from a linear model fit.

    Faithful port of R limma's :func:`arrayWeightsQuick`. Computes
    ``1 / colMeans(res*res / (sigma^2 * (1 - h)))`` where ``h`` is the
    leverage diagonal of the design's hat matrix and ``res`` is the
    residual matrix from the fit.

    Parameters
    ----------
    y : array_like
        Expression matrix, shape (n_genes, n_samples).
    fit : dict
        Linear model fit (from :func:`lm_fit`) carrying ``coefficients``,
        ``design``, and ``sigma``. If ``fit`` carries observation weights
        a warning is emitted (matches R: spot quality weights are not
        taken into account).

    Returns
    -------
    ndarray
        Approximate array quality weights, shape (n_samples,).
    """
    y = np.asarray(get_eawp(y, layer=layer)["exprs"], dtype=np.float64)

    if fit.get("weights") is not None:
        import warnings

        warnings.warn(
            "spot quality weights found but not taken into account",
            UserWarning,
        )

    coefficients = np.asarray(fit["coefficients"], dtype=np.float64)
    design = np.asarray(fit["design"], dtype=np.float64)
    sigma = np.asarray(fit["sigma"], dtype=np.float64)

    res = y - coefficients @ design.T

    # Hat-matrix diagonal of the design itself (no intercept augmentation).
    q, _r, _pivot = linalg.qr(design, pivoting=True, mode="economic")
    rank = int(np.sum(np.abs(np.diag(_r)) > 1e-10))
    q = q[:, :rank]
    h = np.sum(q**2, axis=1)

    mures2 = (sigma[:, np.newaxis] ** 2) * (1.0 - h)[np.newaxis, :]
    return 1.0 / np.nanmean(res * res / mures2, axis=0)


def modify_weights(
    weights,
    status,
    values,
    multipliers,
) -> np.ndarray:
    """
    Multiply rows of a weights matrix by status-specific scalars.

    Port of R limma's ``modifyWeights``. For each entry in ``values``,
    rows whose ``status`` matches that entry are multiplied (in place
    on a copy) by the corresponding multiplier.

    Parameters
    ----------
    weights : array_like
        Weights matrix of shape (n_features, n_arrays). 1-D inputs are
        promoted to a single-column matrix to mirror R's
        ``as.matrix(weights)``.
    status : array_like of str
        Per-row status labels. Length must equal ``nrow(weights)``.
    values : sequence of str
        Status values to act on.
    multipliers : float or sequence of float
        Multiplier for each value. A single multiplier is recycled to
        the length of ``values``.

    Returns
    -------
    ndarray
        Updated weights matrix, same shape as the input (1-D inputs
        come back as a column matrix to match R).
    """
    status = np.asarray([str(s) for s in status])
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)
    weights = weights.copy()
    values = [str(v) for v in values]
    multipliers = np.atleast_1d(np.asarray(multipliers, dtype=np.float64))

    if status.size != weights.shape[0]:
        raise ValueError("nrows of weights must equal length of status")
    if multipliers.size == 1:
        multipliers = np.repeat(multipliers, len(values))
    if len(values) != multipliers.size:
        raise ValueError("no. values doesn't match no. multipliers")

    for value, mult in zip(values, multipliers):
        mask = status == value
        if np.any(mask):
            weights[mask, :] = mult * weights[mask, :]
    return weights

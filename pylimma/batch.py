# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   removeBatchEffect.R                  Copyright (C) 2008-2025 Gordon Smyth,
#                                                                Carolyn de Graaf
#   wsva.R                               Copyright (C) 2015-2017 Yifang Hu,
#                                                                Gordon Smyth
# Python port: Copyright (C) 2026 John Mulvey
"""
Batch-effect removal and surrogate-variable analysis for pylimma.

Faithful ports:

- ``remove_batch_effect`` (``limma/R/removeBatchEffect.R``).
- ``wsva`` (``limma/R/wsva.R``). Weighted surrogate variable
  analysis; has an optional screeplot branch, so it lands in this
  module alongside ``remove_batch_effect`` rather than in
  ``plotting.py``.

Accepts matrix / dict / EList / AnnData via the ``get_eawp`` /
``put_eawp`` dispatchers. RGList / MAList / EListRaw are out of scope
(see ``memory/policy_data_class_wrappers.md``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .classes import get_eawp, put_eawp
from .lmfit import lm_fit


def _factor_levels(factor) -> tuple[np.ndarray, np.ndarray]:
    """Determine levels and string-coded factor values following R's
    ``as.factor()`` dispatch.

    Rules (mirrors R's `as.factor`):
    - ``pd.Categorical`` / ``pd.Series`` with categorical dtype: preserve
      ``categories`` order (matches R's factor-input behaviour).
    - Numeric (int/float) array: levels = numeric sort of unique values,
      rendered back to strings (matches R's `as.factor(numeric_vector)`).
    - Boolean: levels = ``["FALSE", "TRUE"]`` (matches R).
    - Character / object: levels = alphabetical sort of unique string values
      (matches R's `as.factor(character_vector)`).

    Returns ``(levels, str_values)`` both as 1-D numpy arrays of strings.
    """
    # Categorical (including pd.Series of categorical dtype)
    if isinstance(factor, pd.Categorical):
        cat = factor
    elif isinstance(factor, pd.Series) and hasattr(factor, "cat"):
        cat = factor.values
    else:
        cat = None
    if cat is not None and isinstance(cat, pd.Categorical):
        levels = np.array([str(c) for c in cat.categories])
        str_values = np.array([str(v) for v in cat])
        return levels, str_values

    arr = np.asarray(factor)

    # Boolean
    if arr.dtype == bool:
        levels = np.array(["FALSE", "TRUE"])
        str_values = np.where(arr, "TRUE", "FALSE")
        return levels, str_values

    # Numeric (integer or float)
    if np.issubdtype(arr.dtype, np.number):
        unique_sorted = np.sort(np.unique(arr))

        # R's `as.character(numeric)` formatting: integers render without
        # decimals, floats use R's default (effectively Python's default too)
        def _render(x):
            if float(x).is_integer():
                return str(int(x))
            return str(x)

        levels = np.array([_render(v) for v in unique_sorted])
        str_values = np.array([_render(v) for v in arr])
        return levels, str_values

    # Character / object: alphabetical
    str_values = np.array([str(v) for v in arr])
    levels = np.array(sorted(set(str_values)))
    return levels, str_values


def _sum_to_zero_design(factor) -> np.ndarray:
    """Build R-equivalent ``model.matrix(~factor)[,-1]`` with
    ``contrasts = contr.sum`` applied to the factor.

    For a factor with K levels (in R's `as.factor()` order - numeric for
    int/float input, alphabetical for character, preserved for
    ``pd.Categorical``), returns an ``n x (K-1)`` matrix: the first K-1
    levels get a 1 in their column and the last level gets -1 across all
    columns. Intercept is dropped.
    """
    levels, str_values = _factor_levels(factor)
    n = len(str_values)
    K = len(levels)
    if K < 2:
        return np.zeros((n, 0), dtype=np.float64)
    X = np.zeros((n, K - 1), dtype=np.float64)
    for j, lev in enumerate(levels[:-1]):
        X[str_values == lev, j] = 1.0
    X[str_values == levels[-1], :] = -1.0
    return X


def remove_batch_effect(
    x,
    batch: np.ndarray | list | None = None,
    batch2: np.ndarray | list | None = None,
    covariates: np.ndarray | None = None,
    design: np.ndarray | None = None,
    group: np.ndarray | list | None = None,
    *,
    out_layer: str = "batch_removed",
    uns_key: str = "remove_batch_effect",
    layer: str | None = None,
    **lmfit_kwargs,
):
    """
    Remove batch effects from a matrix of expression values.

    Faithful port of R limma's ``removeBatchEffect``
    (``limma/R/removeBatchEffect.R``). Fits a linear model
    against a combined design of experimental conditions and batch
    covariates, then subtracts the estimated batch-coefficient
    contribution from the expression matrix.

    Parameters
    ----------
    x : ndarray, EList, AnnData, or dict
        Expression matrix (genes x samples) or wrapper.
    batch : array-like, optional
        Factor of batch labels. Coded with sum-to-zero contrasts before
        entering the design.
    batch2 : array-like, optional
        Second batch factor, treated the same way as ``batch``.
    covariates : ndarray, optional
        Quantitative covariates (samples x p). Mean-centred before entry.
    design : ndarray, optional
        Design matrix for the experimental conditions to be preserved.
        If omitted and ``group`` is also omitted, a one-group design is
        assumed and a warning is emitted.
    group : array-like, optional
        If given and ``design`` is omitted, sets ``design =
        one_hot(group)``.
    **lmfit_kwargs
        Forwarded to ``lm_fit`` (e.g. ``weights``, ``method``).

    Returns
    -------
    Same class as input (matrix -> ndarray, EList -> EList, AnnData -> None).
    """
    original_input = x
    eawp = get_eawp(x, layer=layer)
    X = np.asarray(eawp["exprs"], dtype=np.float64)

    if batch is None and batch2 is None and covariates is None:
        return put_eawp(
            {"E": X},
            original_input,
            out_layer=out_layer,
            weights_layer=None,
            uns_key=uns_key,
            single_matrix=True,
        )

    parts = []
    if batch is not None:
        parts.append(_sum_to_zero_design(batch))
    if batch2 is not None:
        parts.append(_sum_to_zero_design(batch2))
    if covariates is not None:
        cov = np.asarray(covariates, dtype=np.float64)
        if cov.ndim == 1:
            cov = cov.reshape(-1, 1)
        cov = cov - cov.mean(axis=0, keepdims=True)
        parts.append(cov)
    X_batch = np.concatenate(parts, axis=1) if parts else np.zeros((X.shape[1], 0))

    if group is not None and design is None:
        grp = np.asarray([str(v) for v in group])
        levels = np.array(sorted(set(grp)))
        design = np.zeros((len(grp), len(levels)), dtype=np.float64)
        for j, lev in enumerate(levels):
            design[grp == lev, j] = 1.0

    if design is None:
        warnings.warn(
            "design matrix of interest not specified. Assuming a one-group experiment.",
            UserWarning,
        )
        design = np.ones((X.shape[1], 1), dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)

    full_design = np.concatenate([design, X_batch], axis=1)
    fit = lm_fit(X, design=full_design, **lmfit_kwargs)
    coef = np.asarray(fit["coefficients"], dtype=np.float64).copy()
    # R's lmFit QR pivots out the trailing redundant column, so any NA lands
    # in X_batch's slice and `beta[is.na(beta)] <- 0` correctly zeros the
    # wrong-subtraction. pylimma.lm_fit's QR may pivot a preserved-design
    # column instead, leaving a finite but algebraically-arbitrary value in
    # X_batch. Mirror R's intent by zeroing the ENTIRE row across both slices
    # whenever a NaN appears anywhere in that row's coefficients: if part of
    # the linear combination is unidentified, none of the decomposition's
    # batch-attribution can be trusted for that probe.
    nan_row = np.any(np.isnan(coef), axis=1)
    coef[nan_row, :] = 0.0
    beta = coef[:, design.shape[1] :]
    corrected = X - beta @ X_batch.T

    return put_eawp(
        {"E": corrected},
        original_input,
        out_layer=out_layer,
        weights_layer=None,
        uns_key=uns_key,
        single_matrix=True,
    )


def _lm_effects_residual(
    y: np.ndarray,
    design: np.ndarray,
    array_weights: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    block: np.ndarray | None = None,
    correlation: float | None = None,
) -> np.ndarray:
    """Residual-space effects matrix for wsva.

    Equivalent of R limma's ``.lmEffects(y, design, ...)[, -1]``
    (lmEffects.R): the residual block of the effects matrix with the
    contrast column dropped. Shape ``(ngenes, n - p)``.

    Supports the same side-channel arguments R's .lmEffects forwards
    from wsva's ``...``: ``array_weights``, ``weights`` (per-observation
    matrix), ``block`` + ``correlation``. R's ``gene.weights`` and the
    ``weights``-as-array-weights alias are not supported here because
    wsva's documented use case is SV estimation on residual space only.

    Parameters
    ----------
    y : ndarray
        Expression matrix (n_genes, n_samples).
    design : ndarray
        Design matrix (n_samples, p).
    array_weights : ndarray, optional
        Per-sample weights (length n_samples). Pre-scales y and design
        by sqrt(array_weights), mirroring lmEffects.R:94-99.
    weights : ndarray, optional
        Per-observation weights (n_genes, n_samples). Triggers the
        per-gene QR loop in lmEffects.R:131-150.
    block : ndarray, optional
        Block factor (length n_samples) for GLS.
    correlation : float, optional
        Within-block correlation; required when ``block`` is given.
    """
    from scipy.linalg import qr, solve_triangular

    p = design.shape[1]
    n = design.shape[0]
    if n <= p:
        raise ValueError("No residual degrees of freedom")

    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(design, dtype=np.float64)

    # Array weights: divide out via sqrt transform (lmEffects.R:94-99).
    if array_weights is not None:
        aw = np.asarray(array_weights, dtype=np.float64)
        if aw.size != n:
            raise ValueError("Length of array_weights doesn't match number of arrays")
        if np.any(aw <= 0) or np.any(np.isnan(aw)):
            raise ValueError("array_weights must be positive")
        ws = np.sqrt(aw)
        X = X * ws[:, None]
        y = y * ws[None, :]

    # Block / correlation: GLS via Cholesky (lmEffects.R:102-118).
    R_chol = None
    if block is not None:
        if correlation is None:
            raise ValueError("correlation must be set when block is given")
        block_arr = np.asarray(block).ravel()
        if block_arr.size != n:
            raise ValueError("Length of block does not match number of arrays")
        ub, inv = np.unique(block_arr, return_inverse=True)
        Z = (inv[:, None] == np.arange(len(ub))).astype(np.float64)
        cormatrix = Z @ (float(correlation) * Z.T)
        np.fill_diagonal(cormatrix, 1.0)
        R_chol = np.linalg.cholesky(cormatrix).T  # upper triangular R
        if weights is None:
            # Apply transform y <- solve(R^T, y^T)^T and X <- solve(R^T, X).
            y = solve_triangular(R_chol.T, y.T, lower=True).T
            X = solve_triangular(R_chol.T, X, lower=True)

    # Per-observation weights: per-gene QR loop (lmEffects.R:131-150).
    if weights is not None:
        w_mat = np.asarray(weights, dtype=np.float64)
        if w_mat.shape != (y.shape[0], n):
            raise ValueError("weights must have same dimensions as y")
        if np.any(w_mat <= 0) or np.any(np.isnan(w_mat)):
            raise ValueError("weights must be positive")
        effects = np.zeros((y.shape[0], n))
        for g in range(y.shape[0]):
            ws_g = np.sqrt(w_mat[g])
            wX = X * ws_g[:, None]
            wy = y[g] * ws_g
            if R_chol is not None:
                wy = solve_triangular(R_chol.T, wy, lower=True)
                wX = solve_triangular(R_chol.T, wX, lower=True)
            Q_g, _ = qr(wX, mode="full")
            effects[g] = Q_g.T @ wy
        return effects[:, p:]  # drop contrast columns, keep residual block

    # Common path: single QR of (possibly transformed) X.
    Q, _ = qr(X, mode="full")
    residual = Q[:, p:].T @ y.T
    return residual.T  # (n_genes, n - p)


def wsva(
    y,
    design: np.ndarray,
    n_sv: int = 1,
    weight_by_sd: bool = False,
    plot: bool = False,
    *,
    array_weights: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    block: np.ndarray | None = None,
    correlation: float | None = None,
    **kwargs,
) -> np.ndarray:
    """Weighted surrogate variable analysis.

    Port of R limma's ``wsva`` (Yifang Hu and Gordon Smyth, 2015-2017).
    Returns an ``n_arrays x n_sv`` matrix of surrogate variables.

    When ``weight_by_sd=True``, the algorithm is iterative and each
    iteration weights rows by their residual SD. When ``plot=True``, a
    screeplot of the singular-value spectrum is produced via
    matplotlib (lazy import).

    ``array_weights``, ``weights``, ``block``, and ``correlation`` are
    threaded through to ``.lmEffects`` as R's wsva does via ``...``
    (wsva.R:1, lmEffects.R:1). ``weights`` aliased as array-weights
    (length ``n_arrays``) is promoted to ``array_weights`` to match
    R's lmEffects.R:52-56.
    """
    # R's .lmEffects (lmEffects.R:52-56) promotes a length-n vector
    # passed as `weights` to `array.weights` when array_weights is None.
    # Mirror that aliasing before dispatch.
    eawp = get_eawp(y)
    y_mat = np.asarray(eawp["exprs"], dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    if design.ndim == 1:
        design = design.reshape(-1, 1)

    narrays = y_mat.shape[1]
    p = design.shape[1]
    d = narrays - p

    if array_weights is None and weights is not None:
        w_arr = np.asarray(weights)
        if w_arr.ndim == 1 and w_arr.size == narrays:
            array_weights = w_arr
            weights = None

    if kwargs:
        warnings.warn(
            f"Extra arguments disregarded: {sorted(kwargs.keys())}",
            UserWarning,
        )

    n_sv = max(int(n_sv), 1)
    n_sv = min(n_sv, d)
    if n_sv <= 0:
        raise ValueError("No residual df")

    # Shared kwargs for every call to _lm_effects_residual in this function.
    eff_kwargs = dict(
        array_weights=array_weights,
        weights=weights,
        block=block,
        correlation=correlation,
    )

    if weight_by_sd:
        if plot:
            warnings.warn("Plot not available with weight_by_sd=True", UserWarning)
        current_design = design
        for _ in range(n_sv):
            Effects = _lm_effects_residual(y_mat, current_design, **eff_kwargs)
            s = np.sqrt(np.mean(Effects**2, axis=1))
            Effects_w = s[:, None] * Effects
            U, _, _ = np.linalg.svd(Effects_w, full_matrices=False)
            u = U[:, 0] * s
            sv = (u[:, None] * y_mat).sum(axis=0)
            current_design = np.concatenate([current_design, sv.reshape(-1, 1)], axis=1)
        SV = current_design[:, p:].T  # (n_sv, narrays)
    else:
        Effects = _lm_effects_residual(y_mat, design, **eff_kwargs)
        U, s, _ = np.linalg.svd(Effects, full_matrices=False)
        U = U[:, :n_sv]
        SV = U.T @ y_mat  # (n_sv, narrays)

        if plot:
            from .plotting import _require_matplotlib

            plt = _require_matplotlib()
            lam = s**2
            lam = lam / lam.sum()
            _, ax = plt.subplots()
            ax.plot(np.arange(1, len(lam) + 1), lam, "o")
            ax.set_xlabel("Surrogate variable number")
            ax.set_ylabel("Proportion variance explained")

    A = (SV**2).mean(axis=1)
    SV = (SV / np.sqrt(A)[:, None]).T  # (narrays, n_sv)
    return SV

# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   geneset-ids2indices.R           Copyright (C) 2009-2015 Gordon Smyth,
#                                                            Yifang Hu
#   geneset-roast.R                 Copyright (C) 2008-2020 Gordon Smyth,
#                                                            Di Wu
#   geneset-fry.R                   Copyright (C) 2015-2020 Gordon Smyth,
#                                                            Goknur Giner
#   geneset-camera.R                Copyright (C) 2007-2025 Gordon Smyth,
#                                                            Di Wu
#   geneset-cameraPR.R              Copyright (C) 2017-2025 Gordon Smyth
#   geneset-romer.R                 Copyright (C) 2009-2015 Gordon Smyth,
#                                                            Yifang Hu
#   geneset-wilcox.R                Copyright (C) 2004-2012 Gordon Smyth
#   rankSumTestWithCorrelation.R    Copyright (C) 2007-2012 Gordon Smyth,
#                                                            Di Wu
#   lmEffects.R                     Copyright (C) 2016-2020 Gordon Smyth
# Python port: Copyright (C) 2026 John Mulvey
"""
Gene-set testing routines for pylimma.

Faithful ports of R limma's competitive and self-contained gene-set tests:
``ids2indices``, ``roast``/``mroast`` (rotation test), ``fry``
(closed-form roast limit), ``camera`` (correlation-aware competitive
test), ``camera_pr`` (preranked variant), ``romer`` (rotation mean-rank),
and the Wilcoxon-style ``gene_set_test`` / ``rank_sum_test_with_correlation``.

RNG note
--------
``roast``, ``mroast``, ``romer`` and the simulation branch of
``gene_set_test`` use rotations drawn from a random normal distribution.
R's Mersenne-Twister stream is not reproduced byte-for-byte on the Python
side; these functions take an ``rng`` argument that is forwarded to
``numpy.random.default_rng``. Deterministic summaries (``ngenes``,
observed set statistics, active proportions) match R to machine precision;
Monte-Carlo p-values from independently-seeded streams agree to within
the Monte-Carlo sampling error (see ``known_diff_roast_rng.md``).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg, stats

from .classes import get_eawp
from .lmfit import non_estimable
from .squeeze_var import fit_f_dist, squeeze_var
from .utils import (
    _zscore_t_bailey,
    p_adjust,
    zscore_t,
)

# ---------------------------------------------------------------------------
# ids2indices
# ---------------------------------------------------------------------------


def ids2indices(
    gene_sets,
    identifiers,
    remove_empty: bool = True,
) -> dict:
    """
    Map named gene sets of identifier strings to zero-based integer indices.

    Port of R limma's ``ids2indices``. Indices are returned in Python's
    zero-based convention (the R version returns 1-based indices); this
    is what the downstream Python gene-set functions expect.

    Parameters
    ----------
    gene_sets : dict or list
        Either a dict mapping set names to iterables of identifiers, or a
        single iterable (wrapped as ``{"Set1": gene_sets}``, matching R's
        ``if(!is.list(gene.sets))`` branch).
    identifiers : array_like of str
        Identifier vector; the returned indices are positions in this
        vector.
    remove_empty : bool, default True
        Drop sets that contain no matches.

    Returns
    -------
    dict[str, np.ndarray]
        Dict mapping each set name to an ``int64`` array of zero-based
        indices into ``identifiers``.
    """
    if not isinstance(gene_sets, dict):
        gene_sets = {"Set1": gene_sets}

    ids = np.asarray(identifiers).astype(str)
    # Build position lookup in insertion order, keeping only the first
    # occurrence of each identifier (matches R's which(identifiers %in% x)
    # semantics when identifiers contain duplicates).
    lookup: dict[str, int] = {}
    for pos, name in enumerate(ids):
        if name not in lookup:
            lookup[name] = pos

    out: dict[str, np.ndarray] = {}
    for name, members in gene_sets.items():
        members_arr = np.asarray(members).astype(str)
        hits = np.fromiter(
            (lookup[m] for m in members_arr if m in lookup),
            dtype=np.int64,
        )
        hits.sort()
        out[name] = hits

    if remove_empty:
        out = {k: v for k, v in out.items() if v.size > 0}

    return out


# ---------------------------------------------------------------------------
# _lm_effects (port of R limma's .lmEffects)
# ---------------------------------------------------------------------------


def _lm_effects(
    y,
    design: np.ndarray | None = None,
    contrast: Any = None,
    array_weights: np.ndarray | None = None,
    gene_weights: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    block=None,
    correlation: float | None = None,
) -> np.ndarray:
    """
    Compute matrix of effects from gene-wise linear models.

    Port of R limma's ``.lmEffects``. Returns an (ngenes, df_residual+1)
    matrix whose first column is the primary effect (for the chosen
    contrast) and whose remaining columns are residual effects (in
    arbitrary but fixed order).
    """
    ea = get_eawp(y)
    expr = ea["exprs"]
    ngenes, n = expr.shape

    if np.any(~np.isfinite(expr)):
        raise ValueError("All y values must be finite and non-NA")

    if design is None:
        design = ea.get("design")
    if design is None:
        raise ValueError("design matrix not specified")
    design = np.asarray(design, dtype=np.float64)
    if design.ndim == 1:
        design = design.reshape(-1, 1)
    if design.shape[0] != n:
        raise ValueError("row dimension of design matrix must match column dimension of data")
    p = design.shape[1]
    if n <= p:
        raise ValueError("No residual degrees of freedom")

    # Default contrast: last column of design.
    if contrast is None:
        contrast = p - 1

    # Reform design so that the contrast is the last coefficient.
    contrast_arr = np.atleast_1d(np.asarray(contrast))
    if (
        contrast_arr.ndim == 1
        and contrast_arr.size == 1
        and np.issubdtype(contrast_arr.dtype, np.integer)
    ):
        k = int(contrast_arr.item())
        if k < 0 or k >= p:
            raise ValueError(f"contrast index {k} out of range for design with {p} columns")
        if np.all(design[:, k] == 0):
            raise ValueError("contrast all zero")
        if k < p - 1:
            keep = [j for j in range(p) if j != k]
            X = np.column_stack([design[:, keep], design[:, k : k + 1]])
        else:
            X = design.copy()
    else:
        # contrast is a vector of length p: rotate design so the contrast
        # direction is the last column. Implements contrastAsCoef(first=FALSE).
        contrast_vec = np.asarray(contrast, dtype=np.float64)
        if contrast_vec.size != p:
            raise ValueError("length of contrast must match column dimension of design")
        if np.all(contrast_vec == 0):
            raise ValueError("contrast all zero")
        Q_c, R_c = linalg.qr(contrast_vec.reshape(-1, 1), mode="full")
        # R: design <- t(qr.qty(QR, t(design))) = design %*% Q.
        rotated = design @ Q_c
        if np.sign(R_c[0, 0]) < 0:
            rotated[:, 0] = -rotated[:, 0]
        # Move the (now-first) contrast coefficient to the last column.
        X = np.column_stack([rotated[:, 1:], rotated[:, 0:1]])

    # Allow array.weights to be alternatively passed via 'weights'.
    if array_weights is None and weights is not None and np.asarray(weights).size == n:
        array_weights = weights
        weights = None

    if array_weights is not None:
        aw = np.asarray(array_weights, dtype=np.float64)
        if aw.size != n:
            raise ValueError("Length of array.weights doesn't match number of arrays")
        if np.any(aw <= 0) or np.any(np.isnan(aw)):
            raise ValueError("array.weights must be positive")

    # Allow gene.weights to be alternatively passed via 'weights'.
    if gene_weights is None and weights is not None and np.asarray(weights).size == ngenes:
        gene_weights = weights
        weights = None

    if gene_weights is not None:
        gw = np.asarray(gene_weights, dtype=np.float64)
        if gw.size != ngenes:
            raise ValueError("Length of gene.weights doesn't match number of genes")
        if np.any(gw <= 0) or np.any(np.isnan(gw)):
            raise ValueError("gene.weights must be positive")

    if weights is None:
        weights = ea.get("weights")
    if weights is not None:
        w_mat = np.asarray(weights, dtype=np.float64)
        if w_mat.shape != (ngenes, n):
            raise ValueError("weights must have same dimensions as y")
        if np.any(w_mat <= 0) or np.any(np.isnan(w_mat)):
            raise ValueError("weights must be positive")
    else:
        w_mat = None

    y_mat = expr.copy()

    # Divide out array weights.
    if array_weights is not None:
        ws = np.sqrt(np.asarray(array_weights, dtype=np.float64))
        X = X * ws[:, np.newaxis]
        y_mat = y_mat * ws  # column-wise scale
        array_weights = None

    # Correlation matrix for block design.
    R_chol = None
    if block is not None:
        if correlation is None:
            raise ValueError("correlation must be set")
        block_vec = np.asarray(block)
        if block_vec.size != n:
            raise ValueError("Length of block does not match number of arrays")
        ub = np.unique(block_vec)
        Z = (block_vec[:, None] == ub[None, :]).astype(np.float64)
        cormatrix = Z @ (correlation * Z.T)
        np.fill_diagonal(cormatrix, 1.0)
        R_chol = linalg.cholesky(cormatrix, lower=False)

        if w_mat is None:
            # R: y <- t(backsolve(R, t(y), transpose=TRUE))
            #    X <- backsolve(R, X, transpose=TRUE)
            y_mat = linalg.solve_triangular(R_chol, y_mat.T, trans="T").T
            X = linalg.solve_triangular(R_chol, X, trans="T")

    # QR decomposition (fast path).
    Q, R_qr = linalg.qr(X, mode="full")
    rank = int(np.sum(np.abs(np.diag(R_qr)) > 1e-10))
    if rank < p:
        raise ValueError("design must be full column rank")

    if w_mat is None:
        # R: Effects <- t(qr.qty(qrX, t(y)))
        # qr.qty applies Q^T. Effects has shape (ngenes, n).
        Effects = (Q.T @ y_mat.T).T
        signc = np.sign(R_qr[p - 1, p - 1])
        if p > 1:
            # Keep columns p..n (1-indexed) -> p-1..n-1 (0-indexed)
            Effects = Effects[:, p - 1 :]
        if signc < 0:
            Effects[:, 0] = signc * Effects[:, 0]
    else:
        Effects = np.zeros((ngenes, n), dtype=np.float64)
        signc = np.zeros(ngenes)
        ws_mat = np.sqrt(w_mat)
        for g in range(ngenes):
            wX = X * ws_mat[g, :, np.newaxis]
            wy = y_mat[g, :] * ws_mat[g, :]
            if R_chol is not None:
                wy = linalg.solve_triangular(R_chol, wy, trans="T")
                wX = linalg.solve_triangular(R_chol, wX, trans="T")
            Qg, Rg = linalg.qr(wX, mode="full")
            signc[g] = np.sign(Rg[p - 1, p - 1])
            Effects[g, :] = Qg.T @ wy
        if p > 1:
            Effects = Effects[:, p - 1 :]
        Effects[:, 0] = signc * Effects[:, 0]

    if gene_weights is not None:
        gw = np.asarray(gene_weights, dtype=np.float64)
        Effects = np.sqrt(gw)[:, np.newaxis] * Effects

    return Effects


# ---------------------------------------------------------------------------
# roast / mroast
# ---------------------------------------------------------------------------

_SQRT2 = np.sqrt(2.0)


def _squeeze_var_with_prior(
    var: np.ndarray,
    df: float,
    var_prior,
    df_prior,
) -> np.ndarray:
    """Port of R limma's ``.squeezeVar``: posterior variance given prior."""
    var = np.asarray(var, dtype=np.float64)
    df_prior_arr = np.atleast_1d(df_prior).astype(np.float64)
    var_prior_arr = np.atleast_1d(var_prior).astype(np.float64)
    # (df.prior*var.prior + df*var) / (df.prior + df)
    num = df_prior_arr * var_prior_arr + df * var
    denom = df_prior_arr + df
    # Broadcast scalar priors if needed.
    return num / denom


def _roast_effects(
    effects: np.ndarray,
    gene_weights: np.ndarray | None,
    set_statistic: str,
    var_prior,
    df_prior,
    var_post,
    nrot: int,
    approx_zscore: bool,
    legacy: bool,
    rng: np.random.Generator,
    chunk: int = 1000,
) -> dict:
    """
    Rotation gene-set test given the effects matrix for one set.

    Port of R limma's ``.roastEffects``. The first column of ``effects`` is
    the primary (contrast) effect; the remaining columns are residual
    effects. Rows are genes already subset to the gene set of interest.
    """
    if legacy:
        chunk = nrot

    nset = effects.shape[0]
    neffects = effects.shape[1]
    df_residual = neffects - 1
    df_total = np.asarray(df_prior, dtype=np.float64) + df_residual

    # Observed z-statistics
    modt = effects[:, 0] / np.sqrt(np.asarray(var_post, dtype=np.float64))
    if approx_zscore and not legacy:
        df_total_winsor = np.minimum(df_total, 10000.0)
        modt = _zscore_t_bailey(modt, np.broadcast_to(df_total_winsor, modt.shape))
    else:
        modt = zscore_t(modt, df_total, approx=approx_zscore, method="hill")

    # Active proportions
    if gene_weights is None:
        a1 = float(np.mean(modt > _SQRT2))
        a2 = float(np.mean(modt < -_SQRT2))
    else:
        s = np.sign(gene_weights)
        ss = float(np.sum(np.abs(s)))
        a1 = float(np.sum(s * modt > _SQRT2)) / ss if ss > 0 else 0.0
        a2 = float(np.sum(s * modt < -_SQRT2)) / ss if ss > 0 else 0.0

    # Observed set statistics
    statobs = np.zeros(4)
    if set_statistic not in ("mean", "floormean", "mean50", "msq"):
        raise ValueError(
            f"set.statistic '{set_statistic}' not recognized. "
            "Must be 'mean', 'floormean', 'mean50' or 'msq'."
        )

    chimed = stats.norm.isf(0.25)

    if set_statistic == "mean":
        modt_use = gene_weights * modt if gene_weights is not None else modt
        m = float(np.mean(modt_use))
        statobs[0] = -m
        statobs[1] = m
        statobs[3] = float(np.mean(np.abs(modt_use)))
        modt = modt_use  # Pass to rotation loop
    elif set_statistic == "floormean":
        amodt = np.maximum(np.abs(modt), chimed)
        if gene_weights is not None:
            amodt = gene_weights * amodt
            modt = gene_weights * modt
        statobs[0] = float(np.mean(np.maximum(-modt, 0)))
        statobs[1] = float(np.mean(np.maximum(modt, 0)))
        statobs[2] = float(max(statobs[0], statobs[1]))
        statobs[3] = float(np.mean(amodt))
    elif set_statistic == "mean50":
        if nset % 2 == 0:
            half1 = nset // 2
            half2 = (
                half1 + 1
            )  # 1-based in R -> still means index half1 in 0-based for bottom half start
        else:
            half1 = nset // 2 + 1
            half2 = half1
        # R uses sort(modt, partial=half2); top half = s[1:half1], bottom = s[half2:nset]
        if gene_weights is not None:
            modt = gene_weights * modt
        s_sorted = np.sort(modt)
        # 0-based: top half = s_sorted[0:half1], bottom half = s_sorted[half2-1:nset]
        statobs[0] = float(-np.mean(s_sorted[:half1]))
        statobs[1] = float(np.mean(s_sorted[half2 - 1 : nset]))
        statobs[2] = float(max(statobs[0], statobs[1]))
        s_sorted = np.sort(np.abs(modt))
        statobs[3] = float(np.mean(s_sorted[half2 - 1 : nset]))
    elif set_statistic == "msq":
        modt2 = modt**2
        if gene_weights is not None:
            modt2 = np.abs(gene_weights) * modt2
            modt = gene_weights * modt
        statobs[0] = float(np.sum(modt2[modt < 0]) / nset)
        statobs[1] = float(np.sum(modt2[modt > 0]) / nset)
        statobs[2] = float(max(statobs[0], statobs[1]))
        statobs[3] = float(np.mean(modt2))

    # Rotation loop
    nchunk = int(np.ceil(nrot / chunk))
    nroti = int(np.ceil(nrot / nchunk))
    overshoot = nchunk * nroti - nrot

    count = np.zeros(4, dtype=np.int64)
    FinDf = np.isfinite(np.asarray(df_prior, dtype=np.float64))
    FinDf = np.atleast_1d(FinDf)
    df_prior_arr = np.atleast_1d(np.asarray(df_prior, dtype=np.float64))
    var_prior_arr = np.atleast_1d(np.asarray(var_prior, dtype=np.float64))

    half1_loc = None
    half2_loc = None
    if set_statistic == "mean50":
        if nset % 2 == 0:
            half1_loc = nset // 2
            half2_loc = half1_loc + 1
        else:
            half1_loc = nset // 2 + 1
            half2_loc = half1_loc

    df_total_winsor = None
    if approx_zscore and not legacy:
        df_total_winsor = np.minimum(df_total, 10000.0)

    for chunki in range(nchunk):
        if chunki == nchunk - 1:
            nroti_this = nroti - overshoot
        else:
            nroti_this = nroti
        statrot = np.zeros((nroti_this, 4))

        # Rotated primary effects: modtr has shape (nset, nroti_this)
        R_mat = rng.standard_normal((nroti_this, neffects))
        R_mat = R_mat / np.sqrt(np.sum(R_mat**2, axis=1, keepdims=True))
        modtr = effects @ R_mat.T  # (nset, nroti_this)

        # Moderated rotated variances
        if np.all(FinDf):
            s2r = (np.sum(effects**2, axis=1, keepdims=True) - modtr**2) / df_residual
            df_prior_b = df_prior_arr.reshape(-1, 1) if df_prior_arr.size > 1 else df_prior_arr[0]
            var_prior_b = (
                var_prior_arr.reshape(-1, 1) if var_prior_arr.size > 1 else var_prior_arr[0]
            )
            df_total_b = (
                df_total.reshape(-1, 1) if np.ndim(df_total) and df_total.size > 1 else df_total
            )
            s2r = (df_prior_b * var_prior_b + df_residual * s2r) / df_total_b
        elif np.any(FinDf):
            s2r = (np.sum(effects**2, axis=1, keepdims=True) - modtr**2) / df_residual
            if var_prior_arr.size > 1:
                s20 = var_prior_arr[FinDf].reshape(-1, 1)
            else:
                s20 = var_prior_arr[0]
            df_prior_sub = df_prior_arr[FinDf].reshape(-1, 1)
            df_total_sub = df_total[FinDf].reshape(-1, 1)
            s2r[FinDf, :] = (df_prior_sub * s20 + df_residual * s2r[FinDf, :]) / df_total_sub
        else:
            s2r = np.full_like(modtr, var_prior_arr[0] if var_prior_arr.size == 1 else 1.0)
            if var_prior_arr.size > 1:
                s2r = np.broadcast_to(var_prior_arr.reshape(-1, 1), modtr.shape).copy()

        # Rotated z-statistics
        modtr = modtr / np.sqrt(s2r)
        if approx_zscore and not legacy:
            dwinsor = np.broadcast_to(df_total_winsor.reshape(-1, 1), modtr.shape)
            modtr = _zscore_t_bailey(modtr, dwinsor)
        else:
            df_total_b = (
                np.broadcast_to(np.asarray(df_total).reshape(-1, 1), modtr.shape)
                if np.ndim(df_total) and np.asarray(df_total).size > 1
                else df_total
            )
            modtr = zscore_t(modtr, df_total_b, approx=approx_zscore, method="hill")

        if set_statistic == "mean":
            if gene_weights is not None:
                modtr = gene_weights[:, np.newaxis] * modtr
            m_r = np.mean(modtr, axis=0)
            statrot[:, 0] = -m_r
            statrot[:, 1] = m_r
            statrot[:, 3] = np.mean(np.abs(modtr), axis=0)
            count[0] += int(np.sum(statrot[:, 0] > statobs[0]) + np.sum(statrot[:, 1] > statobs[0]))
            count[1] += int(np.sum(statrot[:, 0] > statobs[1]) + np.sum(statrot[:, 1] > statobs[1]))
            count[3] += int(np.sum(statrot[:, 3] > statobs[3]))

        elif set_statistic == "floormean":
            amodtr = np.maximum(np.abs(modtr), chimed)
            if gene_weights is not None:
                amodtr = gene_weights[:, np.newaxis] * amodtr
                modtr = gene_weights[:, np.newaxis] * modtr
            statrot[:, 0] = np.mean(np.maximum(-modtr, 0), axis=0)
            statrot[:, 1] = np.mean(np.maximum(modtr, 0), axis=0)
            ub = statrot[:, 1] > statrot[:, 0]
            statrot[ub, 2] = statrot[ub, 1]
            statrot[~ub, 2] = statrot[~ub, 0]
            statrot[:, 3] = np.mean(amodtr, axis=0)
            count[0] += int(np.sum(statrot[:, 0] > statobs[0]) + np.sum(statrot[:, 1] > statobs[0]))
            count[1] += int(np.sum(statrot[:, 0] > statobs[1]) + np.sum(statrot[:, 1] > statobs[1]))
            count[2] += int(np.sum(statrot[:, 2] > statobs[2]))
            count[3] += int(np.sum(statrot[:, 3] > statobs[3]))

        elif set_statistic == "mean50":
            if gene_weights is not None:
                modtr = gene_weights[:, np.newaxis] * modtr
            for r in range(nroti_this):
                s_sorted = np.sort(modtr[:, r])
                statrot[r, 0] = -np.mean(s_sorted[:half1_loc])
                statrot[r, 1] = np.mean(s_sorted[half2_loc - 1 : nset])
                statrot[r, 2] = max(statrot[r, 0], statrot[r, 1])
                s_sorted = np.sort(np.abs(modtr[:, r]))
                statrot[r, 3] = np.mean(s_sorted[half2_loc - 1 : nset])
            count[0] += int(np.sum(statrot[:, 0] > statobs[0]) + np.sum(statrot[:, 1] > statobs[0]))
            count[1] += int(np.sum(statrot[:, 0] > statobs[1]) + np.sum(statrot[:, 1] > statobs[1]))
            count[2] += int(np.sum(statrot[:, 2] > statobs[2]))
            count[3] += int(np.sum(statrot[:, 3] > statobs[3]))

        elif set_statistic == "msq":
            if gene_weights is not None:
                gw_sqrt = np.sqrt(np.abs(gene_weights))
                modtr = gw_sqrt[:, np.newaxis] * modtr
            statrot[:, 0] = np.mean(np.maximum(-modtr, 0) ** 2, axis=0)
            statrot[:, 1] = np.mean(np.maximum(modtr, 0) ** 2, axis=0)
            ub = statrot[:, 1] > statrot[:, 0]
            statrot[ub, 2] = statrot[ub, 1]
            statrot[~ub, 2] = statrot[~ub, 0]
            statrot[:, 3] = np.mean(modtr**2, axis=0)
            count[0] += int(np.sum(statrot[:, 0] > statobs[0]) + np.sum(statrot[:, 1] > statobs[0]))
            count[1] += int(np.sum(statrot[:, 0] > statobs[1]) + np.sum(statrot[:, 1] > statobs[1]))
            count[2] += int(np.sum(statrot[:, 2] > statobs[2]))
            count[3] += int(np.sum(statrot[:, 3] > statobs[3]))

    if set_statistic == "mean":
        count[2] = int(min(count[0], count[1]))

    # Denominators: (2,2,1,1)*nrot + 1
    denom = np.array([2 * nrot + 1, 2 * nrot + 1, nrot + 1, nrot + 1], dtype=np.float64)
    p = (count + 1) / denom

    # Assemble data.frame-like output matching R:
    # rows: Down, Up, UpOrDown, Mixed; cols: Active.Prop, P.Value
    active = np.array([a2, a1, max(a1, a2), a1 + a2])
    pvals = p
    out_df = pd.DataFrame(
        {"active_prop": active, "p_value": pvals},
        index=["Down", "Up", "UpOrDown", "Mixed"],
    )
    return {"p_value": out_df, "ngenes_in_set": nset}


def _resolve_rng(rng):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _prep_roast_inputs(
    y,
    design,
    contrast,
    gene_weights,
    var_prior,
    df_prior,
    covariate_trend,
    lm_kwargs: dict,
):
    Effects = _lm_effects(
        y,
        design=design,
        contrast=contrast,
        array_weights=lm_kwargs.get("array_weights"),
        weights=lm_kwargs.get("weights"),
        block=lm_kwargs.get("block"),
        correlation=lm_kwargs.get("correlation"),
    )
    ngenes = Effects.shape[0]
    df_residual = Effects.shape[1] - 1

    s2 = np.mean(Effects[:, 1:] ** 2, axis=1)
    if var_prior is None or df_prior is None:
        sv = squeeze_var(
            s2,
            df_residual,
            covariate=covariate_trend,
            robust=lm_kwargs.get("robust", False),
            winsor_tail_p=lm_kwargs.get("winsor_tail_p", (0.05, 0.1)),
        )
        var_prior_out = sv["var_prior"]
        df_prior_out = sv["df_prior"]
        var_post = sv["var_post"]
    else:
        var_prior_out = var_prior
        df_prior_out = df_prior
        var_post = _squeeze_var_with_prior(s2, df_residual, var_prior, df_prior)

    return Effects, ngenes, df_residual, var_prior_out, df_prior_out, var_post


def _resolve_set_index(idx, ngenes):
    """Convert a gene-set selector to a NumPy int array of 0-based positions."""
    if idx is None:
        return np.arange(ngenes, dtype=np.int64)
    arr = np.asarray(idx)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.dtype.kind == "b":
        return np.where(arr)[0].astype(np.int64)
    return arr.astype(np.int64, copy=True)


def roast(
    y,
    index=None,
    design=None,
    contrast=None,
    geneid=None,
    set_statistic: str = "mean",
    gene_weights: np.ndarray | None = None,
    var_prior: float | None = None,
    df_prior: float | None = None,
    nrot: int = 1999,
    approx_zscore: bool = True,
    legacy: bool = False,
    rng=None,
    **lmfit_kwargs,
):
    """
    Rotation gene-set test (single set).

    Port of R limma's ``roast.default``. When ``index`` is a dict/list of
    sets, dispatches to :func:`mroast`.

    Parameters
    ----------
    y : ndarray, EList, AnnData, or DataFrame
        Expression data.
    index : array_like of int, dict, or list of array_like, optional
        Set members (0-based indices). A dict or list of arrays is treated
        as multiple sets and routed to :func:`mroast`.
    design : array_like, optional
        Design matrix. Defaults to ``y.design`` if available.
    contrast : int or array_like, optional
        Column index (0-based) or contrast vector. Defaults to the last
        column of ``design``.
    geneid : str or array_like, optional
        Optional gene identifier vector (or column name in ``y.genes``).
    set_statistic : {"mean", "floormean", "mean50", "msq"}, default "mean"
        Summary statistic of the moderated z-scores.
    gene_weights : array_like, optional
        Per-gene weights.
    var_prior, df_prior : float, optional
        Hyperparameters. If either is ``None``, both are estimated by
        :func:`squeeze_var`.
    nrot : int, default 1999
        Number of rotations.
    approx_zscore : bool, default True
        Use an approximation for the z-score transform.
    legacy : bool, default False
        If True, use ``zscore_t(..., method="hill")`` for all z-score
        computations (matches R's pre-2019 behaviour).
    rng : int, numpy.random.Generator, or None
        Random-number stream. Deterministic outputs match R; Monte-Carlo
        p-values use this stream.

    Returns
    -------
    dict
        ``{"p_value": DataFrame(Active.Prop, P.Value), "ngenes_in_set": int}``.
    """
    # If index is a list of sets, dispatch to mroast
    if isinstance(index, dict) or (
        isinstance(index, list)
        and len(index)
        and all(hasattr(x, "__len__") and not isinstance(x, str) for x in index)
    ):
        return mroast(
            y,
            index=index,
            design=design,
            contrast=contrast,
            geneid=geneid,
            set_statistic=set_statistic,
            gene_weights=gene_weights,
            var_prior=var_prior,
            df_prior=df_prior,
            nrot=nrot,
            approx_zscore=approx_zscore,
            legacy=legacy,
            rng=rng,
            **lmfit_kwargs,
        )

    rng_state = _resolve_rng(rng)

    # Trend covariate
    covariate = None
    if lmfit_kwargs.get("trend", False) or lmfit_kwargs.get("trend_var", False):
        covariate = np.mean(np.asarray(get_eawp(y)["exprs"]), axis=1)

    Effects, ngenes, df_residual, vp, dp, var_post = _prep_roast_inputs(
        y, design, contrast, gene_weights, var_prior, df_prior, covariate, lmfit_kwargs
    )

    # Subset to gene set
    idx = _resolve_set_index(index, ngenes)
    Effects_set = Effects[idx, :]
    vp_set = vp[idx] if np.ndim(vp) and np.size(vp) > 1 else vp
    dp_set = dp[idx] if np.ndim(dp) and np.size(dp) > 1 else dp
    var_post_set = var_post[idx] if np.ndim(var_post) and np.size(var_post) > 1 else var_post

    NGenesInSet = Effects_set.shape[0]

    # gene_weights alignment
    if gene_weights is not None:
        gw = np.asarray(gene_weights, dtype=np.float64).copy()
        if gw.size not in (NGenesInSet, ngenes):
            raise ValueError(
                "length of gene.weights doesn't agree with number of genes or size of set"
            )
        if gw.size == ngenes:
            gw = gw[idx]
    else:
        gw = None

    return _roast_effects(
        Effects_set,
        gene_weights=gw,
        set_statistic=set_statistic,
        var_prior=vp_set,
        df_prior=dp_set,
        var_post=var_post_set,
        nrot=nrot,
        approx_zscore=approx_zscore,
        legacy=legacy,
        rng=rng_state,
    )


def mroast(
    y,
    index,
    design=None,
    contrast=None,
    geneid=None,
    set_statistic: str = "mean",
    gene_weights: np.ndarray | None = None,
    var_prior: float | None = None,
    df_prior: float | None = None,
    nrot: int = 1999,
    approx_zscore: bool = True,
    legacy: bool = False,
    adjust_method: str = "BH",
    midp: bool = True,
    sort: str = "directional",
    rng=None,
    **lmfit_kwargs,
) -> pd.DataFrame:
    """
    Rotation gene-set test over many sets.

    Port of R limma's ``mroast.default``. See :func:`roast` for the
    single-set API; this version returns a :class:`pandas.DataFrame`.

    The ``sort`` argument accepts ``True``/``False`` (aliased to
    ``"directional"`` / ``"none"``) as well as the R strings.
    """
    rng_state = _resolve_rng(rng)

    covariate = None
    if lmfit_kwargs.get("trend", False):
        covariate = np.mean(np.asarray(get_eawp(y)["exprs"]), axis=1)

    Effects, ngenes, df_residual, vp, dp, var_post = _prep_roast_inputs(
        y, design, contrast, gene_weights, var_prior, df_prior, covariate, lmfit_kwargs
    )

    # Normalise index to a dict of label -> 0-based int index array.
    if index is None:
        index = {"set1": np.arange(ngenes, dtype=np.int64)}
    elif isinstance(index, dict):
        pass
    elif (
        isinstance(index, list)
        and len(index)
        and all(hasattr(x, "__len__") and not isinstance(x, str) for x in index)
    ):
        index = {f"set{i + 1}": v for i, v in enumerate(index)}
    else:
        index = {"set1": index}

    names = list(index.keys())
    if len(set(names)) != len(names):
        raise ValueError("Gene sets don't have unique names")
    nsets = len(index)
    if nsets == 0:
        raise ValueError("index is empty")

    if gene_weights is not None:
        gw_full = np.asarray(gene_weights, dtype=np.float64).copy()
        if gw_full.size != ngenes:
            raise ValueError("gene.weights vector should be of length nrow(y)")
    else:
        gw_full = None

    pv = np.zeros((nsets, 4))
    active = np.zeros((nsets, 4))
    NGenes = np.zeros(nsets, dtype=np.int64)

    for i, name in enumerate(names):
        g = _resolve_set_index(index[name], ngenes)
        E = Effects[g, :]
        vp_i = vp[g] if np.ndim(vp) and np.size(vp) > 1 else vp
        dp_i = dp[g] if np.ndim(dp) and np.size(dp) > 1 else dp
        vpost_i = var_post[g] if np.ndim(var_post) and np.size(var_post) > 1 else var_post
        gw_i = gw_full[g] if gw_full is not None else None

        out = _roast_effects(
            E,
            gene_weights=gw_i,
            set_statistic=set_statistic,
            var_prior=vp_i,
            df_prior=dp_i,
            var_post=vpost_i,
            nrot=nrot,
            approx_zscore=approx_zscore,
            legacy=legacy,
            rng=rng_state,
        )
        # R order: Down, Up, UpOrDown, Mixed
        pv[i, :] = out["p_value"]["p_value"].values
        active[i, :] = out["p_value"]["active_prop"].values
        NGenes[i] = out["ngenes_in_set"]

    Up_wins = pv[:, 1] < pv[:, 0]
    Direction = np.where(Up_wins, "Up", "Down")

    TwoSidedP2 = pv[:, 2].copy()
    MixedP2 = pv[:, 3].copy()
    if midp:
        TwoSidedP2 = TwoSidedP2 - 1.0 / (2.0 * (nrot + 1))
        MixedP2 = MixedP2 - 1.0 / (2.0 * (nrot + 1))

    tab = pd.DataFrame(
        {
            "n_genes": NGenes,
            "prop_down": active[:, 0],
            "prop_up": active[:, 1],
            "direction": Direction,
            "p_value": pv[:, 2],
            "fdr": p_adjust(TwoSidedP2, method=adjust_method),
            "p_value_mixed": pv[:, 3],
            "fdr_mixed": p_adjust(MixedP2, method=adjust_method),
        },
        index=names,
    )

    if midp:
        tab["fdr"] = np.maximum(tab["fdr"].values, pv[:, 2])
        tab["fdr_mixed"] = np.maximum(tab["fdr_mixed"].values, pv[:, 3])

    # Sort
    if isinstance(sort, bool):
        sort = "directional" if sort else "none"
    if sort not in ("directional", "mixed", "none"):
        raise ValueError(f"sort '{sort}' not recognized. Must be 'directional', 'mixed' or 'none'.")
    if sort == "none":
        return tab
    if sort == "directional":
        prop = np.maximum(tab["prop_up"].values, tab["prop_down"].values)
        o = np.lexsort(
            (tab["p_value_mixed"].values, -NGenes.astype(float), -prop, tab["p_value"].values)
        )
    else:
        prop = tab["prop_up"].values + tab["prop_down"].values
        o = np.lexsort(
            (tab["p_value"].values, -NGenes.astype(float), -prop, tab["p_value_mixed"].values)
        )
    return tab.iloc[o]


# ---------------------------------------------------------------------------
# fry
# ---------------------------------------------------------------------------


def _fry_effects(
    effects: np.ndarray,
    index: dict,
    geneid,
    gene_weights,
    sort,
) -> pd.DataFrame:
    G = effects.shape[0]
    neffects = effects.shape[1]
    df_residual = neffects - 1

    names = list(index.keys())
    nsets = len(names)
    if len(set(names)) != len(names):
        raise ValueError("Gene sets don't have unique names")

    NGenes = np.zeros(nsets, dtype=np.int64)
    PValue_Mixed = np.zeros(nsets)
    t_stat = np.zeros(nsets)

    for i, nm in enumerate(names):
        iset = _resolve_set_index(index[nm], G)
        EffectsSet = effects[iset, :].copy()
        if gene_weights is not None:
            iw = gene_weights[iset]
            EffectsSet = iw[:, np.newaxis] * EffectsSet
        MeanEffectsSet = np.mean(EffectsSet, axis=0)
        denom = np.sqrt(np.mean(MeanEffectsSet[1:] ** 2))
        t_stat[i] = MeanEffectsSet[0] / denom if denom > 0 else np.nan
        NGenes[i] = EffectsSet.shape[0]

        if NGenes[i] > 1:
            # SVD of EffectsSet; we need the singular values only.
            # R: SVD <- svd(EffectsSet, nu=0); A <- SVD$d^2
            sv = np.linalg.svd(EffectsSet, compute_uv=False)
            A = sv**2
            d1 = A.size
            d = d1 - 1
            beta_mean = 1.0 / d1
            beta_var = d / d1 / d1 / (d1 / 2.0 + 1.0)
            Fobs = (np.sum(EffectsSet[:, 0] ** 2) - A[-1]) / (A[0] - A[-1])
            Frb_mean = (np.sum(A) * beta_mean - A[-1]) / (A[0] - A[-1])
            COV = np.full((d1, d1), -beta_var / d)
            np.fill_diagonal(COV, beta_var)
            Frb_var = float(A @ COV @ A) / (A[0] - A[-1]) ** 2
            alphaplusbeta = Frb_mean * (1.0 - Frb_mean) / Frb_var - 1.0
            alpha = alphaplusbeta * Frb_mean
            beta_param = alphaplusbeta - alpha
            PValue_Mixed[i] = stats.beta.sf(Fobs, alpha, beta_param)

    Direction = np.where(t_stat < 0, "Down", "Up")
    PValue = 2.0 * stats.t.cdf(-np.abs(t_stat), df=df_residual)
    singleton = NGenes == 1
    PValue_Mixed[singleton] = PValue[singleton]

    if nsets > 1:
        tab = pd.DataFrame(
            {
                "n_genes": NGenes,
                "direction": Direction,
                "p_value": PValue,
                "fdr": p_adjust(PValue, method="BH"),
                "p_value_mixed": PValue_Mixed,
                "fdr_mixed": p_adjust(PValue_Mixed, method="BH"),
            },
            index=names,
        )
    else:
        tab = pd.DataFrame(
            {
                "n_genes": NGenes,
                "direction": Direction,
                "p_value": PValue,
                "p_value_mixed": PValue_Mixed,
            },
            index=names,
        )

    if isinstance(sort, bool):
        sort = "directional" if sort else "none"
    if sort not in ("directional", "mixed", "none"):
        raise ValueError(f"sort '{sort}' not recognized. Must be 'directional', 'mixed' or 'none'.")
    if sort == "none":
        return tab
    if sort == "directional":
        o = np.lexsort((tab["p_value_mixed"].values, -NGenes.astype(float), tab["p_value"].values))
    else:
        o = np.lexsort((tab["p_value"].values, -NGenes.astype(float), tab["p_value_mixed"].values))
    return tab.iloc[o]


def fry(
    y,
    index=None,
    design=None,
    contrast=None,
    geneid=None,
    gene_weights: np.ndarray | None = None,
    standardize: str = "posterior.sd",
    sort="directional",
    **lmfit_kwargs,
) -> pd.DataFrame:
    """
    Fast closed-form limit of ``roast`` (``nrot -> Inf`` with ``df.prior=Inf``).

    Port of R limma's ``fry.default``.
    """
    if gene_weights is not None:
        gw = np.asarray(gene_weights, dtype=np.float64)
        ea = get_eawp(y)
        if gw.size != ea["exprs"].shape[0]:
            raise ValueError("length of gene.weights should equal nrow(y)")
    else:
        gw = None

    if standardize not in ("none", "residual.sd", "posterior.sd", "p2"):
        raise ValueError(
            f"standardize '{standardize}' not recognized. "
            "Must be 'none', 'residual.sd', 'posterior.sd' or 'p2'."
        )

    covariate = None
    if lmfit_kwargs.get("trend", False):
        covariate = np.mean(np.asarray(get_eawp(y)["exprs"]), axis=1)

    Effects = _lm_effects(
        y,
        design=design,
        contrast=contrast,
        array_weights=lmfit_kwargs.get("array_weights"),
        weights=lmfit_kwargs.get("weights"),
        block=lmfit_kwargs.get("block"),
        correlation=lmfit_kwargs.get("correlation"),
    )
    G = Effects.shape[0]
    df_residual = Effects.shape[1] - 1

    if standardize != "none":
        # Gauss-Legendre 128 nodes on [0,1]; Eu2max in R uses "uniform" dist.
        gq_nodes, gq_wts = np.polynomial.legendre.leggauss(128)
        gq_nodes = (gq_nodes + 1) / 2.0
        gq_wts = gq_wts / 2.0
        Eu2max = float(
            np.sum(
                (df_residual + 1) * gq_nodes**df_residual * stats.chi2.ppf(gq_nodes, df=1) * gq_wts
            )
        )
        u2max = np.max(Effects**2, axis=1)
        s2_robust = (np.sum(Effects**2, axis=1) - u2max) / (df_residual + 1 - Eu2max)

        if standardize == "p2":
            sv = squeeze_var(
                s2_robust,
                df=0.92 * df_residual,
                covariate=covariate,
                robust=lmfit_kwargs.get("robust", False),
                winsor_tail_p=lmfit_kwargs.get("winsor_tail_p", (0.05, 0.1)),
            )
            s2_robust = sv["var_post"]
        elif standardize == "posterior.sd":
            s2 = np.mean(Effects[:, 1:] ** 2, axis=1)
            if lmfit_kwargs.get("robust", False):
                # fitFDistRobustly path - use squeeze_var with robust=True and
                # lift out the prior var / df it estimates.
                sv = squeeze_var(
                    s2,
                    df=df_residual,
                    covariate=covariate,
                    robust=True,
                    winsor_tail_p=lmfit_kwargs.get("winsor_tail_p", (0.05, 0.1)),
                )
                df_prior = sv["df_prior"]
                var_prior = sv["var_prior"]
            else:
                fit = fit_f_dist(s2, df1=df_residual, covariate=covariate)
                df_prior = fit["df2"]
                var_prior = fit["scale"]
            s2_robust = _squeeze_var_with_prior(
                s2_robust, df=0.92 * df_residual, var_prior=var_prior, df_prior=df_prior
            )
        # "residual.sd": use s2_robust as-is

        Effects = Effects / np.sqrt(s2_robust)[:, np.newaxis]

    # Normalise index
    if index is None:
        index = {"set1": np.arange(G, dtype=np.int64)}
    elif isinstance(index, dict):
        pass
    elif (
        isinstance(index, list)
        and len(index)
        and all(hasattr(x, "__len__") and not isinstance(x, str) for x in index)
    ):
        index = {f"set{i + 1}": v for i, v in enumerate(index)}
    else:
        index = {"set1": index}

    return _fry_effects(Effects, index, geneid=geneid, gene_weights=gw, sort=sort)


# ---------------------------------------------------------------------------
# camera / inter_gene_correlation / camera_pr
# ---------------------------------------------------------------------------


def inter_gene_correlation(y: np.ndarray, design: np.ndarray) -> dict:
    """
    Variance-inflation factor and inter-gene correlation.

    Port of R limma's ``interGeneCorrelation``. Returns
    ``{"vif": ..., "correlation": ...}``.
    """
    y = np.asarray(y, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    m = y.shape[0]
    Q, R = linalg.qr(design, mode="full")
    rank = int(np.sum(np.abs(np.diag(R)) > 1e-10))
    # R: y <- qr.qty(qrdesign, t(y))[-(1:rank),] -> residual rows of Q^T y^T
    resid = (Q.T @ y.T)[rank:, :]
    # resid has shape (n - rank, m). R: y <- t(y) / sqrt(colMeans(y^2))
    # colMeans along dim=1 of resid gives per-gene mean of residuals^2.
    col_means = np.mean(resid**2, axis=0)
    col_means = np.where(col_means > 0, col_means, 1e-300)
    ny = resid / np.sqrt(col_means)  # shape (n-rank, m)
    # vif = m * mean(colMeans(y)^2); here colMeans on ny is along axis=0
    # but after t() -> t() in R, it's the mean across residual rows per gene.
    vif = m * float(np.mean(np.mean(ny, axis=0) ** 2))
    correlation = (vif - 1) / (m - 1)
    return {"vif": vif, "correlation": correlation}


def camera(
    y,
    index,
    design=None,
    contrast=None,
    weights: np.ndarray | None = None,
    use_ranks: bool = False,
    allow_neg_cor: bool = False,
    inter_gene_cor: float | None = 0.01,
    trend_var: bool = False,
    sort: bool = True,
    directional: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Competitive gene-set test with inter-gene correlation.

    Port of R limma's ``camera.default``.
    """
    if kwargs:
        warnings.warn(f"Extra arguments disregarded: {list(kwargs.keys())}")

    ea = get_eawp(y)
    y_mat = ea["exprs"]
    G, n = y_mat.shape
    ID = None
    if ea.get("probes") is not None and hasattr(ea["probes"], "index"):
        ID = list(ea["probes"].index)
    if G < 3:
        raise ValueError("Too few genes in dataset: need at least 3")

    if not isinstance(index, dict):
        if (
            isinstance(index, list)
            and len(index)
            and all(hasattr(x, "__len__") and not isinstance(x, str) for x in index)
        ):
            index = {f"set{i + 1}": v for i, v in enumerate(index)}
        else:
            index = {"set1": index}
    nsets = len(index)
    if nsets == 0:
        raise ValueError("index is empty")

    if design is None:
        design = ea.get("design")
    if design is None:
        raise ValueError("design matrix not specified")
    design = np.asarray(design, dtype=np.float64)
    if design.shape[0] != n:
        raise ValueError("row dimension of design matrix must match column dimension of data")
    p = design.shape[1]
    df_residual = n - p
    if df_residual < 1:
        raise ValueError("No residual df: cannot compute t-tests")

    if weights is None:
        weights = ea.get("weights")

    fixed_cor = inter_gene_cor is not None and not (
        np.ndim(inter_gene_cor) == 0 and np.isnan(inter_gene_cor)
    )

    if not directional:
        if not use_ranks:
            use_ranks = True
            warnings.warn("Setting `use.ranks=TRUE` for non-directional tests")

    if fixed_cor:
        df_camera = np.inf if use_ranks else G - 2
    else:
        df_camera = min(df_residual, G - 2)

    # Weight handling (array weights first)
    y_use = y_mat.copy()
    design_use = design.copy()
    if weights is not None:
        w_arr = np.asarray(weights, dtype=np.float64)
        if np.any(w_arr <= 0):
            raise ValueError("weights must be positive")
        if w_arr.size == n:
            sw = np.sqrt(w_arr)
            y_use = y_use * sw
            design_use = design_use * sw[:, np.newaxis]
            weights = None
        else:
            if w_arr.size == G:
                weights = np.broadcast_to(w_arr[:, np.newaxis], (G, n)).copy()
            else:
                weights = w_arr
            if weights.shape != y_use.shape:
                raise ValueError("weights not conformal with y")

    # Reform design so contrast is the last column.
    if contrast is None:
        contrast = p - 1
    contrast_arr = np.atleast_1d(np.asarray(contrast))
    if contrast_arr.size == 1 and np.issubdtype(contrast_arr.dtype, np.integer):
        k = int(contrast_arr.item())
        if k < p - 1:
            keep = [j for j in range(p) if j != k]
            design_use = np.column_stack([design_use[:, keep], design_use[:, k : k + 1]])
    else:
        contrast_vec = np.asarray(contrast, dtype=np.float64)
        Q_c, R_c = linalg.qr(contrast_vec.reshape(-1, 1), mode="full")
        rotated = design_use @ Q_c
        if np.sign(R_c[0, 0]) < 0:
            rotated[:, 0] = -rotated[:, 0]
        design_use = np.column_stack([rotated[:, 1:], rotated[:, 0:1]])

    if weights is None:
        Q, R_qr = linalg.qr(design_use, mode="full")
        rank = int(np.sum(np.abs(np.diag(R_qr)) > 1e-10))
        if rank < p:
            raise ValueError("design matrix is not of full rank")
        effects = Q.T @ y_use.T  # (n, G)
        unscaledt = effects[p - 1, :].copy()
        if R_qr[p - 1, p - 1] < 0:
            unscaledt = -unscaledt
    else:
        effects = np.zeros((n, G))
        unscaledt = np.zeros(G)
        sw_mat = np.sqrt(weights)
        y_w = y_use * sw_mat
        for g in range(G):
            xw = design_use * sw_mat[g, :, np.newaxis]
            Qg, Rg = linalg.qr(xw, mode="full")
            rankg = int(np.sum(np.abs(np.diag(Rg)) > 1e-10))
            if rankg < p:
                raise ValueError(f"weighted design matrix not of full rank for gene {g}")
            effects[:, g] = Qg.T @ y_w[g, :]
            unscaledt[g] = effects[p - 1, g]
            if Rg[p - 1, p - 1] < 0:
                unscaledt[g] = -unscaledt[g]

    # Standardised residuals
    U = effects[p:, :]
    sigma2 = np.mean(U**2, axis=0)
    U_t = U.T / np.sqrt(np.maximum(sigma2, 1e-8))[:, None]  # (G, n-p)

    A = np.mean(y_mat, axis=1) if trend_var else None
    sv = squeeze_var(sigma2, df=df_residual, covariate=A)
    var_post = sv["var_post"]
    df_prior = sv["df_prior"]
    modt = unscaledt / np.sqrt(var_post)

    if use_ranks:
        Stat = modt.copy()
        if not directional:
            Stat = np.abs(Stat)
    else:
        df_total = min(df_residual + df_prior, G * df_residual)
        Stat = zscore_t(modt, df=df_total, approx=True, method="hill")

    if not use_ranks:
        meanStat = float(np.mean(Stat))
        varStat = float(np.var(Stat, ddof=1))

    NGenes = np.zeros(nsets, dtype=np.int64)
    Correlation = np.zeros(nsets)
    Down = np.zeros(nsets)
    Up = np.zeros(nsets)
    names = list(index.keys())
    for i, nm in enumerate(names):
        iset = index[nm]
        iset = np.asarray(iset)
        if iset.dtype.kind in ("U", "S", "O") and ID is not None:
            iset = np.array([ID.index(s) for s in iset if s in ID], dtype=np.int64)
        else:
            iset = iset.astype(np.int64, copy=False)
        StatInSet = Stat[iset]
        m = iset.size
        m2 = G - m
        if fixed_cor:
            corr = inter_gene_cor if np.ndim(inter_gene_cor) == 0 else inter_gene_cor[i]
            vif = 1 + (m - 1) * corr
        else:
            if m > 1:
                Uset = U_t[iset, :]
                vif = m * float(np.mean(np.mean(Uset, axis=0) ** 2))
                corr = (vif - 1) / (m - 1)
            else:
                vif = 1.0
                corr = np.nan

        NGenes[i] = m
        Correlation[i] = corr
        if use_ranks:
            corr_use = max(0, corr) if not allow_neg_cor else corr
            pv = rank_sum_test_with_correlation(
                iset, statistics=Stat, correlation=corr_use, df=df_camera
            )
            Down[i] = pv["less"]
            Up[i] = pv["greater"]
        else:
            if not allow_neg_cor:
                vif = max(1, vif)
            meanStatInSet = float(np.mean(StatInSet))
            delta = G / m2 * (meanStatInSet - meanStat)
            varStatPooled = ((G - 1) * varStat - delta**2 * m * m2 / G) / (G - 2)
            two_sample_t = delta / np.sqrt(varStatPooled * (vif / m + 1.0 / m2))
            if np.isinf(df_camera):
                Down[i] = stats.norm.cdf(two_sample_t)
                Up[i] = stats.norm.sf(two_sample_t)
            else:
                Down[i] = stats.t.cdf(two_sample_t, df=df_camera)
                Up[i] = stats.t.sf(two_sample_t, df=df_camera)

    TwoSided = 2 * np.minimum(Down, Up)

    if directional:
        Direction = np.where(Down < Up, "Down", "Up")
        if fixed_cor:
            tab = pd.DataFrame(
                {
                    "n_genes": NGenes,
                    "direction": Direction,
                    "p_value": TwoSided,
                },
                index=names,
            )
        else:
            tab = pd.DataFrame(
                {
                    "n_genes": NGenes,
                    "correlation": Correlation,
                    "direction": Direction,
                    "p_value": TwoSided,
                },
                index=names,
            )
    else:
        if fixed_cor:
            tab = pd.DataFrame(
                {"n_genes": NGenes, "p_value": Up},
                index=names,
            )
        else:
            tab = pd.DataFrame(
                {
                    "n_genes": NGenes,
                    "correlation": Correlation,
                    "p_value": Up,
                },
                index=names,
            )

    if nsets > 1:
        tab["fdr"] = p_adjust(tab["p_value"].values, method="BH")

    if sort and nsets > 1:
        o = np.argsort(tab["p_value"].values, kind="stable")
        tab = tab.iloc[o]

    return tab


def camera_pr(
    statistic,
    index,
    use_ranks: bool = False,
    inter_gene_cor=0.01,
    sort: bool = True,
    directional: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Pre-ranked competitive gene-set test.

    Port of R limma's ``cameraPR.default``.
    """
    if kwargs:
        warnings.warn(f"Extra arguments disregarded: {list(kwargs.keys())}")

    if isinstance(statistic, dict):
        raise TypeError("statistic should be a numeric vector")

    ID = None
    stat_arr: np.ndarray
    if isinstance(statistic, pd.Series):
        ID = list(statistic.index)
        stat_arr = statistic.values.astype(np.float64)
    else:
        stat_arr = np.asarray(statistic, dtype=np.float64)
    if np.any(np.isnan(stat_arr)):
        raise ValueError("NA values for statistic not allowed")
    G = stat_arr.size
    if G < 3:
        raise ValueError("Too few genes in dataset: need at least 3")

    if not isinstance(index, dict):
        if (
            isinstance(index, list)
            and len(index)
            and all(hasattr(x, "__len__") and not isinstance(x, str) for x in index)
        ):
            index = {f"set{i + 1}": v for i, v in enumerate(index)}
        else:
            index = {"set1": index}
    nsets = len(index)

    # inter_gene_cor handling
    igc = np.atleast_1d(np.asarray(inter_gene_cor, dtype=np.float64))
    if np.any(np.isnan(igc)):
        raise ValueError("NA inter.gene.cor not allowed")
    if np.max(np.abs(igc)) >= 1:
        raise ValueError("`inter.gene.cor` must be strictly between -1 and 1")
    if igc.size > 1:
        if igc.size != nsets:
            raise ValueError("Length of `inter.gene.cor` doesn't match number of sets")
        fixed_cor = False
        igc_arr = igc
    else:
        fixed_cor = True
        igc_arr = np.full(nsets, igc[0])

    if not directional:
        if np.min(stat_arr) < 0:
            stat_arr = np.abs(stat_arr)
            warnings.warn("Converting `statistic` to absolute values for non-directional tests")
        if not use_ranks:
            use_ranks = True
            warnings.warn("Setting `use.ranks=TRUE` for non-directional tests")

    df_camera = np.inf if use_ranks else G - 2

    meanStat = float(np.mean(stat_arr))
    varStat = float(np.var(stat_arr, ddof=1))

    NGenes = np.zeros(nsets, dtype=np.int64)
    Down = np.zeros(nsets)
    Up = np.zeros(nsets)
    names = list(index.keys())
    for i, nm in enumerate(names):
        iset = np.asarray(index[nm])
        if iset.dtype.kind in ("U", "S", "O") and ID is not None:
            iset = np.array([ID.index(s) for s in iset if s in ID], dtype=np.int64)
        else:
            iset = iset.astype(np.int64, copy=False)
        StatInSet = stat_arr[iset]
        m = iset.size
        NGenes[i] = m
        if use_ranks:
            pv = rank_sum_test_with_correlation(
                iset, statistics=stat_arr, correlation=igc_arr[i], df=df_camera
            )
            Down[i] = pv["less"]
            Up[i] = pv["greater"]
        else:
            vif = 1 + (m - 1) * igc_arr[i]
            m2 = G - m
            meanStatInSet = float(np.mean(StatInSet))
            delta = G / m2 * (meanStatInSet - meanStat)
            varStatPooled = ((G - 1) * varStat - delta**2 * m * m2 / G) / (G - 2)
            two_sample_t = delta / np.sqrt(varStatPooled * (vif / m + 1.0 / m2))
            if np.isinf(df_camera):
                Down[i] = stats.norm.cdf(two_sample_t)
                Up[i] = stats.norm.sf(two_sample_t)
            else:
                Down[i] = stats.t.cdf(two_sample_t, df=df_camera)
                Up[i] = stats.t.sf(two_sample_t, df=df_camera)

    TwoSided = 2 * np.minimum(Down, Up)

    if directional:
        Direction = np.where(Down < Up, "Down", "Up")
        if fixed_cor:
            tab = pd.DataFrame(
                {
                    "n_genes": NGenes,
                    "direction": Direction,
                    "p_value": TwoSided,
                },
                index=names,
            )
        else:
            tab = pd.DataFrame(
                {
                    "n_genes": NGenes,
                    "correlation": igc_arr,
                    "direction": Direction,
                    "p_value": TwoSided,
                },
                index=names,
            )
    else:
        if fixed_cor:
            tab = pd.DataFrame(
                {"n_genes": NGenes, "p_value": Up},
                index=names,
            )
        else:
            tab = pd.DataFrame(
                {"n_genes": NGenes, "correlation": igc_arr, "p_value": Up},
                index=names,
            )

    if nsets > 1:
        tab["fdr"] = p_adjust(tab["p_value"].values, method="BH")

    if sort and nsets > 1:
        o = np.argsort(tab["p_value"].values, kind="stable")
        tab = tab.iloc[o]

    return tab


# ---------------------------------------------------------------------------
# romer
# ---------------------------------------------------------------------------


def _mean_half(x: np.ndarray, n: int) -> tuple:
    """Return (top_half_mean, bottom_half_mean) matching R's .meanHalf."""
    l = x.size
    a = np.sort(x)
    top = float(np.mean(a[:n]))
    if l % 2 == 0:
        bottom = float(np.mean(a[n:]))
    else:
        bottom = float(np.mean(a[n - 1 :]))
    return top, bottom


def romer(
    y,
    index,
    design=None,
    contrast=None,
    array_weights: np.ndarray | None = None,
    block=None,
    correlation: float | None = None,
    set_statistic: str = "mean",
    nrot: int = 9999,
    shrink_resid: bool = True,
    rng=None,
    **kwargs,
) -> pd.DataFrame:
    """
    Rotation mean-rank gene-set enrichment analysis.

    Port of R limma's ``romer.default``.
    """
    if kwargs:
        warnings.warn(f"Extra arguments disregarded: {list(kwargs.keys())}")

    rng_state = _resolve_rng(rng)

    ea = get_eawp(y)
    y_mat = ea["exprs"].copy()
    ngenes, n = y_mat.shape

    if not isinstance(index, dict):
        if (
            isinstance(index, list)
            and len(index)
            and all(hasattr(x, "__len__") and not isinstance(x, str) for x in index)
        ):
            index = {f"set{i + 1}": v for i, v in enumerate(index)}
        else:
            index = {"set": index}
    nsets = len(index)
    if nsets == 0:
        raise ValueError("index is empty")
    set_names = list(index.keys())
    SetSizes = np.array(
        [_resolve_set_index(index[nm], ngenes).size for nm in set_names],
        dtype=np.int64,
    )

    if design is None:
        raise ValueError("design matrix not specified")
    design = np.asarray(design, dtype=np.float64)
    if design.shape[0] != n:
        raise ValueError("row dimension of design matrix must match column dimension of data")
    ne = non_estimable(design)
    if ne is not None:
        print("Coefficients not estimable:", " ".join(ne))
    p = design.shape[1]
    if p < 2:
        raise ValueError("design needs at least two columns")
    d = n - p

    if contrast is None:
        contrast = p - 1

    contrast_arr = np.atleast_1d(np.asarray(contrast))
    if contrast_arr.size == 1 and np.issubdtype(contrast_arr.dtype, np.integer):
        k = int(contrast_arr.item())
        if k < p - 1:
            keep = [j for j in range(p) if j != k]
            design = np.column_stack([design[:, keep], design[:, k : k + 1]])
    else:
        contrast_vec = np.asarray(contrast, dtype=np.float64)
        Q_c, R_c = linalg.qr(contrast_vec.reshape(-1, 1), mode="full")
        rotated = design @ Q_c
        if np.sign(R_c[0, 0]) < 0:
            rotated[:, 0] = -rotated[:, 0]
        design = np.column_stack([rotated[:, 1:], rotated[:, 0:1]])

    if array_weights is not None:
        aw = np.asarray(array_weights, dtype=np.float64)
        if np.any(aw <= 0):
            raise ValueError("array.weights must be positive")
        if aw.size != n:
            raise ValueError("Length of array.weights doesn't match number of array")
        design = design * np.sqrt(aw)[:, np.newaxis]
        y_mat = y_mat * np.sqrt(aw)

    if block is not None:
        if correlation is None:
            raise ValueError("correlation must be set")
        block_vec = np.asarray(block)
        if block_vec.size != n:
            raise ValueError("Length of block does not match number of arrays")
        ub = np.unique(block_vec)
        Z = (block_vec[:, None] == ub[None, :]).astype(np.float64)
        cormatrix = Z @ (correlation * Z.T)
        np.fill_diagonal(cormatrix, 1.0)
        R_chol = linalg.cholesky(cormatrix, lower=False)
        y_mat = linalg.solve_triangular(R_chol, y_mat.T, trans="T").T
        design = linalg.solve_triangular(R_chol, design, trans="T")

    Q, R_qr = linalg.qr(design, mode="full")
    signc = float(np.sign(R_qr[p - 1, p - 1]))
    effects = Q.T @ y_mat.T  # (n, ngenes)

    # s2 <- colMeans(effects[-(1:p),]^2)
    s2 = np.mean(effects[p:, :] ** 2, axis=0)

    sv = squeeze_var(s2, df=d)
    d0 = sv["df_prior"]
    s02 = sv["var_prior"]
    sd_post = np.sqrt(sv["var_post"])

    # Y <- effects[-(1:p0),,drop=FALSE]   p0 = p-1
    p0 = p - 1
    Y = effects[p0:, :].copy()  # (d+1, ngenes)
    YY = np.sum(Y**2, axis=0)
    B = Y[0, :]
    modt = signc * B / sd_post

    if shrink_resid:
        from .utils import p_adjust as _p_adjust  # noqa: F401 (not used)
        from .utils import prop_true_null as _prop_true_null

        p_value = 2 * stats.t.sf(np.abs(modt), df=d0 + d)
        proportion = 1 - _prop_true_null(p_value)
        stdev_unscaled = np.full(
            ngenes,
            1.0
            / abs(
                R_qr[
                    int(np.sum(np.abs(np.diag(R_qr)) > 1e-10)) - 1,
                    int(np.sum(np.abs(np.diag(R_qr)) > 1e-10)) - 1,
                ]
            ),
        )
        var_unscaled = stdev_unscaled**2
        df_total = np.full(ngenes, d) + sv["df_prior"]
        stdev_coef_lim = (0.1, 4.0)
        var_prior_lim = (
            stdev_coef_lim[0] ** 2 / sv["var_prior"],
            stdev_coef_lim[1] ** 2 / sv["var_prior"],
        )
        from .ebayes import _tmixture_vector

        var_prior = _tmixture_vector(modt, stdev_unscaled, df_total, proportion, var_prior_lim)
        if np.isnan(var_prior):
            var_prior = 1.0 / sv["var_prior"]
            warnings.warn("Estimation of var.prior failed - set to default value")
        r = (var_unscaled + var_prior) / var_unscaled
        if sv["df_prior"] > 1e6:
            kernel = modt**2 * (1 - 1 / r) / 2
        else:
            kernel = (1 + df_total) / 2 * np.log((modt**2 + df_total) / (modt**2 / r + df_total))
        lods = np.log(proportion / (1 - proportion)) - np.log(r) / 2 + kernel
        ProbDE = np.exp(lods) / (1 + np.exp(lods))
        Y[0, :] = Y[0, :] * np.sqrt(var_unscaled / (var_unscaled + var_prior * ProbDE))

    if set_statistic not in ("mean", "floormean", "mean50"):
        raise ValueError(
            f"set.statistic '{set_statistic}' not recognized. "
            "Must be 'mean', 'floormean', or 'mean50'."
        )

    # Pre-compute per-gene-set membership structure (AllIndices, Set)
    all_indices = np.concatenate([_resolve_set_index(index[nm], ngenes) for nm in set_names])
    Set_vec = np.concatenate(
        [
            np.full(_resolve_set_index(index[nm], ngenes).size, si, dtype=np.int64)
            for si, nm in enumerate(set_names)
        ]
    )

    def _rank_r(x):
        # R's rank() with default ties="average"
        return stats.rankdata(x, method="average")

    def _rowsum_mean(rank_mat):
        # rank_mat shape (len(all_indices), ncol); group sum by Set_vec, divide
        # by SetSizes.
        if rank_mat.ndim == 1:
            rank_mat = rank_mat.reshape(-1, 1)
        out = np.zeros((nsets, rank_mat.shape[1]))
        for c in range(rank_mat.shape[1]):
            out[:, c] = np.bincount(Set_vec, weights=rank_mat[:, c], minlength=nsets)
        out = out / SetSizes[:, np.newaxis]
        return out

    p_value = np.zeros((nsets, 3))

    if set_statistic == "mean":
        obs_ranks = np.column_stack(
            [_rank_r(modt), ngenes - _rank_r(modt) + 1, _rank_r(np.abs(modt))]
        )
        obs_set_ranks = _rowsum_mean(obs_ranks[all_indices, :])

        for _ in range(nrot):
            R_vec = rng_state.standard_normal((1, d + 1))
            R_vec = R_vec / np.sqrt(np.sum(R_vec**2))
            Br = (R_vec @ Y).ravel()
            s2r = (YY - Br**2) / d
            if np.isfinite(d0):
                sdr_post = np.sqrt((d0 * s02 + d * s2r) / (d0 + d))
            else:
                sdr_post = np.sqrt(s02)
            modtr = signc * Br / sdr_post

            rot_ranks = np.column_stack(
                [_rank_r(modtr), ngenes - _rank_r(modtr) + 1, _rank_r(np.abs(modtr))]
            )
            rot_set_ranks = _rowsum_mean(rot_ranks[all_indices, :])
            p_value += (rot_set_ranks >= obs_set_ranks).astype(np.float64)

    elif set_statistic == "floormean":
        obs_ranks = np.column_stack(
            [
                _rank_r(np.maximum(modt, 0)),
                _rank_r(np.maximum(-modt, 0)),
                _rank_r(np.maximum(np.abs(modt), 1)),
            ]
        )
        obs_set_ranks = _rowsum_mean(obs_ranks[all_indices, :])

        for _ in range(nrot):
            R_vec = rng_state.standard_normal((1, d + 1))
            R_vec = R_vec / np.sqrt(np.sum(R_vec**2))
            Br = (R_vec @ Y).ravel()
            s2r = (YY - Br**2) / d
            if np.isfinite(d0):
                sdr_post = np.sqrt((d0 * s02 + d * s2r) / (d0 + d))
            else:
                sdr_post = np.sqrt(s02)
            modtr = signc * Br / sdr_post

            rot_ranks = np.column_stack(
                [
                    _rank_r(np.maximum(modtr, 0)),
                    _rank_r(np.maximum(-modtr, 0)),
                    _rank_r(np.maximum(np.abs(modtr), 1)),
                ]
            )
            rot_set_ranks = _rowsum_mean(rot_ranks[all_indices, :])
            p_value += (rot_set_ranks >= obs_set_ranks).astype(np.float64)

    elif set_statistic == "mean50":
        s_r = _rank_r(modt)
        s_abs_r = _rank_r(np.abs(modt))

        m_half = np.floor((SetSizes + 1) / 2).astype(np.int64)

        s_rank_mixed = np.zeros(nsets)
        s_rank_up = np.zeros(nsets)
        s_rank_down = np.zeros(nsets)
        for i, nm in enumerate(set_names):
            iset = _resolve_set_index(index[nm], ngenes)
            mh = _mean_half(s_r[iset], m_half[i])
            s_rank_up[i] = mh[1]
            s_rank_down[i] = mh[0]
            s_rank_mixed[i] = _mean_half(s_abs_r[iset], m_half[i])[1]

        for _ in range(nrot):
            R_vec = rng_state.standard_normal((1, d + 1))
            R_vec = R_vec / np.sqrt(np.sum(R_vec**2))
            Br = (R_vec @ Y).ravel()
            s2r = (YY - Br**2) / d
            if np.isfinite(d0):
                sdr_post = np.sqrt((d0 * s02 + d * s2r) / (d0 + d))
            else:
                sdr_post = np.sqrt(s02)
            modtr = signc * Br / sdr_post

            s_r2 = _rank_r(modtr)
            s_abs_r2 = _rank_r(np.abs(modtr))

            for j, nm in enumerate(set_names):
                iset = _resolve_set_index(index[nm], ngenes)
                mh2 = _mean_half(s_r2[iset], m_half[j])
                s_up_2 = mh2[1]
                s_down_2 = mh2[0]
                s_mixed_2 = _mean_half(s_abs_r2[iset], m_half[j])[1]
                if s_up_2 >= s_rank_up[j]:
                    p_value[j, 0] += 1
                if s_down_2 <= s_rank_down[j]:
                    p_value[j, 1] += 1
                if s_mixed_2 >= s_rank_mixed[j]:
                    p_value[j, 2] += 1

    p_value = (p_value + 1) / (nrot + 1)
    out = pd.DataFrame(
        {
            "n_genes": SetSizes,
            "up": p_value[:, 0],
            "down": p_value[:, 1],
            "mixed": p_value[:, 2],
        },
        index=set_names,
    )
    return out


# ---------------------------------------------------------------------------
# gene_set_test / rank_sum_test_with_correlation
# ---------------------------------------------------------------------------


def rank_sum_test_with_correlation(
    index,
    statistics: np.ndarray,
    correlation: float = 0.0,
    df: float = np.inf,
) -> dict:
    """
    Wilcoxon rank-sum test with an inter-gene correlation adjustment.

    Port of R limma's ``rankSumTestWithCorrelation``.

    Returns
    -------
    dict
        ``{"less": p_lower_tail, "greater": p_upper_tail}`` matching R's
        named vector.
    """
    stats_arr = np.asarray(statistics, dtype=np.float64)
    n = stats_arr.size
    r = stats.rankdata(stats_arr, method="average")
    idx = np.asarray(index, dtype=np.int64)
    r1 = r[idx]
    n1 = r1.size
    n2 = n - n1
    U = n1 * n2 + n1 * (n1 + 1) / 2.0 - float(np.sum(r1))
    mu = n1 * n2 / 2.0

    if correlation == 0 or n1 == 1:
        sigma2 = n1 * n2 * (n + 1) / 12.0
    else:
        sigma2 = (
            np.arcsin(1.0) * n1 * n2
            + np.arcsin(0.5) * n1 * n2 * (n2 - 1)
            + np.arcsin(correlation / 2.0) * n1 * (n1 - 1) * n2 * (n2 - 1)
            + np.arcsin((correlation + 1) / 2.0) * n1 * (n1 - 1) * n2
        )
        sigma2 = sigma2 / 2.0 / np.pi

    ties = len(np.unique(r)) != n
    if ties:
        _, counts = np.unique(r, return_counts=True)
        counts = counts.astype(np.float64)
        adjustment = float(np.sum(counts * (counts + 1) * (counts - 1)) / (n * (n + 1) * (n - 1)))
        sigma2 = sigma2 * (1 - adjustment)

    zlower = (U + 0.5 - mu) / np.sqrt(sigma2)
    zupper = (U - 0.5 - mu) / np.sqrt(sigma2)

    if np.isinf(df):
        p_less = stats.norm.sf(zupper)
        p_greater = stats.norm.cdf(zlower)
    else:
        p_less = stats.t.sf(zupper, df=df)
        p_greater = stats.t.cdf(zlower, df=df)

    return {"less": float(p_less), "greater": float(p_greater)}


def gene_set_test(  # noqa: A002 (`type` kwarg shadows builtin; deliberate R parity)
    index,
    statistics: np.ndarray,
    alternative: str = "mixed",
    type: str = "auto",
    ranks_only: bool = True,
    nsim: int = 9999,
    rng=None,
) -> float:
    """
    Competitive gene-set test.

    Port of R limma's ``geneSetTest``. When ``ranks_only=True`` the
    p-value is obtained analytically via
    :func:`rank_sum_test_with_correlation`; otherwise a permutation
    distribution is simulated using ``rng``.

    The ``type`` keyword shadows the Python builtin `type`; this is
    deliberate for R-signature parity (limma's ``geneSetTest(type=...)``).
    The shadow is local to this function and the body never invokes
    ``type()``; callers keyword-passing ``type=...`` works as expected.
    """
    alt_map = {
        "two.sided": "either",
        "less": "down",
        "greater": "up",
    }
    if alternative not in ("mixed", "either", "down", "up", "less", "greater", "two.sided"):
        raise ValueError(
            f"alternative '{alternative}' not recognized. "
            "Must be one of 'mixed', 'either', 'down', 'up', "
            "'less', 'greater', 'two.sided'."
        )
    alternative = alt_map.get(alternative, alternative)

    type_lower = type.lower()
    if type_lower not in ("auto", "t", "f"):
        raise ValueError(f"type '{type}' not recognized. Must be 'auto', 't' or 'f'.")

    stats_arr = np.asarray(statistics, dtype=np.float64)
    allsamesign = bool(np.all(stats_arr >= 0) or np.all(stats_arr <= 0))
    if type_lower == "auto":
        type_lower = "f" if allsamesign else "t"
    if type_lower == "f" and alternative != "mixed":
        raise ValueError('Only alternative="mixed" is possible with F-like statistics.')
    if alternative == "mixed":
        stats_arr = np.abs(stats_arr)
    if alternative == "down":
        stats_arr = -stats_arr
        alternative = "up"

    idx = np.asarray(index, dtype=np.int64)

    if ranks_only:
        pvals = rank_sum_test_with_correlation(index=idx, statistics=stats_arr, df=np.inf)
        if alternative == "down":
            return float(pvals["less"])
        if alternative == "up":
            return float(pvals["greater"])
        if alternative == "either":
            return float(2.0 * min(pvals["less"], pvals["greater"]))
        # "mixed"
        return float(pvals["greater"])
    else:
        rng_state = _resolve_rng(rng)
        ssel = stats_arr[idx]
        ssel = ssel[~np.isnan(ssel)]
        nsel = ssel.size
        if nsel == 0:
            return 1.0
        stat_all = stats_arr[~np.isnan(stats_arr)]
        msel = float(np.mean(ssel))
        if alternative == "either":

            def posstat(x):
                return abs(x)
        else:

            def posstat(x):
                return x

        msel = posstat(msel)
        ntail = 1
        for _ in range(nsim):
            sample = rng_state.choice(stat_all, size=nsel, replace=False)
            if posstat(float(np.mean(sample))) >= msel:
                ntail += 1
        return float(ntail) / (nsim + 1)


def wilcox_gst(index, statistics, **kwargs) -> float:
    """
    Mean-rank gene-set test using a Wilcoxon / rank-sum test.

    Port of R limma's ``wilcoxGST`` (``geneset-wilcox.R``). Thin
    wrapper around :func:`gene_set_test` with ``ranks_only=True``.
    """
    kwargs.pop("ranks_only", None)
    return gene_set_test(index=index, statistics=statistics, ranks_only=True, **kwargs)


def top_romer(x, n: int = 10, alternative: str = "up"):
    """
    Extract the top gene sets from :func:`romer` output.

    Port of R limma's ``topRomer`` (``geneset-romer.R``). ``x`` is
    expected to be a DataFrame (or similar tabular object) with
    ``Up`` / ``Down`` / ``Mixed`` / ``NGenes`` columns, matching
    :func:`romer`'s return schema.
    """
    import pandas as pd

    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
    alternative = alternative.lower()
    if alternative not in ("up", "down", "mixed"):
        raise ValueError("alternative must be 'up', 'down', or 'mixed'")
    n = min(n, x.shape[0])
    # R's order() is stable; mergesort is the numpy equivalent.
    if alternative == "up":
        sort_keys = [x["Up"].values, x["Mixed"].values, -x["NGenes"].values]
    elif alternative == "down":
        sort_keys = [x["Down"].values, x["Mixed"].values, -x["NGenes"].values]
    else:  # mixed
        sort_keys = [
            x["Mixed"].values,
            np.minimum(x["Up"].values, x["Down"].values),
            -x["NGenes"].values,
        ]
    # lexsort sorts by last key primarily; invert order so first key has highest priority.
    order = np.lexsort(tuple(reversed(sort_keys)))
    return x.iloc[order[:n], :]

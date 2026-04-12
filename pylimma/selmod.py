# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   selmod.R  Copyright (C) 2008 Alicia Oshlack, Gordon Smyth
# Python port: Copyright (C) 2026 John Mulvey
"""
``select_model`` - AIC / BIC / Mallows' Cp model selection for gene-wise
linear models. Port of R limma's ``selectModel``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def select_model(
    y: np.ndarray,
    design_list: Sequence[np.ndarray] | dict,
    criterion: str = "aic",
    df_prior: float = 0,
    s2_prior: float | None = None,
    s2_true: np.ndarray | None = None,
    **lmfit_kwargs,
) -> dict:
    """
    Gene-wise model comparison via AIC, BIC, or Mallows' Cp.

    Port of R limma's ``selectModel(y, designlist, criterion, df.prior,
    s2.prior, s2.true, ...)`` (``selmod.R``).

    Parameters
    ----------
    y : ndarray or DataFrame
        Expression matrix, rows = genes, columns = arrays. No NaNs.
    design_list : sequence or dict of design matrices
        Collection of candidate design matrices. When a dict is
        supplied its keys become the model names (matching R's
        ``names(designlist)``).
    criterion : {"aic", "bic", "mallowscp"}, default "aic"
    df_prior : float, default 0
        Prior degrees of freedom for the variance estimate.
    s2_prior : float, optional
        Prior variance. Required when ``df_prior > 0``.
    s2_true : ndarray, optional
        True per-gene variance (required when
        ``criterion="mallowscp"``).
    **lmfit_kwargs : forwarded to :func:`lm_fit`.

    Returns
    -------
    dict
        ``IC``: ``DataFrame`` of per-model information criteria.
        ``pref``: ``pd.Categorical`` of preferred model per gene.
        ``criterion``: the criterion actually used.
    """
    from .lmfit import lm_fit

    y = np.asarray(y, dtype=np.float64)
    if np.any(np.isnan(y)):
        raise ValueError("NAs not allowed")
    narrays = y.shape[1]

    if isinstance(design_list, dict):
        model_names = list(design_list.keys())
        designs = list(design_list.values())
    else:
        designs = list(design_list)
        model_names = [str(i + 1) for i in range(len(designs))]
    nmodels = len(designs)

    if df_prior > 0 and s2_prior is None:
        raise ValueError("s2_prior must be set when df_prior > 0")
    if df_prior == 0:
        s2_prior = 0.0

    criterion = criterion.lower()
    if criterion not in ("aic", "bic", "mallowscp"):
        raise ValueError(
            "criterion must be one of 'aic', 'bic', 'mallowscp'"
        )

    IC = None
    if criterion == "mallowscp":
        if s2_true is None:
            raise ValueError("Need s2.true values")
        s2_true = np.asarray(s2_true, dtype=np.float64)
        for i, design in enumerate(designs):
            fit = lm_fit(y, design, **lmfit_kwargs)
            df_res = np.asarray(fit["df_residual"], dtype=np.float64)
            npar = narrays - float(df_res[0])
            if IC is None:
                IC = np.full((y.shape[0], nmodels), np.nan)
                if s2_true.size != y.shape[0] and s2_true.size != 1:
                    raise ValueError("s2_true wrong length")
            sigma = np.asarray(fit["sigma"], dtype=np.float64)
            IC[:, i] = df_res * sigma ** 2 / s2_true + npar * 2 - narrays
    else:
        ntotal = df_prior + narrays
        penalty = np.log(narrays) if criterion == "bic" else 2.0
        for i, design in enumerate(designs):
            fit = lm_fit(y, design, **lmfit_kwargs)
            df_res = np.asarray(fit["df_residual"], dtype=np.float64)
            sigma = np.asarray(fit["sigma"], dtype=np.float64)
            npar = narrays - float(df_res[0]) + 1
            s2_post = (df_prior * s2_prior + df_res * sigma ** 2) / ntotal
            if IC is None:
                IC = np.full((y.shape[0], nmodels), np.nan)
            IC[:, i] = ntotal * np.log(s2_post) + npar * penalty

    IC_df = pd.DataFrame(IC, columns=model_names)
    pref_idx = np.argmin(IC, axis=1)
    pref = pd.Categorical.from_codes(pref_idx, categories=model_names)
    return {"IC": IC_df, "pref": pref, "criterion": criterion}

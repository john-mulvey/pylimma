# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   auROC.R   Copyright (C) 2003-2020 Gordon Smyth
# Python port: Copyright (C) 2026 John Mulvey
"""
Area under ROC curve for empirical data.

Port of R limma's ``auROC`` (``prior_art/limma/R/auROC.R``).
"""
from __future__ import annotations

import numpy as np


def au_roc(truth, stat=None) -> float:
    """
    Area under the empirical ROC curve.

    Mirrors R's ``limma::auROC(truth, stat=NULL)``. ``truth`` is a
    logical / integer vector of test outcomes. When ``stat`` is
    supplied, cases are sorted by ``stat`` in decreasing order (ties
    resolved by averaging sensitivity). When ``stat`` is ``None`` the
    truth order is taken as the ranking.

    Returns NaN when ``truth`` contains any NA / NaN, when ``truth``
    is constant, or when ``stat`` contains any NA.
    """
    truth = np.asarray(truth)
    if np.any(pd_isna(truth)):
        return float("nan")
    ntests = truth.size
    truth_bool = truth.astype(bool)
    truth_int = truth_bool.astype(int)
    npos = int(truth_int.sum())
    if npos == 0 or npos == ntests:
        return float("nan")

    if stat is None:
        sensitivity = np.cumsum(truth_int) / npos
        return float(np.mean(sensitivity[~truth_bool]))

    stat = np.asarray(stat).ravel()
    if stat.size != ntests:
        raise ValueError("lengths differ")
    if np.any(np.isnan(stat.astype(float, copy=False))):
        return float("nan")

    # Decreasing-order sort, stable to match R's order(..., decreasing=TRUE).
    o = np.argsort(-stat, kind="stable")
    truth_int = truth_int[o]
    truth_bool = truth_bool[o]
    stat = stat[o]
    sensitivity = np.cumsum(truth_int) / npos

    # Replace sensitivity with averages across tied-stat runs.
    tie_to_prev = stat[:-1] == stat[1:]
    if np.any(tie_to_prev):
        iseq2prev = np.concatenate([[False], tie_to_prev])
        tied_first = np.where(~iseq2prev)[0]
        tied_last = np.concatenate([tied_first[1:] - 1, [ntests - 1]])
        sens_last = sensitivity[tied_last]
        sens_prev = np.concatenate([[0.0], sens_last[:-1]])
        sens_avg = (sens_last + sens_prev) / 2.0
        sensitivity = np.repeat(sens_avg, tied_last - tied_first + 1)

    return float(np.mean(sensitivity[~truth_bool]))


def pd_isna(a):
    """Stripped-down equivalent of ``pandas.isna`` for any array-like."""
    import pandas as pd
    return pd.isna(a)

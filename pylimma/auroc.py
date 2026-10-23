# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   auROC.R   Copyright (C) 2003-2020 Gordon Smyth
# Python port: Copyright (C) 2026 John Mulvey
"""
Area under ROC curve for empirical data.

Port of R limma's ``auROC`` (``limma/R/auROC.R``).
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
    is constant, or when ``stat`` contains any NA. As R's ``as.vector``,
    matrices are read column by column; ``truth`` is read as R's
    ``as.logical`` (numbers, or strings such as "TRUE" / "F"), and other
    strings raise, where R fails with "missing value where TRUE/FALSE
    needed". ``stat`` may be numbers, logicals or strings (ranked by code
    point; R uses the locale).
    """
    truth = np.asarray(truth).ravel(order="F")
    if np.any(pd_isna(truth)):
        return float("nan")
    ntests = truth.size
    truth_bool = _as_logical(truth)
    truth_int = truth_bool.astype(int)
    npos = int(truth_int.sum())
    if npos == 0 or npos == ntests:
        return float("nan")

    if stat is None:
        sensitivity = np.cumsum(truth_int) / npos
        return float(np.mean(sensitivity[~truth_bool]))

    stat = np.asarray(stat).ravel(order="F")
    if stat.size != ntests:
        raise ValueError("lengths differ")
    if np.any(pd_isna(stat)):
        return float("nan")

    # R: order(stat, decreasing=TRUE), ties in their original order; the
    # sort runs on the sorted unique values' codes, so logicals and strings
    # rank as R orders them
    _, codes = np.unique(stat, return_inverse=True)
    o = np.argsort(-codes, kind="stable")
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


_R_TRUE = {"TRUE", "true", "True", "T"}
_R_FALSE = {"FALSE", "false", "False", "F"}


def _as_logical(truth: np.ndarray) -> np.ndarray:
    """R's as.logical(truth): numbers are TRUE when non-zero and the strings
    R recognises are TRUE / FALSE; any other string raises."""
    if truth.dtype.kind not in ("U", "S", "O") or not any(isinstance(v, str) for v in truth):
        return truth.astype(bool)
    values = [str(v) for v in truth]
    unknown = [v for v in values if v not in _R_TRUE | _R_FALSE]
    if unknown:
        raise ValueError(
            f"truth must be logical, numeric or 'TRUE' / 'FALSE' strings, not {unknown[0]!r}"
        )
    return np.array([v in _R_TRUE for v in values])


def pd_isna(a):
    """Stripped-down equivalent of ``pandas.isna`` for any array-like."""
    import pandas as pd

    return pd.isna(a)

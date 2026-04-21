# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   fitmixture.R   Copyright (C) 2007 Gordon Smyth
# Python port: Copyright (C) 2026 John Mulvey
"""
``fitmixture`` - fit a two-sample mixture model via non-linear least
squares. Port of R limma's ``fitmixture``.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

from .utils import logcosh


def fitmixture(log2e, mixprop, niter: int = 4, trace: bool = False) -> dict:
    """
    Fit a mixture model by non-linear least squares.

    Port of ``limma/R/fitmixture.R``.

    Parameters
    ----------
    log2e : ndarray
        log2 expression matrix, shape (nprobes, narrays).
    mixprop : array-like of float, length narrays
        Mixture proportion of sample 1 in each array.
    niter : int, default 4
        Non-linear least squares iterations.
    trace : bool, default False
        If True, print stdev / beta summaries per iteration.

    Returns
    -------
    dict with keys ``A``, ``M``, ``stdev``.
    """
    log2e = np.asarray(log2e, dtype=np.float64)
    if log2e.ndim == 1:
        log2e = log2e.reshape(-1, 1)
    mixprop = np.asarray(mixprop, dtype=np.float64)
    narrays = log2e.shape[1]
    nprobes = log2e.shape[0]
    y = 2.0 ** log2e

    # Linear pre-fit: y ~ cbind(mixprop, 1 - mixprop) per-probe.
    X = np.column_stack([mixprop, 1 - mixprop])
    start, _, _, _ = linalg.lstsq(X, y.T)
    start = np.maximum(start, 1.0)
    # b = log(start[0]) - log(start[1]) per probe.
    b = np.log(start[0, :]) - np.log(start[1, :])

    z = np.log(y)
    pm = np.ones(nprobes)[:, None] @ (2 * mixprop - 1)[None, :]

    sold = None
    for i in range(niter):
        mub = logcosh(b / 2)[:, None] + np.log(1 + np.tanh(b / 2)[:, None] * pm)
        a = np.mean(z - mub, axis=1)
        mu = a[:, None] + mub
        if trace:
            s = np.sqrt(narrays / (narrays - 2) * np.mean((z - mu) ** 2, axis=1))
            if sold is not None:
                print("stdev changes", np.percentile(sold - s, [0, 25, 50, 75, 100]))
            sold = s
        tb = np.tanh(b / 2)[:, None]
        dmu = (tb + pm) / (1 + tb * pm) / 2
        numer = np.mean(dmu * (z - mu), axis=1)
        denom = 1e-6 + np.mean((dmu - np.mean(dmu, axis=1, keepdims=True)) ** 2, axis=1)
        b = b + numer / denom
        if trace:
            print("betas", np.percentile(b, [0, 25, 50, 75, 100]))

    mub = logcosh(b / 2)[:, None] + np.log(1 + np.tanh(b / 2)[:, None] * pm)
    a = np.mean(z - mub, axis=1)
    mu = a[:, None] + mub
    s = np.sqrt(narrays / (narrays - 2) * np.mean((z - mu) ** 2, axis=1))
    l2 = np.log(2.0)
    return {"A": a / l2, "M": b / l2, "stdev": s / l2}

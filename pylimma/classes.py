# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   classes.R     Copyright (C) 2003-2022 Gordon Smyth
#   subsetting.R  Copyright (C) 2013-2020 Gordon Smyth
#   lmfit.R       Copyright (C) 2008-2022 Gordon Smyth (getEAWP only)
# Python port: Copyright (C) 2026 John Mulvey
"""
Lightweight data classes and polymorphic input/output dispatchers for pylimma.

Exports
-------
EList           : input wrapper - dict subclass with E, weights, genes, targets, design.
MArrayLM        : fit-result wrapper - dict subclass holding linear-model fits.
get_eawp        : polymorphic input dispatcher, port of R limma's getEAWP().
put_eawp        : polymorphic output dispatcher. No R counterpart - see note below.

Why both classes subclass dict
------------------------------
Every key is reachable via both result["key"] and result.key. The dict-subclass
design preserves backward compatibility for any caller that treats results as
dicts, including all existing R-parity tests.

Why put_eawp exists when R has no counterpart
---------------------------------------------
R achieves input-class-preserving output through S3/S4 method dispatch on the
input class - normalizeBetweenArrays.EList vs .matrix vs .RGList are separate
methods, each knowing what class to return because it only ever sees one input
type. Python lacks that machinery. To get EList-in EList-out, ndarray-in
ndarray-out, and AnnData-in AnnData-mutate from a single Python function, we
need an explicit output-side dispatcher. Without put_eawp we would re-duplicate
the very pattern get_eawp was built to eliminate, just at the tail of every
shape-preserving function.
"""

from __future__ import annotations

import numbers
import warnings
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from .utils import _as_dense_array

# ----------------------------------------------------------------------------
# Weight-shape dispatcher (port of R limma's asMatrixWeights)
# ----------------------------------------------------------------------------


def _shape_error(name: str, actual, expected, message: str) -> ValueError:
    """Shape-mismatch error for a genes x samples matrix argument.

    pylimma follows limma's genes x samples orientation for explicit
    matrix arguments, the transpose of AnnData's samples x genes. When the
    argument is exactly the transpose of what was expected, say so.
    A square matrix (as many genes as samples) cannot be detected.
    """
    actual, expected = tuple(actual), tuple(expected)
    if len(actual) == 2 and actual != expected and actual == expected[::-1]:
        message = (
            f"{message}. {name} has shape {actual} but pylimma expects genes x samples "
            f"{expected}; it looks transposed (AnnData stores samples x genes). "
            f"Pass {name}.T."
        )
    return ValueError(message)


def as_matrix_weights(
    weights,
    dim: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Normalise a weight specification to a fresh ``(n_probes, n_arrays)``
    matrix.

    Port of R limma's private ``asMatrixWeights``
    (``limma/R/weights.R:57-88``). Accepts every shape R does:

    1. full ``(G, N)`` matrix - returned unchanged (copy).
    2. ``(1, N)`` row matrix with ``N == dim[1]`` - broadcast down rows.
    3. scalar / length-1 - broadcast to the full ``(G, N)``.
    4. length-G vector (**probe weights**) - broadcast across columns.
    5. length-N vector (**array weights**) - broadcast across rows.
    6. anything else - ``ValueError("weights is of unexpected size")``.

    The result is always freshly allocated so downstream ``weights[...] = ...``
    writes do not leak into the caller's memory (R's copy-on-modify
    semantics leave the caller's weights unchanged).

    R's branch order is preserved verbatim. In particular, if ``G == N``
    a length-G vector is treated as probe weights (branch 4), matching
    R's ambiguity resolution.

    R also tags the output with ``attr(weights, "arrayweights") <- TRUE``
    when it fills branches 2 or 5; ``_as_matrix_weights_with_flag``
    returns that flag, which ``lm_series`` / ``gls_series`` use to choose
    their fast path as R does.

    Parameters
    ----------
    weights : scalar, 1-D, or 2-D array-like
        Weight specification.
    dim : tuple of (int, int) or None
        Target shape ``(n_probes, n_arrays)``. When ``None`` the input
        is returned as a fresh 2-D matrix (column-matrix for 1-D input,
        matching R's ``as.matrix`` on a vector).

    Returns
    -------
    ndarray
        Matrix of shape ``dim`` (or the input's 2-D shape if ``dim`` is
        ``None``), always a fresh copy.
    """
    return _as_matrix_weights_with_flag(weights, dim)[0]


def _as_matrix_weights_with_flag(weights, dim=None) -> tuple[np.ndarray, bool]:
    """as_matrix_weights plus R's "arrayweights" attribute: True when the
    weights were array weights broadcast down the rows (branches 2 and 5)."""
    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        # Mirror R's as.matrix(vector): column matrix (K, 1).
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError("weights must be scalar, 1-D, or 2-D")

    if dim is None:
        return arr.copy(), False

    if len(dim) < 2:
        raise ValueError("dim must be a length-2 shape")
    target = (int(round(dim[0])), int(round(dim[1])))
    if target[0] < 1 or target[1] < 1:
        raise ValueError("zero or negative dimensions not allowed")

    dw = arr.shape

    # Full matrix already.
    if dw == target:
        return arr.copy(), False

    if min(dw) != 1:
        raise _shape_error("weights", dw, target, "weights is of unexpected shape")

    # Row matrix of array weights: (1, N) with N == target[1] > 1.
    if dw[1] > 1 and dw[1] == target[1]:
        return np.broadcast_to(arr, target).copy(), True

    lw = int(np.prod(dw))

    # Scalar / length-G probe weights. Broadcast column-wise.
    if lw == 1 or lw == target[0]:
        return np.broadcast_to(arr.reshape(-1, 1), target).copy(), False

    # Length-N array weights. Broadcast row-wise.
    if lw == target[1]:
        return np.broadcast_to(arr.reshape(1, -1), target).copy(), True

    raise ValueError("weights is of unexpected size")


# ----------------------------------------------------------------------------
# Slot-classification tables (port of subsetting.R:84-103 and :107-182)
# ----------------------------------------------------------------------------

_ELIST_IJ = ("E", "Eb", "weights")
_ELIST_IX = ("genes",)
_ELIST_JX = ("targets", "design")
_ELIST_I: tuple[str, ...] = ()

_MARRAYLM_IJ = (
    "coefficients",
    "stdev_unscaled",
    "t",
    "p_value",
    "lods",
    "weights",
)
_MARRAYLM_IX = ("genes",)
_MARRAYLM_JX: tuple[str, ...] = ()
_MARRAYLM_I = (
    "Amean",
    "sigma",
    "df_residual",
    "df_prior",
    "df_total",
    "s2_post",
    "F",
    "F_p_value",
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _is_anndata(obj: Any) -> bool:
    try:
        from anndata import AnnData
    except ImportError:
        return False
    return isinstance(obj, AnnData)


def _resolve_index(idx, names, n: int, dim: str):
    """Resolve a row/column selector to an integer array or slice.

    Accepts slices, logical masks (a single TRUE or FALSE selects every or
    no position, as R recycles it; other lengths must be ``n``), integer
    positions and names (matched against ``names``; R's
    subsetListOfArrays stops with "Subscript not found in rownames" /
    "colnames" for a name that is not there, or when there are no names).
    ``dim`` is "rownames" or "colnames". Returns a numpy index.
    """
    if idx is None:
        return slice(None)
    if isinstance(idx, slice):
        return idx
    arr = np.atleast_1d(np.asarray(idx))
    # a mask held in an object array (an object-dtype pandas column) is logical
    if arr.dtype.kind == "O" and arr.size and all(isinstance(v, (bool, np.bool_)) for v in arr):
        arr = arr.astype(bool)
    if arr.dtype == bool:
        if arr.size == 1:
            arr = np.repeat(arr, n)
        elif arr.size != n:
            what = "rows" if dim == "rownames" else "columns"
            raise ValueError(f"a logical subscript has {arr.size} values for {n} {what}")
        return np.where(arr)[0]
    if arr.dtype.kind in ("U", "S") or (
        arr.dtype.kind == "O" and any(isinstance(v, str) for v in arr)
    ):
        # R's match(): the first of repeated names
        position = {}
        for k, name in enumerate([] if names is None else names):
            position.setdefault(name, k)
        missing = [name for name in arr if name not in position]
        if missing:
            raise KeyError(f"Subscript not found in {dim}: {missing[0]!r}")
        return np.asarray([position[name] for name in arr], dtype=int)
    return arr.astype(int)


def _slice_2d(x, i, j):
    """Slice a 2D array-like by rows then columns, preserving 2D-ness."""
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        return (
            x.iloc[i, :].iloc[:, j]
            if not isinstance(j, slice) or j != slice(None)
            else x.iloc[i, :]
        )
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr[i]
    return (
        arr[np.ix_(np.atleast_1d(i), np.atleast_1d(j))]
        if not (isinstance(i, slice) or isinstance(j, slice))
        else arr[i, :][:, j]
    )


def _slice_rows(x, i):
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        return x.iloc[i]
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr[i]
    return arr[i, :] if isinstance(i, slice) else arr[np.atleast_1d(i), :]


def _slice_cols(x, j):
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        return x.iloc[j]
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr[j]
    return arr[j, :] if isinstance(j, slice) else arr[np.atleast_1d(j), :]


def _slice_1d(x, i):
    if x is None:
        return None
    arr = np.asarray(x)
    return arr[i] if isinstance(i, slice) else arr[np.atleast_1d(i)]


def _print_head(x, name: str, max_rows: int = 5) -> str:
    """R-style printHead - truncated display of a component."""
    if x is None:
        return f"${name}\nNULL\n"
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            n = x.shape[0]
            shown = x[: min(max_rows, n)]
            lines = [f"${name}", str(shown)]
            if n > max_rows:
                lines.append(f"{n - max_rows} more rows ...")
            return "\n".join(lines) + "\n"
        elif x.ndim == 1:
            n = x.shape[0]
            shown = x[: min(max_rows * 4, n)]
            lines = [f"${name}", str(shown)]
            if n > max_rows * 4:
                lines.append(f"{n - max_rows * 4} more elements ...")
            return "\n".join(lines) + "\n"
    if isinstance(x, pd.DataFrame):
        n = len(x)
        lines = [f"${name}", str(x.head(max_rows))]
        if n > max_rows:
            lines.append(f"{n - max_rows} more rows ...")
        return "\n".join(lines) + "\n"
    return f"${name}\n{x}\n"


# ----------------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------------


class _LargeDataObject(dict):
    """Shared behaviour for EList and MArrayLM.

    Subclasses must define:
        _matrix_key : str
            Name of the primary (genes x samples) matrix slot.
        _slot_ij    : tuple[str, ...]   - 2D slots sliced by both i and j
        _slot_ix    : tuple[str, ...]   - slots sliced by i only (row-aligned)
        _slot_jx    : tuple[str, ...]   - slots sliced by j only (col-aligned)
        _slot_i     : tuple[str, ...]   - 1D slots sliced by i (one per gene)
    """

    _matrix_key: str = ""
    _slot_ij: tuple[str, ...] = ()
    _slot_ix: tuple[str, ...] = ()
    _slot_jx: tuple[str, ...] = ()
    _slot_i: tuple[str, ...] = ()

    # ---- attribute access ----

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}") from e

    def __setattr__(self, name: str, value) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}") from e

    # ---- dimensions ----

    @property
    def shape(self) -> tuple[int, int]:
        m = self.get(self._matrix_key)
        if m is None:
            return (0, 0)
        arr = np.asarray(m) if not hasattr(m, "shape") else m
        if arr.ndim == 1:
            return (arr.shape[0], 1)
        return arr.shape[:2]

    def dim(self) -> tuple[int, int]:
        return self.shape

    @property
    def nrow(self) -> int:
        return self.shape[0]

    @property
    def ncol(self) -> int:
        return self.shape[1]

    def __len__(self) -> int:
        return dict.__len__(self)

    @property
    def dimnames(self) -> tuple[list | None, list | None]:
        m = self.get(self._matrix_key)
        if isinstance(m, pd.DataFrame):
            return (list(m.index), list(m.columns))
        genes = self.get("genes")
        targets = self.get("targets")
        row = None
        col = None
        if isinstance(genes, pd.DataFrame):
            row = list(genes.index)
        elif genes is not None and np.ndim(genes) == 1 and len(genes) == self.nrow:
            # row names stored as a bare list (lm_fit on AnnData or a named
            # DataFrame), R's rownames(fit)
            row = list(genes)
        if targets is not None and isinstance(targets, pd.DataFrame):
            col = list(targets.index)
        return (row, col)

    # ---- head / tail ----

    def _head_count(self, n) -> float:
        """R's head / tail n: a negative n leaves out -n rows."""
        if np.ndim(n) != 0:
            raise ValueError("n must be a single number")
        return max(self.nrow + n, 0) if n < 0 else min(n, self.nrow)

    def head(self, n: int = 6) -> "_LargeDataObject":
        # R: seq_len(n) truncates a fraction
        return self._subset(slice(0, int(self._head_count(n))), slice(None))

    def tail(self, n: int = 6) -> "_LargeDataObject":
        # R: seq.int(to = nrow, length.out = n) rounds a fraction up
        count = int(np.ceil(self._head_count(n)))
        return self._subset(slice(self.nrow - count, self.nrow), slice(None))

    # ---- subsetting ----

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            return self._subset(key[0], key[1])
        return dict.__getitem__(self, key)

    def __setitem__(self, key, value) -> None:
        if isinstance(key, tuple):
            raise TypeError("Tuple subscripts are read-only on " + type(self).__name__)
        dict.__setitem__(self, key, value)

    def _subset(self, i, j) -> "_LargeDataObject":
        row_names, col_names = self.dimnames
        i_idx = _resolve_index(i, row_names, self.nrow, "rownames")
        j_idx = _resolve_index(j, col_names, self.ncol, "colnames")

        out = deepcopy(self)

        for k in self._slot_ij:
            v = out.get(k)
            if v is not None:
                out[k] = _slice_2d(v, i_idx, j_idx)
        for k in self._slot_ix:
            v = out.get(k)
            if v is not None:
                out[k] = _slice_rows(v, i_idx)
        for k in self._slot_jx:
            v = out.get(k)
            if v is not None:
                out[k] = _slice_rows(v, j_idx)
        for k in self._slot_i:
            v = out.get(k)
            if v is not None and np.asarray(v).size > 1:
                out[k] = _slice_1d(v, i_idx)

        return out

    # ---- repr ----

    def __repr__(self) -> str:
        lines = [f'An object of class "{type(self).__name__}"']
        for name, value in self.items():
            lines.append(_print_head(value, name))
        return "\n".join(lines).rstrip() + "\n"


# ----------------------------------------------------------------------------
# Public classes
# ----------------------------------------------------------------------------


class EList(_LargeDataObject):
    """Expression-list container (Python equivalent of R limma's EList).

    Required slot: E (2D numeric, genes x samples).
    Optional slots: weights, genes, targets, design, plus arbitrary extras.

    Examples
    --------
    >>> el = EList({"E": np.random.randn(100, 8), "design": np.eye(8)})
    >>> el.shape
    (100, 8)
    >>> el[:10, :].shape
    (10, 8)
    """

    _matrix_key = "E"
    _slot_ij = _ELIST_IJ
    _slot_ix = _ELIST_IX
    _slot_jx = _ELIST_JX
    _slot_i = _ELIST_I

    def __init__(self, data=None, /, **kwargs):
        if data is None:
            data = {}
        if isinstance(data, EList):
            super().__init__(dict(data))
        elif isinstance(data, dict):
            super().__init__(data)
        else:
            raise TypeError(f"EList expected dict or EList, got {type(data).__name__}")
        for k, v in kwargs.items():
            self[k] = v


class MArrayLM(_LargeDataObject):
    """Linear-model-fit container (Python equivalent of R limma's MArrayLM).

    Holds the output of lm_fit / contrasts_fit / e_bayes / treat.
    """

    _matrix_key = "coefficients"
    _slot_ij = _MARRAYLM_IJ
    _slot_ix = _MARRAYLM_IX
    _slot_jx = _MARRAYLM_JX
    _slot_i = _MARRAYLM_I

    def __init__(self, data=None, /, **kwargs):
        if data is None:
            data = {}
        if isinstance(data, MArrayLM):
            super().__init__(dict(data))
        elif isinstance(data, dict):
            super().__init__(data)
        else:
            raise TypeError(f"MArrayLM expected dict or MArrayLM, got {type(data).__name__}")
        for k, v in kwargs.items():
            self[k] = v

    def _subset(self, i, j) -> "MArrayLM":
        """R's ``[.MArrayLM`` (subsetting.R).

        ``fit[i, :]`` subsets genes only. Selecting columns (anything other
        than ``:``) also subsets cov_coefficients, in the order of the
        selected columns (R keeps pivot order for a rank-deficient design,
        so its covariance does not line up with the coefficients; see
        :doc:`/validation/known_differences`), creates an
        identity ``contrasts`` matrix if none is present and subsets it,
        subsets var_prior, and regenerates F and F_p_value from the kept
        columns.
        """
        if isinstance(j, slice) and j == slice(None):
            return super()._subset(i, j)

        names = self.get("contrast_names") or self.get("coef_names")
        ncoef = self.ncol
        j_idx = np.arange(ncoef)[_resolve_index(j, names, ncoef, "colnames")]

        # cov_coefficients covers only estimable coefficients (NaN rows here,
        # fewer columns in R); selecting a non-estimable coefficient is an error
        cov = self.get("cov_coefficients")
        if cov is not None:
            estimable = ~np.isnan(np.diag(np.asarray(cov, dtype=np.float64)))
            if not estimable.all():
                if self.get("pivot") is None:
                    raise ValueError("design matrix not of full rank but pivot is missing")
                pivot = np.asarray(self["pivot"])
                if np.isin(j_idx, pivot[estimable.sum() :]).any():
                    raise ValueError("Subsetting to non-estimable coefficients is not allowed.")

        fit = MArrayLM(dict(self))
        if fit.get("contrasts") is None:
            fit["contrasts"] = np.eye(ncoef)
            if names is not None:
                fit["contrast_names"] = list(names)
        out = super(MArrayLM, fit)._subset(i, j_idx)

        if cov is not None:
            out["cov_coefficients"] = np.asarray(cov, dtype=np.float64)[np.ix_(j_idx, j_idx)]
        out["contrasts"] = np.asarray(out["contrasts"])[:, j_idx]
        if out.get("contrast_names") is not None:
            out["contrast_names"] = [out["contrast_names"][k] for k in j_idx]
        if out.get("var_prior") is not None:
            out["var_prior"] = np.atleast_1d(out["var_prior"])[j_idx]

        if out.get("F") is not None:
            if j_idx.size:
                from scipy import stats

                from .decide_tests import classify_tests_f

                f_stat, df1, df2 = classify_tests_f(out, fstat_only=True)
                out["F"] = f_stat
                if np.ravel(df2)[0] > 1e6:
                    out["F_p_value"] = stats.chi2.sf(df1 * f_stat, df1)
                else:
                    out["F_p_value"] = stats.f.sf(f_stat, df1, df2)
            else:
                out.pop("F", None)
                out.pop("F_p_value", None)
        return out

    # ---- R-parity convenience methods (lmfit.R, classes.R) ----

    def fitted(self) -> np.ndarray:
        """
        Fitted values ``coefficients @ design.T``. Port of R limma's
        ``fitted.MArrayLM`` (``lmfit.R``). Raises when the fit holds
        contrasts rather than raw coefficients.
        """
        if self.get("contrasts") is not None:
            raise ValueError(
                "Object contains contrasts rather than coefficients, "
                "so fitted values cannot be computed."
            )
        design = self.get("design")
        coef = self.get("coefficients")
        if design is None or coef is None:
            raise ValueError("Fit must contain coefficients and design")
        return np.asarray(coef) @ np.asarray(design).T

    def residuals(self, y) -> np.ndarray:
        """
        Residuals ``y - fitted``. Port of R limma's
        ``residuals.MArrayLM`` (``lmfit.R``).
        """
        return np.asarray(y) - self.fitted()

    def as_dataframe(self, row_names=None) -> pd.DataFrame:
        """
        Flatten the fit into a ``pandas.DataFrame`` with one row per
        probe. Port of R limma's ``as.data.frame.MArrayLM``
        (``classes.R``). Only slots whose first dimension matches the
        number of probes are retained.
        """
        if self.get("coefficients") is None:
            import warnings as _w

            _w.warn("NULL coefficients, returning empty data.frame")
            return pd.DataFrame()
        coef = np.asarray(self["coefficients"])
        if coef.ndim == 1:
            coef = coef.reshape(-1, 1)
        nprobes, ncoef = coef.shape
        columns: dict[str, np.ndarray] = {}
        for name, value in self.items():
            arr = np.asarray(value) if value is not None else None
            if arr is None:
                continue
            if arr.ndim == 0 or arr.size == 0:
                continue
            if arr.shape[0] != nprobes:
                continue
            if arr.ndim == 1:
                columns[name] = arr
            else:
                for j in range(arr.shape[1]):
                    col_name = f"{name}.{j + 1}" if arr.shape[1] > 1 else name
                    columns[col_name] = arr[:, j]
        return pd.DataFrame(columns, index=row_names)


# ----------------------------------------------------------------------------
# Polymorphic input dispatcher
# ----------------------------------------------------------------------------

_UNSUPPORTED_WRAPPER_NAMES = {
    "RGList",
    "MAList",
    "EListRaw",
    "ExpressionSet",
    "eSet",
    "PLMset",
    "marrayNorm",
    "DGEList",
}


def get_eawp(
    obj,
    *,
    layer: str | None = None,
    weights_layer: str | None = None,
) -> dict:
    """Polymorphic input dispatcher - port of R limma's getEAWP().

    Accepts a variety of expression-data containers and returns a plain dict
    with R-parity keys. The returned dict is consumed by linear-modelling
    functions; it is deliberately not wrapped in an EList.

    Parameters
    ----------
    obj : ndarray, DataFrame, dict, EList, or AnnData
        Expression data. Dict-with-'E' is accepted for backwards compatibility
        and is treated identically to an EList. scipy.sparse, backed, lazy
        and dask-held matrices are densified (R's as.matrix()); masked cells
        of a numpy masked array are treated as missing values (NaN).
    layer : str, optional
        AnnData-only. Name of an adata.layers[...] entry to use in place of X.
    weights_layer : str, optional
        AnnData-only. Name of an adata.layers[...] entry to read as
        observation weights. When ``None`` (default) and ``layer`` ends in
        ``"_E"``, the companion layer ``{layer[:-2]}_weights`` is auto-loaded
        if it exists (the voom/vooma convention); when set explicitly, the
        named layer is read instead. Use this when you've customised
        :func:`voom`'s ``weights_layer=`` output name on the write side and
        need the read side to match.

    Returns
    -------
    dict
        With keys: exprs (numpy 2D, genes x samples), weights, probes, Amean,
        design, targets. Missing components are None.

    Raises
    ------
    TypeError
        If obj is of an unsupported class. Two-channel microarray wrappers
        (RGList, MAList, EListRaw) and Bioconductor S4 containers
        (ExpressionSet, eSet, PLMset, marrayNorm) are deliberately out of
        scope; extract the expression matrix instead.
    """
    if obj is None:
        raise TypeError("data object is None")

    y: dict[str, Any] = {
        "exprs": None,
        "weights": None,
        "probes": None,
        "Amean": None,
        "design": None,
        "targets": None,
    }

    if layer is not None and not _is_anndata(obj):
        raise TypeError(
            f"layer={layer!r} is only supported for AnnData input; "
            f"got {type(obj).__name__}. For EList / dict / ndarray / "
            "DataFrame inputs, pass the data directly from the slot "
            "you intended."
        )
    if weights_layer is not None and not _is_anndata(obj):
        raise TypeError(
            f"weights_layer={weights_layer!r} is only supported for "
            f"AnnData input; got {type(obj).__name__}. For EList / dict / "
            "ndarray / DataFrame inputs, pass weights= directly."
        )

    if _is_anndata(obj):
        # X and layers may be scipy.sparse, backed, lazy or dask-held.
        adata = obj
        if layer is not None:
            mat = adata.layers[layer]
        else:
            mat = adata.X
            if mat is None:
                raise ValueError("adata.X is None: there are no expression values (pass layer=)")
        y["exprs"] = _expression_values(mat, vector="row").T
        # Capture adata.var / adata.obs as full DataFrames even when they
        # have zero annotation columns - the index still carries
        # var_names / obs_names, and consumers (top_table genelist,
        # fit['targets'], diagnostic plots) care about that index.
        # Mirrors the DataFrame branch's non-RangeIndex capture logic.
        if adata.var is not None:
            y["probes"] = adata.var
        if adata.obs is not None:
            y["targets"] = adata.obs
        y["Amean"] = _row_means(y["exprs"])
        # Load companion weights. If the caller passed weights_layer=
        # explicitly, use that layer verbatim. Otherwise fall back to
        # the voom/vooma convention: when layer ends in "_E", look for
        # the companion layer {stem}_weights. AnnData layers are stored
        # in n_samples x n_genes orientation; transpose to limma's
        # n_genes x n_samples. Also pick up the design that voom/vooma
        # stashed at adata.uns[stem]["design"] so downstream consumers
        # (notably lm_fit) see it through the same eawp["design"] slot
        # that EList input populates, matching R's
        #   if(is.null(design)) design <- y$design
        # one-liner.
        if weights_layer is not None:
            if weights_layer not in adata.layers:
                raise KeyError(
                    f"weights_layer={weights_layer!r} not found in "
                    f"adata.layers (available: {list(adata.layers)})"
                )
            y["weights"] = _as_dense_array(adata.layers[weights_layer], np.float64).T
        if layer is not None and layer.endswith("_E"):
            stem = layer[:-2]
            if weights_layer is None:
                weights_candidate = f"{stem}_weights"
                if weights_candidate in adata.layers:
                    y["weights"] = _as_dense_array(adata.layers[weights_candidate], np.float64).T
            uns_payload = adata.uns.get(stem)
            if isinstance(uns_payload, dict):
                if uns_payload.get("design") is not None:
                    y["design"] = _stored_design_for_obs(uns_payload, adata, stem)
        return y

    if isinstance(obj, EList):
        return _eawp_from_elist_like(obj)

    cls_name = type(obj).__name__
    if cls_name in _UNSUPPORTED_WRAPPER_NAMES:
        raise TypeError(
            f"{cls_name} is not supported by pylimma. "
            "Two-channel and Bioconductor S4 wrappers are out of scope. "
            "Extract the expression matrix "
            "and pass it directly, or wrap it in pylimma.EList."
        )

    if isinstance(obj, dict):
        if "E" in obj:
            return _eawp_from_elist_like(obj)
        if "exprs" in obj:
            y.update({k: obj.get(k) for k in y.keys() if k in obj})
            if y["exprs"] is not None:
                y["exprs"] = _expression_values(y["exprs"], vector="row")
                if y["Amean"] is None:
                    y["Amean"] = _row_means(y["exprs"])
            return y
        raise TypeError(
            "dict input to get_eawp must contain key 'E' (EList-style) "
            "or 'exprs' (getEAWP-output-style)"
        )

    if isinstance(obj, pd.DataFrame):
        # R lmFit: vapply(object, is.numeric); logical columns are not numeric.
        # pandas' check also covers nullable extension dtypes (Float64, Int64).
        numeric_cols = obj.dtypes.apply(
            lambda dt: pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_bool_dtype(dt)
        )
        if numeric_cols.all():
            y["exprs"] = obj.to_numpy(dtype=np.float64, na_value=np.nan)
        elif (~numeric_cols).sum() == 1 and not numeric_cols.iloc[0] and len(obj.columns) > 1:
            # R: identical(sum(WhichNotNumeric), 1L) - the one non-numeric
            # column must be the first; it becomes the gene IDs.
            y["exprs"] = obj.iloc[:, 1:].to_numpy(dtype=np.float64, na_value=np.nan)
            y["probes"] = obj.iloc[:, [0]]
        else:
            raise TypeError(
                "Expression object should be numeric, instead it is a data.frame with "
                f"{int((~numeric_cols).sum())} non-numeric columns (a single non-numeric "
                "first column is treated as gene IDs)"
            )
        # Mirror R's getEAWP: if the input has non-default row names,
        # wrap them as a one-column probes DataFrame so gene names
        # propagate into lm_fit / top_table. Skip when the index is
        # the default RangeIndex (no meaningful names set).
        if y["probes"] is None and not isinstance(obj.index, pd.RangeIndex):
            y["probes"] = pd.DataFrame(
                {"ProbeID": obj.index.astype(str).values},
                index=obj.index,
            )
        y["Amean"] = _row_means(y["exprs"])
        return y

    y["exprs"] = _expression_values(obj, vector="row")
    y["Amean"] = _row_means(y["exprs"])
    return y


def _expression_values(values, vector: str) -> np.ndarray:
    """An expression matrix as getEAWP reads it: dense float64, a vector as
    one row (vector="row", R's matrix(object, nrow=1)) or one column
    (vector="column", R's as.matrix of an EList's E); not numeric (or
    complex) raises R's "Data object doesn't contain numeric expression
    values", and None or more than two dimensions a clear error."""
    if values is None:
        raise ValueError("data object isn't of a recognized data class: no expression values")
    arr = _as_dense_array(values)
    if arr.dtype == object:
        # numbers held as objects, as pandas' nullable dtypes give them
        flat = arr.ravel()
        missing = pd.isna(flat)
        if all(
            isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))
            for v in flat[~missing]
        ):
            arr = np.where(missing, np.nan, flat).astype(np.float64).reshape(arr.shape)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
        raise TypeError("Data object doesn't contain numeric expression values")
    if arr.ndim == 1:
        arr = arr.reshape((1, -1) if vector == "row" else (-1, 1))
    if arr.ndim != 2:
        raise TypeError(f"expression data must be 2-dimensional, got {arr.ndim}D")
    return arr.astype(np.float64, copy=False)


def _row_means(exprs: np.ndarray) -> np.ndarray:
    """R's rowMeans(exprs, na.rm=TRUE): NaN for a row with no observed value,
    without numpy's "Mean of empty slice" warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(exprs, axis=1)


def _stored_design_for_obs(uns_payload: dict, adata, stem: str):
    """The design voom/vooma stored in adata.uns[stem], with its rows in the
    order of adata.obs_names.

    anndata copies adata.uns unchanged when samples are subset or reordered;
    R's v[, idx] subsets the design rows with the samples. put_eawp records
    the obs_names the design was written for, so its rows can follow them.
    """
    design = uns_payload["design"]
    stored = uns_payload.get("obs_names")
    current = [str(name) for name in adata.obs_names]
    if stored is None or [str(name) for name in stored] == current:
        return design
    stored = [str(name) for name in stored]
    if len(set(stored)) < len(stored) or len(set(current)) < len(current):
        raise ValueError(
            f"sample names are not unique, so the design stored in adata.uns[{stem!r}] "
            "cannot be matched to the samples after a subset or reorder; pass "
            "design= explicitly."
        )
    row_of = {name: i for i, name in enumerate(stored)}
    missing = [name for name in current if name not in row_of]
    if missing:
        raise ValueError(
            f"adata.obs_names include samples that are not in the design stored in "
            f"adata.uns[{stem!r}] (e.g. {missing[0]!r}; were samples renamed?); "
            "pass design= explicitly."
        )
    rows = [row_of[name] for name in current]
    if isinstance(design, pd.DataFrame):
        return design.iloc[rows]
    return np.asarray(design)[rows]


def _eawp_from_elist_like(obj) -> dict:
    """Extract EAWP dict from an EList or EList-shaped dict."""
    exprs = _expression_values(obj["E"], vector="column")
    return {
        "exprs": exprs,
        "weights": obj.get("weights"),
        "probes": obj.get("genes"),
        "Amean": _row_means(exprs),
        "design": obj.get("design"),
        "targets": obj.get("targets"),
    }


def _elist_from_layer(y, layer: str | None = None, weights_layer: str | None = None):
    """Resolve an AnnData layer selection into an EList at a function's entry.

    For functions whose internals call ``get_eawp(y)`` more than once
    (the gene-set tests), resolving ``layer`` / ``weights_layer`` once up
    front lets those internals stay unchanged. When both are None the
    input is returned untouched; otherwise the result carries the same
    exprs / weights / probes / targets / design that ``get_eawp`` reads,
    so the voom companion-weights and design pickup behave as for an
    EList produced by voom in R.
    """
    if layer is None and weights_layer is None:
        return y
    eawp = get_eawp(y, layer=layer, weights_layer=weights_layer)
    slots = {
        "E": eawp["exprs"],
        "weights": eawp["weights"],
        "genes": eawp["probes"],
        "targets": eawp["targets"],
        "design": eawp["design"],
    }
    return EList({k: v for k, v in slots.items() if v is not None})


# ----------------------------------------------------------------------------
# Polymorphic output dispatcher
# ----------------------------------------------------------------------------


def put_eawp(
    slots: dict,
    original,
    *,
    out_layer: str = "pylimma_E",
    weights_layer: str | None = "pylimma_weights",
    uns_key: str = "pylimma",
    single_matrix: bool = False,
):
    """Polymorphic output dispatcher - package a result in a form matching
    the original input's class.

    This has no direct R counterpart; see module docstring for rationale.

    Parameters
    ----------
    slots : dict
        Updated slot values. Typically {'E': ..., 'weights': ..., 'design':
        ..., ...}. 'E' is required.
    original : the object originally passed in by the user
        Determines the output format.
    out_layer : str
        For AnnData input, the layer name to which the new E is written.
    weights_layer : str or None
        For AnnData input, the layer name for weights. If None, weights are
        placed inside adata.uns[uns_key] instead.
    uns_key : str
        For AnnData input, the adata.uns[...] key for ancillary metadata
        (design, lib_size, span, targets, etc.).
    single_matrix : bool, default False
        When True (appropriate for functions whose only meaningful output is
        the transformed expression matrix, e.g. normalize_between_arrays),
        ndarray input returns a bare ndarray. When False (the default, for
        functions with multi-slot output like voom), ndarray input returns
        a plain dict containing all slots - matching the pre-existing
        convention.

    Returns
    -------
    np.ndarray, dict, EList, or None
        - ndarray when original is ndarray and single_matrix=True.
        - dict when original is ndarray (default) or a plain dict.
        - EList when original is an EList.
        - None when original is AnnData (side-effects on adata).

    Notes
    -----
    **AnnData view semantics.** If ``original`` is a view
    (``adata[:, mask]``), writing to ``adata.layers`` / ``adata.uns``
    triggers anndata's ``ImplicitModificationWarning`` and actualises
    the view into a standalone copy. The write lands on that copy;
    the parent AnnData is untouched. This matches scanpy's convention
    and is not a pylimma-specific behaviour. To operate in place on a
    subset, call ``adata[:, mask].copy()`` first or subset the parent
    explicitly. A bare one-liner like ``pylimma.voom(adata[:, mask])``
    discards the actualised copy along with the return value; bind the
    view first (``view = adata[:, mask]; pylimma.voom(view)``) to keep
    the result reachable.
    """
    if "E" not in slots:
        raise ValueError("put_eawp: slots must contain 'E'")

    if _is_anndata(original):
        adata = original
        E = np.asarray(slots["E"])
        adata.layers[out_layer] = E.T if E.ndim == 2 else E
        if slots.get("weights") is not None and weights_layer is not None:
            # Normalise weights to a (n_genes, n_samples) matrix before
            # writing. Covers scalar / length-G probe / length-N array /
            # (1, N) row / full (G, N) shapes via as_matrix_weights;
            # without this a 1-D caller would write a 1-D layer and
            # anndata would reject it. Today's public callers (voom,
            # vooma, voom_with_quality_weights) all produce 2-D weights
            # so this is a defensive fix for future callers.
            W = as_matrix_weights(slots["weights"], E.shape)
            adata.layers[weights_layer] = W.T
        uns_payload = {
            k: v
            for k, v in slots.items()
            if k not in ("E",)
            and v is not None
            and not (k == "weights" and weights_layer is not None)
        }
        if "design" in uns_payload:
            # The samples the design rows belong to (see _stored_design_for_obs).
            uns_payload["obs_names"] = [str(name) for name in adata.obs_names]
        if uns_payload:
            adata.uns[uns_key] = uns_payload
        return None

    if isinstance(original, EList):
        out = EList(dict(original))
        for k, v in slots.items():
            if v is not None:
                out[k] = v
        return out

    if isinstance(original, dict):
        out = dict(original)
        for k, v in slots.items():
            if v is not None:
                out[k] = v
        return out

    # ndarray (or anything else coerced to array)
    if single_matrix:
        return np.asarray(slots["E"])
    return {k: v for k, v in slots.items() if v is not None}


# ----------------------------------------------------------------------------
# Helper for fit-consuming functions (e_bayes, contrasts_fit, top_table, ...)
# ----------------------------------------------------------------------------


def _resolve_fit_input(data, key: str):
    """Resolve the fit input for functions that consume an existing fit.

    Returns
    -------
    (fit, adata, adata_key)
        fit: the MArrayLM / dict to operate on.
        adata: the AnnData if input was AnnData, else None.
        adata_key: the uns[] key, if input was AnnData, else None.

    When adata is not None, the caller must write the result back to
    adata.uns[adata_key] and return None.
    """
    if _is_anndata(data):
        if key not in data.uns:
            raise ValueError(
                f"No fit results found in adata.uns[{key!r}]. Did you run lm_fit() first?"
            )
        fit = data.uns[key]
        if not _var_matches_fit(fit, data):
            raise ValueError(
                f"adata.var_names do not match the genes of the fit in adata.uns[{key!r}]: "
                "genes were subset, reordered or renamed after fitting. Re-run "
                "lm_fit on this object."
            )
        return _with_list_names(fit), data, key
    return _with_list_names(data), None, None


def _fit_var(fit, adata) -> pd.DataFrame:
    """adata.var rows for the fit's rows: reduced by R's uniquegenelist when
    the fit has ndups > 1, as lmFit reduces y$probes."""
    from .dups import unique_genelist  # dups imports this module

    return unique_genelist(
        adata.var, ndups=int(fit.get("ndups") or 1), spacing=int(fit.get("spacing") or 1)
    )


def _var_matches_fit(fit, adata) -> bool:
    """True when the fit's rows are adata's genes, in order.

    anndata copies adata.uns unchanged when genes are subset, reordered or
    renamed, so a stored fit can outlive the var axis it was computed on.
    lm_fit records the var_names of _fit_var as fit["genes"]; a fit without
    genes is checked by its row count.
    """
    expected = [str(g) for g in _fit_var(fit, adata).index]
    genes = fit.get("genes")
    if genes is None:
        return np.shape(fit["coefficients"])[0] == len(expected)
    names = genes.index if isinstance(genes, pd.DataFrame) else genes
    return [str(g) for g in names] == expected


def _with_list_names(fit):
    """Undo h5ad's list -> ndarray conversion of the name slots.

    write_h5ad / read_h5ad turns list-valued slots into ndarrays, and name
    lookups (.index, `or`) expect lists. Applied to every fit input, not
    only AnnData, so a fit taken out of a reloaded adata.uns by hand also
    works. Returns a shallow copy when a slot needs converting, so the
    caller's object is left untouched.
    """
    if not isinstance(fit, dict):
        return fit
    stale = [s for s in ("coef_names", "contrast_names") if isinstance(fit.get(s), np.ndarray)]
    if not stale:
        return fit
    fit = type(fit)(fit)
    for slot in stale:
        fit[slot] = fit[slot].astype(str).tolist()
    return fit


def get_fit(adata, key: str = "pylimma") -> MArrayLM:
    """Extract a fit stored in ``adata.uns[key]`` as a standalone MArrayLM.

    pylimma stores AnnData fits as plain dicts (for h5ad compatibility).
    This helper returns an independent :class:`MArrayLM` with the name
    slots restored to lists (h5ad stores them as arrays).

    The fit has no ``targets`` slot: ``adata.obs`` may have been subset,
    reordered or edited since the fit was made, so it cannot be
    guaranteed to describe the fitted samples. Read ``adata.obs``
    directly if you need the sample metadata.

    Parameters
    ----------
    adata : AnnData
        Object holding a fit from :func:`lm_fit` / :func:`e_bayes` /
        :func:`contrasts_fit` / :func:`treat`.
    key : str, default "pylimma"
        The ``adata.uns`` entry holding the fit.

    Returns
    -------
    MArrayLM
        A deep copy: editing it does not change ``adata.uns[key]``.
    """
    if not _is_anndata(adata):
        raise TypeError(f"get_fit expects an AnnData object; got {type(adata).__name__}")
    fit, _, _ = _resolve_fit_input(adata, key)
    return MArrayLM(deepcopy(dict(fit)))

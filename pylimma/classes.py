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

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Weight-shape dispatcher (port of R limma's asMatrixWeights)
# ----------------------------------------------------------------------------


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
    writes do not leak into the caller's memory (see
    ``known_diff_weights_mutation.md``).

    R's branch order is preserved verbatim. In particular, if ``G == N``
    a length-G vector is treated as probe weights (branch 4), matching
    R's ambiguity resolution.

    R also tags the output with ``attr(weights, "arrayweights") <- TRUE``
    when it fills branches 2 or 5. pylimma does not propagate that
    attribute because the ``lm_series`` / ``mrlm`` / ``gls_series`` fast-
    path dispatchers value-check ``weights[0] == weights[i]`` at runtime
    instead (see ``lmfit.py`` ``has_probe_weights`` branch).

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
    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        # Mirror R's as.matrix(vector): column matrix (K, 1).
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError("weights must be scalar, 1-D, or 2-D")

    if dim is None:
        return arr.copy()

    if len(dim) < 2:
        raise ValueError("dim must be a length-2 shape")
    target = (int(round(dim[0])), int(round(dim[1])))
    if target[0] < 1 or target[1] < 1:
        raise ValueError("zero or negative dimensions not allowed")

    dw = arr.shape

    # Full matrix already.
    if dw == target:
        return arr.copy()

    if min(dw) != 1:
        raise ValueError("weights is of unexpected shape")

    # Row matrix of array weights: (1, N) with N == target[1] > 1.
    if dw[1] > 1 and dw[1] == target[1]:
        return np.broadcast_to(arr, target).copy()

    lw = int(np.prod(dw))

    # Scalar / length-G probe weights. Broadcast column-wise.
    if lw == 1 or lw == target[0]:
        return np.broadcast_to(arr.reshape(-1, 1), target).copy()

    # Length-N array weights. Broadcast row-wise.
    if lw == target[1]:
        return np.broadcast_to(arr.reshape(1, -1), target).copy()

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


def _resolve_index(idx, names):
    """Resolve a row/column selector to an integer array or slice.

    Accepts bool masks, int arrays, string lists (resolved against names),
    or slices. Returns a form usable as a numpy index.
    """
    if idx is None:
        return slice(None)
    if isinstance(idx, slice):
        return idx
    arr = np.atleast_1d(np.asarray(idx))
    if arr.dtype == bool:
        return np.where(arr)[0]
    if arr.dtype.kind in ("U", "S", "O") and names is not None:
        names_list = list(names)
        out = []
        for name in arr:
            if name not in names_list:
                raise KeyError(f"Subscript not found: {name!r}")
            out.append(names_list.index(name))
        return np.asarray(out, dtype=int)
    return arr.astype(int)


def _slice_2d(x, i, j):
    """Slice a 2D array-like by rows then columns, preserving 2D-ness."""
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        return x.iloc[i, :].iloc[:, j] if not isinstance(j, slice) or j != slice(None) else x.iloc[i, :]
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr[i]
    return arr[np.ix_(np.atleast_1d(i), np.atleast_1d(j))] if not (
        isinstance(i, slice) or isinstance(j, slice)
    ) else arr[i, :][:, j]


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
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            ) from e

    def __setattr__(self, name: str, value) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            ) from e

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
        if genes is not None and isinstance(genes, pd.DataFrame):
            row = list(genes.index)
        if targets is not None and isinstance(targets, pd.DataFrame):
            col = list(targets.index)
        return (row, col)

    # ---- head / tail ----

    def head(self, n: int = 6) -> "_LargeDataObject":
        n = min(max(n, 0), self.nrow)
        return self._subset(slice(0, n), slice(None))

    def tail(self, n: int = 6) -> "_LargeDataObject":
        n = min(max(n, 0), self.nrow)
        start = max(self.nrow - n, 0)
        return self._subset(slice(start, self.nrow), slice(None))

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
        i_idx = _resolve_index(i, row_names)
        j_idx = _resolve_index(j, col_names)

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
            raise TypeError(
                f"EList expected dict or EList, got {type(data).__name__}"
            )
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
            raise TypeError(
                f"MArrayLM expected dict or MArrayLM, got {type(data).__name__}"
            )
        for k, v in kwargs.items():
            self[k] = v

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
    "RGList", "MAList", "EListRaw", "ExpressionSet", "eSet",
    "PLMset", "marrayNorm", "DGEList",
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
        and is treated identically to an EList.
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
        scope - see policy_data_class_wrappers in project memory.
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
        import scipy.sparse as _sp

        def _densify(mat):
            # scanpy workflows routinely store counts as scipy.sparse;
            # np.asarray on sparse yields a 0-d object array, so
            # densify explicitly first.
            if _sp.issparse(mat):
                mat = mat.toarray()
            return np.asarray(mat, dtype=np.float64)

        adata = obj
        if layer is not None:
            mat = adata.layers[layer]
        else:
            mat = adata.X
        y["exprs"] = _densify(mat).T
        # Capture adata.var / adata.obs as full DataFrames even when they
        # have zero annotation columns - the index still carries
        # var_names / obs_names, and consumers (top_table genelist,
        # fit['targets'], diagnostic plots) care about that index.
        # Mirrors the DataFrame branch's non-RangeIndex capture logic.
        if adata.var is not None:
            y["probes"] = adata.var
        if adata.obs is not None:
            y["targets"] = adata.obs
        y["Amean"] = np.nanmean(y["exprs"], axis=1)
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
            y["weights"] = _densify(adata.layers[weights_layer]).T
        if layer is not None and layer.endswith("_E"):
            stem = layer[:-2]
            if weights_layer is None:
                weights_candidate = f"{stem}_weights"
                if weights_candidate in adata.layers:
                    y["weights"] = _densify(
                        adata.layers[weights_candidate]
                    ).T
            uns_payload = adata.uns.get(stem)
            if isinstance(uns_payload, dict):
                if uns_payload.get("design") is not None:
                    y["design"] = uns_payload["design"]
        return y

    if isinstance(obj, EList):
        return _eawp_from_elist_like(obj)

    cls_name = type(obj).__name__
    if cls_name in _UNSUPPORTED_WRAPPER_NAMES:
        raise TypeError(
            f"{cls_name} is not supported by pylimma. "
            "Two-channel and Bioconductor S4 wrappers are out of scope "
            "(policy_data_class_wrappers). Extract the expression matrix "
            "and pass it directly, or wrap it in pylimma.EList."
        )

    if isinstance(obj, dict):
        if "E" in obj:
            return _eawp_from_elist_like(obj)
        if "exprs" in obj:
            y.update({k: obj.get(k) for k in y.keys() if k in obj})
            if y["exprs"] is not None:
                y["exprs"] = np.asarray(y["exprs"], dtype=np.float64)
                if y["Amean"] is None:
                    y["Amean"] = np.nanmean(y["exprs"], axis=1)
            return y
        raise TypeError(
            "dict input to get_eawp must contain key 'E' (EList-style) "
            "or 'exprs' (getEAWP-output-style)"
        )

    if isinstance(obj, pd.DataFrame):
        numeric_cols = obj.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))
        if numeric_cols.all():
            y["exprs"] = obj.values.astype(np.float64)
        elif (~numeric_cols).sum() == 1 and len(obj.columns) > 1:
            y["exprs"] = obj.iloc[:, 1:].values.astype(np.float64)
            y["probes"] = obj.iloc[:, [0]]
        else:
            raise TypeError(
                "DataFrame input must be all-numeric or have exactly one "
                "non-numeric column (treated as gene IDs)"
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
        y["Amean"] = np.nanmean(y["exprs"], axis=1)
        return y

    arr = np.asarray(obj)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise TypeError(
            f"expression data must be 2-dimensional, got {arr.ndim}D"
        )
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError("expression data must be numeric")
    y["exprs"] = arr.astype(np.float64, copy=False)
    y["Amean"] = np.nanmean(y["exprs"], axis=1)
    return y


def _eawp_from_elist_like(obj) -> dict:
    """Extract EAWP dict from an EList or EList-shaped dict."""
    exprs = np.asarray(obj["E"], dtype=np.float64)
    return {
        "exprs": exprs,
        "weights": obj.get("weights"),
        "probes": obj.get("genes"),
        "Amean": np.nanmean(exprs, axis=1),
        "design": obj.get("design"),
        "targets": obj.get("targets"),
    }


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
        uns_payload = {k: v for k, v in slots.items()
                       if k not in ("E",) and v is not None
                       and not (k == "weights" and weights_layer is not None)}
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
                f"No fit results found in adata.uns[{key!r}]. "
                "Did you run lm_fit() first?"
            )
        return data.uns[key], data, key
    return data, None, None

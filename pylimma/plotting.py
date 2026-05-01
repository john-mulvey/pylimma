# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   plotWithHighlights.R                  Copyright (C) 2014-2020 Gordon Smyth
#   plots-ma.R                            Copyright (C) 2003-2024 Gordon Smyth,
#                                                                 James Wettenhall
#   plotMD.R                              Copyright (C) 2003-2024 Gordon Smyth,
#                                                                 James Wettenhall
#   plots-fit.R (volcanoplot)             Copyright (C) 2006-2020 Gordon Smyth
#   plots-fit.R (plotSA)                  Copyright (C) 2009-2025 Gordon Smyth
#   plotdensities.R                       Copyright (C) 2003-2015 Natalie Thorne,
#                                                                 Gordon Smyth
#   plotMDS.R                             Copyright (C) 2009-2025 Gordon Smyth,
#                                                                 Di Wu,
#                                                                 Yifang Hu
#   venn.R                                Copyright (C) 2003-2020 Gordon Smyth,
#                                                                 James Wettenhall,
#                                                                 Yifang Hu,
#                                                                 Francois Pepin
#   coolmap.R                             Copyright (C) 2016-2019 Gordon Smyth
#   barcodeplot.R                         Copyright (C) 2008-2019 Gordon Smyth,
#                                                                 Di Wu,
#                                                                 Yifang Hu
# Python port: Copyright (C) 2026 John Mulvey
"""
Diagnostic and exploratory plots for pylimma.

Functions:

- plot_with_highlights: scatterplot with colour/size highlighting.
- plot_ma / plot_md: mean-average / mean-difference plots.
- volcano_plot: log-fold-change vs significance.
- plot_sa: sqrt(sigma) vs Amean, with variance-trend overlay.
- plot_densities: per-sample kernel-density curves.
- plot_mds / _mds_coordinates: multidimensional scaling.
- venn_counts / venn_diagram: intersection counts and 2/3-set Venns.
- coolmap: clustered log2-expression heatmap.
- barcode_plot: enrichment plot for one or two gene sets.

matplotlib is imported lazily inside each plot function, so numeric
helpers (``_mds_coordinates``, ``venn_counts``) stay importable without
it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .classes import EList, MArrayLM
from .utils import p_adjust, tricube_moving_average

_LEGEND_ANCHORS = {
    "bottomright": "lower right",
    "bottom": "lower center",
    "bottomleft": "lower left",
    "left": "center left",
    "topleft": "upper left",
    "top": "upper center",
    "topright": "upper right",
    "right": "center right",
    "center": "center",
}


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "pylimma plotting functions require matplotlib. "
            "Install with `pip install pylimma[plot]`."
        ) from e
    return plt


def _resolve_legend(legend):
    """Translate R's ``legend`` parameter into (show, matplotlib_loc)."""
    if legend is False or legend == "none":
        return False, None
    if legend is True:
        pos = "topleft"
    else:
        pos = str(legend)
    if pos not in _LEGEND_ANCHORS:
        raise ValueError(
            f"'{pos}' is not a valid legend anchor. Must be one of {list(_LEGEND_ANCHORS)}"
        )
    return True, _LEGEND_ANCHORS[pos]


# ----------------------------------------------------------------------------
# plotWithHighlights
# ----------------------------------------------------------------------------


def plot_with_highlights(
    x,
    y,
    status=None,
    values=None,
    hl_pch="o",
    hl_col=None,
    hl_cex=1.0,
    legend="topright",
    bg_pch="o",
    bg_col="black",
    bg_cex=0.3,
    pch=None,
    col=None,
    cex=None,
    xlab: str = "",
    ylab: str = "",
    main: str | None = None,
    ax=None,
    **kwargs,
):
    """Scatterplot with colour/size highlighting for special groups of points.

    Port of R limma's ``plotWithHighlights`` (Gordon Smyth, 2014-2020).
    Returns the matplotlib Axes so callers can further decorate.
    """
    plt = _require_matplotlib()

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if ax is None:
        _, ax = plt.subplots()

    # Status NULL or all NA -> plot all as background
    def _all_na(arr):
        if arr is None:
            return True
        a = np.asarray(arr)
        if a.dtype.kind in ("U", "S", "O"):
            return all(v is None or (isinstance(v, float) and np.isnan(v)) for v in a.tolist())
        return bool(np.all(np.isnan(a.astype(float, copy=False))))

    if status is None or _all_na(status):
        ax.scatter(x, y, marker=bg_pch, c=bg_col, s=(float(bg_cex) * 25) ** 1, **kwargs)
        if main is not None:
            ax.set_title(main)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        return ax

    status_arr = np.asarray([str(v) for v in np.asarray(status).tolist()])

    # Build `values` if not supplied: unique status values (most frequent
    # first becomes background; remaining are highlights, in descending
    # frequency order).
    if values is None:
        unique, counts = np.unique(status_arr, return_counts=True)
        order = np.argsort(-counts, kind="stable")
        status_values = unique[order]
        values = status_values[1:].tolist()
    else:
        values = [
            str(v) for v in (values if isinstance(values, (list, tuple, np.ndarray)) else [values])
        ]

    nvalues = len(values)
    if nvalues == 0:
        ax.scatter(x, y, marker=bg_pch, c=bg_col, s=(float(bg_cex) * 25))
        if main is not None:
            ax.set_title(main)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        return ax

    # Allow legacy pch/col/cex
    if hl_pch == "o" and pch is not None:
        hl_pch = pch
    if hl_col is None and col is not None:
        hl_col = col
    if hl_cex == 1.0 and cex is not None:
        hl_cex = cex

    # Background mask: status not in values
    values_set = set(values)
    bg_mask = np.array([s not in values_set for s in status_arr])
    nonhi = bool(np.any(bg_mask))

    # Resolve per-category pch / col / cex (length nvalues each, recycling)
    def _rep_len(x, n):
        arr = np.atleast_1d(np.asarray(x, dtype=object))
        return np.array([arr[i % len(arr)] for i in range(n)], dtype=object)

    hl_pch_v = _rep_len(hl_pch, nvalues)
    hl_cex_v = np.array(_rep_len(hl_cex, nvalues).tolist(), dtype=np.float64)
    if hl_col is None:
        base = 1 if nonhi else 0
        default_cycle = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
        hl_col_v = np.array(
            [default_cycle[(base + i) % len(default_cycle)] for i in range(nvalues)], dtype=object
        )
    else:
        hl_col_v = _rep_len(hl_col, nvalues)

    # Draw background first
    if nonhi:
        ax.scatter(x[bg_mask], y[bg_mask], marker=bg_pch, c=bg_col, s=(float(bg_cex) * 25))

    # Draw highlighted points by category
    for i, val in enumerate(values):
        sel = status_arr == val
        if not np.any(sel):
            continue
        ax.scatter(
            x[sel],
            y[sel],
            marker=hl_pch_v[i],
            c=[hl_col_v[i]],
            s=(float(hl_cex_v[i]) * 25),
            label=val,
        )

    # Ensure axes limits include all data (R's plot sets type="n" then adds)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if main is not None:
        ax.set_title(main)

    show_legend, loc = _resolve_legend(legend)
    if show_legend:
        ax.legend(loc=loc)

    return ax


# ----------------------------------------------------------------------------
# plotMA / plotMD
# ----------------------------------------------------------------------------


def _get_matrix_E(object, *, want_weights=False):
    """Extract (E, weights) from ndarray / EList / dict. Returns
    ``(E, weights)`` where weights may be None."""
    if isinstance(object, EList):
        E = np.asarray(object["E"], dtype=np.float64)
        w = object.get("weights") if want_weights else None
        return E, w
    if isinstance(object, dict) and "E" in object:
        E = np.asarray(object["E"], dtype=np.float64)
        w = object.get("weights") if want_weights else None
        return E, w
    arr = np.asarray(object)
    if arr.ndim != 2:
        raise ValueError("plot requires a 2D expression matrix")
    return arr.astype(np.float64), None


def _ma_from_matrix(E, col, weights=None, zero_weights=False):
    """Compute (A, M) for plotMA/plotMD matrix path.

    ``A = (E[:, col] + rowMeans(E[:, -col])) / 2`` (midpoint, per R),
    ``M = E[:, col] - rowMeans(E[:, -col])``.
    """
    if E.shape[1] < 2:
        raise ValueError("Need at least two columns")
    other_cols = np.delete(np.arange(E.shape[1]), col)
    others = E[:, other_cols]
    # rowMeans(na.rm=TRUE) - tolerate NaN in the averages
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ave = np.nanmean(others, axis=1)
    m = E[:, col] - ave
    a = (E[:, col] + ave) / 2.0
    if weights is not None and not zero_weights:
        w = np.asarray(weights)[:, col]
        bad = np.isnan(w) | (w <= 0)
        m = m.copy()
        m[bad] = np.nan
    return a, m


def plot_ma(
    object,
    array: int = 0,
    coef: int | None = None,
    xlab: str = "Average log-expression",
    ylab: str = "log-fold-change",
    main: str | None = None,
    status=None,
    zero_weights: bool = False,
    ax=None,
    **kwargs,
):
    """MA plot for a matrix, EList, or MArrayLM.

    Port of R limma's ``plotMA``. Dispatches on the class of ``object``:

    - ``ndarray`` / matrix-like: ``(A, M)`` computed from column ``array``
      vs the mean of the remaining columns.
    - ``EList``: the ``E`` slot is extracted and the matrix path runs.
    - ``MArrayLM``: ``x = Amean``, ``y = coefficients[:, coef]``.

    Index convention: ``array`` and ``coef`` are 0-based (matching
    ``top_table``'s convention throughout pylimma).
    """
    if isinstance(object, MArrayLM):
        if object.get("Amean") is None:
            raise ValueError("Amean component is absent.")
        c = coef if coef is not None else 0
        if isinstance(c, str):
            names = object.get("contrast_names") or object.get("coef_names")
            if names is None or c not in list(names):
                raise ValueError(f"coef {c!r} not found")
            c = list(names).index(c)
        coefs = np.asarray(object["coefficients"])
        if coefs.ndim == 1:
            y = coefs
        else:
            y = coefs[:, c]
        x = np.asarray(object["Amean"], dtype=np.float64)
    else:
        E, w = _get_matrix_E(object, want_weights=True)
        x, y = _ma_from_matrix(E, int(array), weights=w, zero_weights=zero_weights)

    return plot_with_highlights(
        x, y, status=status, xlab=xlab, ylab=ylab, main=main, ax=ax, **kwargs
    )


def plot_md(
    object,
    column: int = 0,
    coef: int | None = None,
    xlab: str = "Average log-expression",
    ylab: str = "log-fold-change",
    main: str | None = None,
    status=None,
    zero_weights: bool = False,
    ax=None,
    **kwargs,
):
    """Mean-difference plot. Port of R limma's ``plotMD``.

    Dispatch and numeric substrate mirror ``plot_ma`` exactly (``plotMA``
    and ``plotMD`` use identical matrix-path computations in R). The only
    behavioural difference is that when ``coef`` is supplied for an
    MArrayLM, it overrides ``column``.
    """
    if isinstance(object, MArrayLM):
        if object.get("Amean") is None:
            raise ValueError("Amean component is absent.")
        c = coef if coef is not None else column
        if isinstance(c, str):
            names = object.get("contrast_names") or object.get("coef_names")
            if names is None or c not in list(names):
                raise ValueError(f"coef {c!r} not found")
            c = list(names).index(c)
        coefs = np.asarray(object["coefficients"])
        if coefs.ndim == 1:
            y = coefs
        else:
            y = coefs[:, c]
        x = np.asarray(object["Amean"], dtype=np.float64)
    else:
        E, w = _get_matrix_E(object, want_weights=True)
        x, y = _ma_from_matrix(E, int(column), weights=w, zero_weights=zero_weights)

    return plot_with_highlights(
        x, y, status=status, xlab=xlab, ylab=ylab, main=main, ax=ax, **kwargs
    )


# ----------------------------------------------------------------------------
# volcanoplot / plotSA
# ----------------------------------------------------------------------------


def volcano_plot(
    fit,
    coef: int | str = 0,
    style: str = "p-value",
    highlight: int = 0,
    names=None,
    hl_col: str = "blue",
    xlab: str = "Log2 Fold Change",
    ylab: str | None = None,
    pch="o",
    cex: float = 0.35,
    ax=None,
    **kwargs,
):
    """Volcano plot of log-fold-change vs significance.

    Port of R limma's ``volcanoplot``. ``coef`` is 0-based (pylimma
    convention). ``style`` is one of ``"p-value"`` or ``"b-statistic"``.
    """
    plt = _require_matplotlib()
    if not isinstance(fit, MArrayLM) and not isinstance(fit, dict):
        raise ValueError("fit must be an MArrayLM")

    style_low = str(style).lower()
    if style_low not in ("p-value", "b-statistic"):
        raise ValueError(f"style must be 'p-value' or 'b-statistic', got {style!r}")

    c = coef
    if isinstance(c, str):
        names_list = fit.get("contrast_names") or fit.get("coef_names")
        if names_list is None or c not in list(names_list):
            raise ValueError(f"coef {c!r} not found")
        c = list(names_list).index(c)

    coefs = np.asarray(fit["coefficients"])
    x = coefs[:, c] if coefs.ndim > 1 else coefs

    if style_low == "p-value":
        if fit.get("p_value") is None:
            raise ValueError("No p-values found in linear model fit object. Please run e_bayes.")
        p = np.asarray(fit["p_value"])
        y = -np.log10(p[:, c] if p.ndim > 1 else p)
        if ylab is None:
            ylab = "-log10(P-value)"
    else:
        if fit.get("lods") is None:
            raise ValueError(
                "No B-statistics found in linear model fit object. Please run e_bayes."
            )
        lods = np.asarray(fit["lods"])
        y = lods[:, c] if lods.ndim > 1 else lods
        if ylab is None:
            ylab = "Log Odds of Differential Expression"

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(x, y, marker=pch, c="black", s=(float(cex) * 25))
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if highlight > 0:
        if names is None:
            gene_names = np.arange(1, len(x) + 1)
        else:
            gene_names = np.asarray(names)
        gene_names = np.array([str(g) for g in gene_names.tolist()])
        o = np.argsort(-y, kind="stable")
        i_top = o[: int(highlight)]
        for i in i_top:
            ax.text(x[i], y[i], gene_names[i][:8], color=hl_col, fontsize=8)

    return ax


def plot_sa(
    fit,
    xlab: str = "Average log-expression",
    ylab: str = "sqrt(sigma)",
    zero_weights: bool = False,
    pch="o",
    cex: float = 0.3,
    col: tuple[str, str] = ("black", "red"),
    ax=None,
    **kwargs,
):
    """Sigma vs Amean plot. Port of R limma's ``plotSA``.

    The y-axis is ``sqrt(sigma)``. When ``s2_prior`` is an ndarray, the
    variance-trend overlay is ``sqrt(sqrt(s2_prior[order(x)]))`` in sorted
    x order. When ``s2_prior`` is scalar, a single horizontal line is
    drawn at ``sqrt(sqrt(s2_prior))``.

    Under robust eBayes (per-gene ``df_prior``), outlier genes are
    identified via ``p_adjust(2 * pmin(pdn, pup), method='BH') <= 0.5``
    and plotted in ``col[1]``.
    """
    plt = _require_matplotlib()
    from scipy.stats import f as f_dist

    if not isinstance(fit, (MArrayLM, dict)):
        raise ValueError("fit must be an MArrayLM object")

    x = fit.get("Amean")
    y = fit.get("sigma")
    dfg = fit.get("df_residual")
    df0 = fit.get("df_prior")
    s20 = fit.get("s2_prior")
    if x is None:
        raise ValueError("fit['Amean'] is None")
    if y is None:
        raise ValueError("fit['sigma'] is None")
    if dfg is None:
        raise ValueError("fit['df_residual'] is None")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).copy()
    dfg = np.asarray(dfg, dtype=np.float64)
    if dfg.ndim == 0:
        dfg = np.full(len(x), float(dfg))

    if zero_weights:
        dfg = np.maximum(dfg, 1e-6)
    else:
        y[dfg < 1e-6] = np.nan
        if fit.get("weights") is not None:
            w = np.asarray(fit["weights"])
            allzero = np.nansum(w > 0, axis=1) == 0
            y[allzero] = np.nan

    colv = np.array([col[0]] * len(y), dtype=object)

    df0_arr = np.atleast_1d(np.asarray(df0, dtype=np.float64))
    if df0_arr.size > 1:
        df2 = float(np.max(df0_arr))
        s20_arr = np.atleast_1d(np.asarray(s20, dtype=np.float64))
        if s20_arr.size == 1:
            s20_bcast = np.full(len(y), float(s20_arr[0]))
        else:
            s20_bcast = s20_arr
        with np.errstate(invalid="ignore", divide="ignore"):
            s2 = y**2 / s20_bcast
            pdn = f_dist.cdf(s2, dfn=dfg, dfd=df2)
            pup = f_dist.sf(s2, dfn=dfg, dfd=df2)
        raw = 2 * np.minimum(pdn, pup)
        fdr = p_adjust(raw, method="BH")
        colv[fdr <= 0.5] = col[1]

    y_plot = np.sqrt(y)

    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(x, y_plot, marker=pch, c=colv.tolist(), s=(float(cex) * 25))
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if s20 is not None:
        s20_arr = np.atleast_1d(np.asarray(s20, dtype=np.float64))
        if s20_arr.size == 1:
            ax.axhline(float(np.sqrt(np.sqrt(s20_arr[0]))), color="blue")
        else:
            o = np.argsort(x, kind="stable")
            ax.plot(x[o], np.sqrt(np.sqrt(s20_arr[o])), color="blue")

    if df0_arr.size > 1:
        ax.legend(["Normal", "Outlier"], loc="upper right")

    return ax


# ----------------------------------------------------------------------------
# plotDensities
# ----------------------------------------------------------------------------


def _r_density_gaussian(
    x: np.ndarray,
    n_pts: int = 512,
    cut: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``stats::density.default`` with ``kernel='gaussian'``.

    Uses ``bw.nrd0`` for bandwidth and R's FFT-based binned KDE: linear
    binning via ``massdist``, FFT convolution with a Gaussian kernel,
    linear interpolation back onto the requested grid.
    """
    # Import the shared bandwidth helper from normalize.py
    from .normalize import _r_bw_nrd0

    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    N = len(x)
    if N < 2:
        raise ValueError("need at least 2 data points")

    bw = _r_bw_nrd0(x)

    n_user = int(n_pts)
    n = max(n_user, 512)
    if n > 512:
        n = 2 ** int(np.ceil(np.log2(n)))

    from_ = float(x.min()) - cut * bw
    to = float(x.max()) + cut * bw
    lo = from_ - 4.0 * bw
    up = to + 4.0 * bw

    y = np.zeros(2 * n)
    xdelta = (up - lo) / (n - 1)
    w = 1.0 / N
    xpos = (x - lo) / xdelta
    ix = np.floor(xpos).astype(np.int64)
    fx = xpos - ix
    interior = (ix >= 0) & (ix <= n - 2)
    np.add.at(y, ix[interior], (1 - fx[interior]) * w)
    np.add.at(y, ix[interior] + 1, fx[interior] * w)
    left = ix == -1
    right = ix == n - 1
    if left.any():
        np.add.at(y, np.zeros(int(left.sum()), dtype=np.int64), fx[left] * w)
    if right.any():
        np.add.at(y, np.full(int(right.sum()), n - 1, dtype=np.int64), (1 - fx[right]) * w)

    dx = (up - lo) / (n - 1)
    kords = np.empty(2 * n)
    kords[: n + 1] = np.arange(n + 1) * dx
    kords[n + 1 :] = -np.arange(n - 1, 0, -1) * dx

    # Gaussian kernel at bandwidth bw, evaluated on kords
    kv = np.exp(-0.5 * (kords / bw) ** 2) / (bw * np.sqrt(2 * np.pi))

    Y = np.fft.fft(y)
    K = np.fft.fft(kv)
    # numpy.fft.ifft already divides by N (= 2n), matching R's
    # ``fft(..., inverse=TRUE) / length(y)`` idiom.
    dens_ext = np.maximum(0.0, np.real(np.fft.ifft(Y * np.conj(K)))[:n])

    xords = np.linspace(lo, up, n)
    x_out = np.linspace(from_, to, n_user)
    y_out = np.interp(x_out, xords, dens_ext)
    return x_out, y_out


def plot_densities(
    object,
    log: bool = True,
    group=None,
    col=None,
    main: str | None = None,
    legend: str | bool = "topleft",
    ax=None,
    **kwargs,
):
    """Kernel-density plots of sample intensities.

    Port of R limma's single-channel ``plotDensities.default`` /
    ``plotDensities.EList``. For ``EList`` input, ``E`` is extracted; if
    ``log=False``, values are exponentiated (``2 ** E``) before plotting.
    RGList / MAList inputs are out of scope.
    """
    plt = _require_matplotlib()

    if isinstance(object, EList):
        E = np.asarray(object["E"], dtype=np.float64)
        if not log:
            E = 2.0**E
        colnames = None
        if isinstance(object.get("E"), pd.DataFrame):
            colnames = list(object["E"].columns)
    elif isinstance(object, dict) and "E" in object:
        E = np.asarray(object["E"], dtype=np.float64)
        if not log:
            E = 2.0**E
        colnames = None
    elif isinstance(object, pd.DataFrame):
        E = object.values.astype(np.float64)
        colnames = list(object.columns)
    else:
        E = np.asarray(object, dtype=np.float64)
        colnames = None

    if E.ndim != 2:
        raise ValueError("plotDensities requires a 2D expression matrix")
    narray = E.shape[1]
    if colnames is None:
        colnames = [str(i + 1) for i in range(narray)]

    if group is None:
        group = colnames
    group_arr = np.array([str(g) for g in group])
    levels = list(dict.fromkeys(group_arr.tolist()))  # preserve appearance
    ngroup = len(levels)

    default_cycle = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    if col is None:
        col_list = [default_cycle[i % len(default_cycle)] for i in range(ngroup)]
    else:
        col_v = col if isinstance(col, (list, tuple, np.ndarray)) else [col]
        col_list = [str(col_v[i % len(col_v)]) for i in range(ngroup)]

    level_to_col = {lev: col_list[i] for i, lev in enumerate(levels)}
    arraycol = [level_to_col[g] for g in group_arr.tolist()]

    if ax is None:
        _, ax = plt.subplots()

    X = np.zeros((512, narray))
    Y = np.zeros((512, narray))
    for a in range(narray):
        col_data = E[:, a]
        col_data = col_data[np.isfinite(col_data)]
        xx, yy = _r_density_gaussian(col_data, n_pts=512)
        X[:, a] = xx
        Y[:, a] = yy
        ax.plot(xx, yy, color=arraycol[a], linewidth=2)

    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    if main is not None:
        ax.set_title(main)

    show_legend, loc = _resolve_legend(legend)
    if show_legend and ngroup > 1:
        handles = [
            plt.Line2D([], [], color=level_to_col[lev], linewidth=2, label=lev) for lev in levels
        ]
        ax.legend(handles=handles, loc=loc)

    return ax


# ----------------------------------------------------------------------------
# plotMDS / _mds_coordinates
# ----------------------------------------------------------------------------


def _mds_coordinates(
    x: np.ndarray,
    top: int = 500,
    labels=None,
    gene_selection: str = "pairwise",
    ndim: int = 2,
) -> dict:
    """Compute multidimensional-scaling coordinates from an expression
    matrix. Pure-numeric helper (no plotting).

    Port of the numeric substrate of R limma's ``plotMDS.default``.

    Returns a dict with keys ``x``, ``y``, ``var_explained``,
    ``distance_matrix_squared``, ``gene_selection``, ``top``,
    ``eigen_values``, ``eigen_vectors``.
    """
    if gene_selection not in ("pairwise", "common"):
        raise ValueError(f"gene_selection must be 'pairwise' or 'common', got {gene_selection!r}")

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be 2-dimensional")
    nsamples = x.shape[1]
    if nsamples < 3:
        raise ValueError(f"Only {nsamples} columns of data: need at least 3")

    # Drop rows with missing/Inf values
    good = np.all(np.isfinite(x), axis=1)
    x = x[good, :]
    nprobes = x.shape[0]
    if nprobes == 0:
        raise ValueError("No finite rows")

    top = min(int(top), nprobes)

    ndim_clip = max(int(ndim), 2)
    if nsamples < ndim_clip:
        raise ValueError("ndim is greater than number of samples")
    if nprobes < ndim_clip:
        raise ValueError("ndim is greater than number of rows of data")

    dd = np.zeros((nsamples, nsamples), dtype=np.float64)
    if gene_selection == "pairwise":
        topindex = nprobes - top  # 0-based: keep elements [topindex:nprobes]
        for i in range(1, nsamples):
            for j in range(i):
                diffs2 = (x[:, i] - x[:, j]) ** 2
                # partial sort: select largest `top` values
                sorted_arr = np.sort(diffs2)
                dd[i, j] = np.mean(sorted_arr[topindex:])
    else:
        if nprobes > top:
            rm = x.mean(axis=1, keepdims=True)
            s = ((x - rm) ** 2).mean(axis=1)
            o = np.argsort(-s, kind="stable")
            x = x[o[:top], :]
        for i in range(1, nsamples):
            dd[i, :i] = ((x[:, [i]] - x[:, :i]) ** 2).mean(axis=0)

    dd = dd + dd.T
    rm = dd.mean(axis=1)
    dd = dd - rm[:, None]
    dd = dd.T - (rm - rm.mean())[:, None]

    # MDS via eigendecomposition of -dd/2
    B = -dd / 2.0
    B = (B + B.T) / 2.0  # enforce symmetry for eigh
    eigvals, eigvecs = np.linalg.eigh(B)
    # eigh returns ASCENDING - reverse to descending to match R's eigen().
    order = np.argsort(-eigvals, kind="stable")
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    lam = np.maximum(eigvals, 0.0)
    var_explained = lam / lam.sum() if lam.sum() > 0 else lam

    # Default dim.plot = (1, 2) in R; in pylimma's 0-based convention
    # these are indices (0, 1).
    i1, i2 = 0, 1
    coord1 = eigvecs[:, i1] * np.sqrt(lam[i1])
    coord2 = eigvecs[:, i2] * np.sqrt(lam[i2])

    return {
        "x": coord1,
        "y": coord2,
        "var_explained": var_explained,
        "distance_matrix_squared": dd,
        "gene_selection": gene_selection,
        "top": top,
        "eigen_values": eigvals,
        "eigen_vectors": eigvecs,
    }


def plot_mds(
    x,
    top: int = 500,
    labels=None,
    pch=None,
    cex: float = 1.0,
    dim_plot: tuple[int, int] = (0, 1),
    gene_selection: str = "pairwise",
    xlab: str | None = None,
    ylab: str | None = None,
    var_explained: bool = True,
    ax=None,
    **kwargs,
):
    """Multidimensional-scaling plot. Port of R limma's ``plotMDS.default``.

    Dimensions are 0-based (pylimma convention).
    """
    plt = _require_matplotlib()

    if isinstance(x, EList):
        x_mat = np.asarray(x["E"], dtype=np.float64)
        sample_names = None
        E_df = x.get("E")
        if isinstance(E_df, pd.DataFrame):
            sample_names = list(E_df.columns)
    elif isinstance(x, pd.DataFrame):
        x_mat = x.values.astype(np.float64)
        sample_names = list(x.columns)
    else:
        x_mat = np.asarray(x, dtype=np.float64)
        sample_names = None

    ndim = max(int(dim_plot[0]), int(dim_plot[1])) + 1
    result = _mds_coordinates(x_mat, top=top, gene_selection=gene_selection, ndim=ndim)

    eigvals = result["eigen_values"]
    eigvecs = result["eigen_vectors"]
    lam = np.maximum(eigvals, 0.0)
    i1, i2 = int(dim_plot[0]), int(dim_plot[1])
    xcoord = eigvecs[:, i1] * np.sqrt(lam[i1])
    ycoord = eigvecs[:, i2] * np.sqrt(lam[i2])

    axislabel = "Leading logFC dim" if gene_selection == "pairwise" else "Principal Component"
    if xlab is None:
        xlab = f"{axislabel} {i1 + 1}"
    if ylab is None:
        ylab = f"{axislabel} {i2 + 1}"
    if var_explained:
        pct = np.round(result["var_explained"] * 100).astype(int)
        xlab = f"{xlab} ({pct[i1]}%)"
        ylab = f"{ylab} ({pct[i2]}%)"

    if ax is None:
        _, ax = plt.subplots()

    if labels is None and pch is None:
        labels = (
            sample_names
            if sample_names is not None
            else [str(i + 1) for i in range(x_mat.shape[1])]
        )

    if labels is not None:
        labels = [str(lab) for lab in labels]
        margin = 0.05 * (xcoord.max() - xcoord.min() + 1e-12)
        ax.set_xlim(xcoord.min() - margin, xcoord.max() + margin)
        ax.set_ylim(ycoord.min() - margin, ycoord.max() + margin)
        for xi, yi, lab in zip(xcoord, ycoord, labels):
            ax.text(xi, yi, lab, ha="center", va="center", fontsize=10 * cex)
    else:
        ax.scatter(xcoord, ycoord, marker=pch if pch is not None else "o", s=(float(cex) * 25))

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    return ax


# ----------------------------------------------------------------------------
# venn
# ----------------------------------------------------------------------------


def venn_counts(x, include: str = "both") -> pd.DataFrame:
    """Cross-tabulate significance indicators. Port of R limma's
    ``vennCounts``.

    ``include`` is ``"both"``, ``"up"``, or ``"down"``.
    """
    if include not in ("both", "up", "down"):
        raise ValueError(f"include must be 'both', 'up', or 'down', got {include!r}")

    if isinstance(x, pd.DataFrame):
        col_names = list(x.columns)
        x_arr = x.values.astype(np.float64)
    else:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        col_names = [f"col{i}" for i in range(x_arr.shape[1])]

    if include == "both":
        indicator = np.sign(np.abs(x_arr))
    elif include == "up":
        indicator = np.sign((x_arr > 0).astype(np.int64))
    else:
        indicator = np.sign((x_arr < 0).astype(np.int64))

    ncontrasts = indicator.shape[1]
    noutcomes = 2**ncontrasts
    outcomes = np.zeros((noutcomes, ncontrasts), dtype=np.int64)
    for j in range(ncontrasts):
        # R: rep(0:1, times=2^(j-1), each=2^(ncontrasts-j))  (1-based j)
        # In 0-based:   times = 2^j, each = 2^(ncontrasts-1-j)
        times = 2**j
        each = 2 ** (ncontrasts - 1 - j)
        pattern = np.tile(np.repeat(np.array([0, 1]), each), times)
        outcomes[:, j] = pattern

    # Count rows by outcome pattern. Match R's `table` output order: the
    # last column cycles fastest. R's loop:
    #   for (i in 1:ncontrasts) xlist[[i]] <- factor(x[,ncontrasts-i+1], ...)
    # so the leftmost factor in xlist is column `ncontrasts`, whose level
    # ordering cycles SLOWEST. table() cycles the LAST factor fastest,
    # i.e. column 1. Our outcomes matrix above does the same (j=0 cycles
    # slowest). Verify by matching the outcome pattern.
    indicator_int = indicator.astype(np.int64)
    counts = np.zeros(noutcomes, dtype=np.int64)
    for r in range(indicator_int.shape[0]):
        row = indicator_int[r, :]
        # Match this row against each outcome row
        idx = 0
        # outcome at idx has column j = (idx >> (ncontrasts - 1 - j)) & 1
        # but simpler: outcomes[idx, j] gives the pattern. We want the idx
        # where all columns match. From the construction:
        #   outcomes[idx, j] = (idx // (2**(ncontrasts-1-j))) % 2
        for j in range(ncontrasts):
            bit = int(row[j])
            if bit not in (0, 1):
                bit = 0
            idx += bit * (2 ** (ncontrasts - 1 - j))
        counts[idx] += 1

    result = pd.DataFrame(outcomes, columns=col_names)
    result["Counts"] = counts
    return result


def venn_diagram(
    object,
    include: str = "both",
    names=None,
    cex=(1.5, 1.0, 0.7),
    lwd: float = 1.0,
    circle_col=None,
    counts_col=None,
    show_include=None,
    main: str | None = None,
    ax=None,
    **kwargs,
):
    """2- or 3-circle Venn diagram.

    Port of R limma's ``vennDiagram`` for up to 3 sets. Circle centres and
    radii are hard-coded to match R's layout. For 4+ sets this port
    raises ``NotImplementedError``.
    """
    plt = _require_matplotlib()

    if isinstance(include, (list, tuple, np.ndarray)):
        include_list = [str(v) for v in include]
    else:
        include_list = [str(include)]
    len_inc = min(len(include_list), 2)

    # Get counts (VennCounts-shaped DataFrame or the input to convert)
    if isinstance(object, pd.DataFrame) and "Counts" in object.columns:
        # Already VennCounts-like; LenInc forced to 1
        counts_df = object
        include_list = [include_list[0]]
        len_inc = 1
    else:
        counts_df = venn_counts(object, include=include_list[0])
        if len_inc > 1:
            counts2_df = venn_counts(object, include=include_list[1])

    nsets = counts_df.shape[1] - 1
    if nsets > 3:
        raise NotImplementedError(f"pylimma's venn_diagram only supports 1-3 sets; got {nsets}")

    if names is None:
        names = list(counts_df.columns[:nsets])

    if circle_col is None:
        circle_col_list = ["black"] * nsets
    else:
        cc = circle_col if isinstance(circle_col, (list, tuple, np.ndarray)) else [circle_col]
        circle_col_list = [str(cc[i % len(cc)]) for i in range(nsets)]

    if counts_col is None:
        counts_col_list = ["black"] * len_inc
    else:
        cc = counts_col if isinstance(counts_col, (list, tuple, np.ndarray)) else [counts_col]
        counts_col_list = [str(cc[i % len(cc)]) for i in range(len_inc)]

    if show_include is None:
        show_include = len_inc == 2

    # Build count lookup keyed by bit-pattern string
    def _pattern_counts(df):
        patterns = df.iloc[:, :nsets].astype(int).values
        cts = df["Counts"].astype(int).values
        return {"".join(str(v) for v in row): int(c) for row, c in zip(patterns, cts)}

    z = _pattern_counts(counts_df)
    z2 = _pattern_counts(counts2_df) if len_inc == 2 else None

    if ax is None:
        _, ax = plt.subplots()
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.axis("off")

    theta = np.linspace(0, 2 * np.pi, 361)
    xcentres = {1: [0.0], 2: [-1.0, 1.0], 3: [-1.0, 1.0, 0.0]}[nsets]
    ycentres = {
        1: [0.0],
        2: [0.0, 0.0],
        3: [1.0 / np.sqrt(3), 1.0 / np.sqrt(3), -2.0 / np.sqrt(3)],
    }[nsets]
    r = 1.5
    xtext = {1: [-1.2], 2: [-1.2, 1.2], 3: [-1.2, 1.2, 0.0]}[nsets]
    ytext = {1: [1.8], 2: [1.8, 1.8], 3: [2.4, 2.4, -3.0]}[nsets]

    for k in range(nsets):
        ax.plot(
            xcentres[k] + r * np.cos(theta),
            ycentres[k] + r * np.sin(theta),
            color=circle_col_list[k],
            linewidth=lwd,
        )
        ax.text(xtext[k], ytext[k], names[k], fontsize=12 * cex[0], ha="center")

    if nsets in (1, 2):
        ax.plot([-3, 3, 3, -3, -3], [-2.5, -2.5, 2.5, 2.5, -2.5], color="black", linewidth=lwd)
    else:
        ax.plot([-3, 3, 3, -3, -3], [-3.5, -3.5, 3.3, 3.3, -3.5], color="black", linewidth=lwd)

    def _show_counts(zz, cex_val, adj, color, leg):
        if nsets == 1:
            ax.text(
                2.3, -2.1, str(zz["1"]), color=color, fontsize=10 * cex_val, ha="center", va=adj
            )
            ax.text(0, 0, str(zz["0"]), color=color, fontsize=10 * cex_val, ha="center", va=adj)
            if show_include:
                ax.text(-2.3, -2.1, leg, color=color, fontsize=10 * cex_val, ha="center", va=adj)
        elif nsets == 2:
            ax.text(
                2.3, -2.1, str(zz["00"]), color=color, fontsize=10 * cex_val, ha="center", va=adj
            )
            ax.text(
                1.5, 0.1, str(zz["10"]), color=color, fontsize=10 * cex_val, ha="center", va=adj
            )
            ax.text(
                -1.5, 0.1, str(zz["01"]), color=color, fontsize=10 * cex_val, ha="center", va=adj
            )
            ax.text(0, 0.1, str(zz["11"]), color=color, fontsize=10 * cex_val, ha="center", va=adj)
            if show_include:
                ax.text(-2.3, -2.1, leg, color=color, fontsize=10 * cex_val, ha="center", va=adj)
        else:
            ax.text(
                2.5, -3, str(zz["000"]), color=color, fontsize=10 * cex_val, ha="center", va=adj
            )
            ax.text(
                0, -1.7, str(zz["001"]), color=color, fontsize=10 * cex_val, ha="center", va=adj
            )
            ax.text(1.5, 1, str(zz["010"]), color=color, fontsize=10 * cex_val, ha="center", va=adj)
            ax.text(
                0.75, -0.35, str(zz["011"]), color=color, fontsize=10 * cex_val, ha="center", va=adj
            )
            ax.text(
                -1.5, 1, str(zz["100"]), color=color, fontsize=10 * cex_val, ha="center", va=adj
            )
            ax.text(
                -0.75,
                -0.35,
                str(zz["101"]),
                color=color,
                fontsize=10 * cex_val,
                ha="center",
                va=adj,
            )
            ax.text(0, 0.9, str(zz["110"]), color=color, fontsize=10 * cex_val, ha="center", va=adj)
            ax.text(0, 0, str(zz["111"]), color=color, fontsize=10 * cex_val, ha="center", va=adj)
            if show_include:
                ax.text(-2.5, -3, leg, color=color, fontsize=10 * cex_val, ha="center", va=adj)

    adj = "center" if len_inc == 1 else "bottom"
    _show_counts(z, cex[0], adj, counts_col_list[0], include_list[0])
    if len_inc == 2:
        _show_counts(z2, cex[0], "top", counts_col_list[1], include_list[1])

    if main is not None:
        ax.set_title(main)
    return ax


# ----------------------------------------------------------------------------
# coolmap
# ----------------------------------------------------------------------------

# coolmap's R version delegates layout to gplots::heatmap.2; this port
# re-implements the layout in matplotlib, so gplots is not an upstream.

_COOLMAP_LINKAGES = {
    "none",
    "ward.D",
    "single",
    "complete",
    "average",
    "mcquitty",
    "median",
    "centroid",
    "ward.D2",
}


def _coolmap_linkage(name: str) -> str:
    n = str(name)
    if n in ("w", "wa", "war", "ward"):
        n = "ward.D2"
    if n not in _COOLMAP_LINKAGES:
        raise ValueError(f"linkage must be one of {_COOLMAP_LINKAGES}, got {name!r}")
    # Translate R -> scipy method names
    mapping = {
        "ward.D": "ward",  # R's ward.D is the older (non-squared) method;
        # scipy only has squared-distance 'ward'. We map
        # to 'ward' which matches R's ward.D2.
        "ward.D2": "ward",
        "mcquitty": "weighted",
    }
    return mapping.get(n, n)


_COOLMAP_PALETTES = {"redblue", "redgreen", "yellowblue", "whitered"}


def _coolmap_cmap(name: str):
    from matplotlib.colors import LinearSegmentedColormap

    if name not in _COOLMAP_PALETTES:
        raise ValueError(f"col must be one of {_COOLMAP_PALETTES}, got {name!r}")
    if name == "redblue":
        return LinearSegmentedColormap.from_list("redblue", ["#0000ee", "white", "#ee0000"], N=256)
    if name == "redgreen":
        return LinearSegmentedColormap.from_list("redgreen", ["green", "black", "red"], N=256)
    if name == "yellowblue":
        return LinearSegmentedColormap.from_list(
            "yellowblue", ["#0000ee", "white", "#eeee00"], N=256
        )
    return LinearSegmentedColormap.from_list("whitered", ["white", "#ee0000"], N=256)


def coolmap(
    x,
    cluster_by: str = "de pattern",
    col=None,
    linkage_row: str = "complete",
    linkage_col: str = "complete",
    show_dendrogram: str = "both",
    ax=None,
    **kwargs,
):
    """Clustered heatmap with log2-expression colour scheme.

    Port of the numeric substrate of R limma's ``coolmap``. Hierarchical
    clustering and row z-scoring match R; layout is re-implemented in
    matplotlib (R's ``coolmap`` delegates to ``gplots::heatmap.2``, which
    is not ported). Returns the matplotlib Figure.
    """
    plt = _require_matplotlib()
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    if cluster_by not in ("de pattern", "expression level"):
        raise ValueError(
            f"cluster_by must be 'de pattern' or 'expression level', got {cluster_by!r}"
        )
    if show_dendrogram not in ("both", "row", "column", "none"):
        raise ValueError(
            f"show_dendrogram must be one of 'both', 'row', 'column', "
            f"'none'; got {show_dendrogram!r}"
        )

    if col is None:
        col = "redblue" if cluster_by == "de pattern" else "yellowblue"
    if isinstance(col, str):
        cmap = _coolmap_cmap(col)
    else:
        cmap = col

    if isinstance(x, pd.DataFrame):
        row_names = list(x.index)
        col_names = list(x.columns)
        M = x.values.astype(np.float64)
    else:
        M = np.asarray(x, dtype=np.float64)
        row_names = [str(i + 1) for i in range(M.shape[0])]
        col_names = [str(j + 1) for j in range(M.shape[1])]

    if M.shape[1] < 2:
        raise ValueError("Need at least 2 rows and 2 columns")

    link_row = _coolmap_linkage(linkage_row)
    link_col = _coolmap_linkage(linkage_col)

    if cluster_by == "de pattern":
        row_means = np.nanmean(M, axis=1)
        df = M.shape[1] - 1
        isna = np.isnan(M)
        if isna.any():
            df_row = df - isna.sum(axis=1)
            df_row[df_row == 0] = 1
        else:
            df_row = np.full(M.shape[0], df, dtype=np.float64)
        Z = M - row_means[:, None]
        V = np.nansum(Z**2, axis=1) / df_row
        Z = Z / np.sqrt(V + 0.01)[:, None]
        sym = True
    else:
        Z = M
        sym = False

    # Column clustering
    if linkage_col == "none":
        col_order = np.arange(Z.shape[1])
        show_dendrogram_eff = show_dendrogram.replace("both", "row").replace("column", "none")
    else:
        col_link = linkage(pdist(Z.T, metric="euclidean"), method=link_col)
        col_order = leaves_list(col_link)
        show_dendrogram_eff = show_dendrogram

    if linkage_row == "none":
        row_order = np.arange(Z.shape[0])
        show_dendrogram_eff = show_dendrogram_eff.replace("both", "column").replace("row", "none")
        row_link = None
    else:
        row_link = linkage(pdist(Z, metric="euclidean"), method=link_row)
        row_order = leaves_list(row_link)

    Z_reordered = Z[np.ix_(row_order, col_order)]
    row_labels = [row_names[i] for i in row_order]
    col_labels = [col_names[j] for j in col_order]

    if sym:
        vmax = float(np.nanmax(np.abs(Z_reordered)))
        vmin = -vmax
    else:
        vmin = float(np.nanmin(Z_reordered))
        vmax = float(np.nanmax(Z_reordered))

    fig, ax_main = plt.subplots(figsize=(6, 6))
    im = ax_main.imshow(Z_reordered, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_main.set_xticks(np.arange(len(col_labels)))
    ax_main.set_xticklabels(col_labels, rotation=90, fontsize=8)
    ax_main.set_yticks(np.arange(len(row_labels)))
    ax_main.set_yticklabels(row_labels, fontsize=6)
    fig.colorbar(im, ax=ax_main, label="Z-Score" if sym else "log2(expression)")
    return fig


# ----------------------------------------------------------------------------
# barcodeplot
# ----------------------------------------------------------------------------


def barcode_plot(
    statistics,
    index=None,
    index2=None,
    gene_weights=None,
    weights_label: str = "Weight",
    labels: tuple[str, str] = ("Down", "Up"),
    quantiles: tuple[float, float] = (-np.sqrt(2), np.sqrt(2)),
    col_bars=None,
    alpha: float = 0.4,
    worm: bool = True,
    span_worm: float = 0.45,
    xlab: str = "Statistic",
    ax=None,
    **kwargs,
):
    """Barcode plot of one or two gene sets. Port of R limma's
    ``barcodeplot``.
    """
    plt = _require_matplotlib()

    stat = np.asarray(statistics, dtype=np.float64).copy()
    nstat = len(stat)

    # Check / coerce index
    if index is None:
        if gene_weights is None:
            raise ValueError("Must specify at least one of index or gene_weights")
        if len(gene_weights) != nstat:
            raise ValueError("No index and length(gene_weights) doesn't equal length(statistics)")
        index = np.ones(nstat, dtype=bool)
        index2 = None
    else:
        idx = np.asarray(index)
        if idx.dtype == bool:
            if len(idx) != nstat:
                raise ValueError("Length of index disagrees with statistics")
        else:
            if len(idx) > nstat:
                raise ValueError("Length of index disagrees with statistics")

    # index2 handling (two-set from two indexes -> collapse to weights)
    if index2 is not None:
        if gene_weights is not None:
            warnings.warn("gene_weights ignored", UserWarning)
        w = np.zeros(nstat)
        w[_as_logical(index, nstat)] = 1.0
        w[_as_logical(index2, nstat)] = -1.0
        gene_weights = w
        index = np.ones(nstat, dtype=bool)
        index2 = None

    gene_weights2 = None
    if gene_weights is not None:
        gw = np.asarray(gene_weights, dtype=np.float64)
        if np.any(np.isnan(gw)):
            raise ValueError("Need to provide gene_weights without NAs")
        if np.all(gw == 0):
            raise ValueError("gene_weights equal to zero: no selected genes to plot")
        idx_mask = _as_logical(index, nstat)
        if len(gw) != int(idx_mask.sum()):
            raise ValueError("Length of gene_weights disagrees with size of set")

        one = bool(np.all(gw >= 0) or np.all(gw <= 0))

        if one:
            index2 = None
            gw1_full = np.zeros(nstat)
            gw1_full[idx_mask] = gw
            index = gw1_full != 0
            gene_weights = gw1_full[index]
        else:
            gw12 = np.zeros(nstat)
            gw12[idx_mask] = gw
            index = gw12 > 0
            index2 = gw12 < 0
            gene_weights = gw12[index]
            gene_weights2 = gw12[index2]

    TWO = index2 is not None and np.any(_as_logical(index2, nstat))

    idx_mask = _as_logical(index, nstat)
    set1 = {
        "idx": idx_mask,
        "weight": np.full(nstat, np.nan),
        "wt": np.full(nstat, np.nan),
    }
    if TWO:
        idx2_mask = _as_logical(index2, nstat)
        set2 = {
            "idx": idx2_mask,
            "weight": np.full(nstat, np.nan),
            "wt": np.full(nstat, np.nan),
        }

    if gene_weights is not None:
        set1["weight"] = np.zeros(nstat)
        set1["weight"][idx_mask] = gene_weights
        set1["wt"] = np.abs(set1["weight"]) / np.sum(np.abs(set1["weight"]))
        if TWO:
            set2["weight"] = np.zeros(nstat)
            set2["weight"][idx2_mask] = gene_weights2
            total = np.sum(np.abs(set1["weight"])) + np.sum(np.abs(set2["weight"]))
            set1["wt"] = np.abs(set1["weight"]) / total
            set2["wt"] = np.abs(set2["weight"]) / total

    # Sort ascending (R default, decreasing=FALSE)
    order = np.argsort(stat, kind="stable")
    stat = stat[order]
    for k in set1:
        set1[k] = set1[k][order]
    if TWO:
        for k in set2:
            set2[k] = set2[k][order]

    n = int(np.sum(~np.isnan(stat)))
    if n == 0:
        warnings.warn("No valid statistics", UserWarning)
        if ax is None:
            _, ax = plt.subplots()
        return ax
    stat = stat[:n]
    for k in set1:
        set1[k] = set1[k][:n]
    if TWO:
        for k in set2:
            set2[k] = set2[k][:n]

    r = np.where(set1["idx"])[0]
    if TWO:
        r2 = np.where(set2["idx"])[0]
        if len(r2) == 0:
            TWO = False
    if len(r) == 0:
        if TWO:
            r = r2
            set1 = set2
            TWO = False
        else:
            warnings.warn("No selected genes to plot", UserWarning)
            if ax is None:
                _, ax = plt.subplots()
            return ax

    WTS = False
    wt1 = set1["wt"][r]
    len_up = np.array([1.0])
    if not np.any(np.isnan(wt1)):
        max_abs_w = np.max(np.abs(set1["weight"][r]))
        if max_abs_w > 0:
            len_up = set1["weight"][r] / max_abs_w
        if not TWO:
            if wt1.size >= 2 and (np.max(wt1) > np.min(wt1)):
                WTS = True
        if TWO:
            wt12 = np.concatenate([set1["wt"][r], np.abs(set2["wt"][r2])])
            if wt12.size >= 2 and (np.max(wt12) > np.min(wt12)):
                WTS = True
            max_wt = max(np.max(set1["wt"][r]), np.max(set2["wt"][r2]))
            len_up = set1["wt"][r] / max_wt
            # TODO: confirm whether _len_down should be used (R barcodeplot.R:188
            # passes len.down to segments() drawing the second-set bars; the
            # corresponding plotting branch is not yet ported)
            _len_down = set2["wt"][r2] / max_wt

    pos_dir = bool(np.all(len_up > 0))
    shift = 0.1 if WTS else 0.0

    if col_bars is None:
        col_bars = ("red", "blue") if TWO else ("black",)
    else:
        if not isinstance(col_bars, (list, tuple, np.ndarray)):
            col_bars = (col_bars,)

    ylim_worm = [-2.1, 2.1] if worm else [-1.0, 1.0]
    if not TWO:
        ylim_worm = [0.0, 2.1] if worm else [0.0, 1.0]

    # TODO: confirm whether _ylim should be used (R barcodeplot.R:283 derives
    # barlim from ylim for the bar-rectangle drawing, which is not yet ported)
    _ylim = [-1.0, 1.5]
    if TWO:
        _ylim = [-1.5, 1.5]

    if ax is None:
        _, ax = plt.subplots()
    ax.set_xlim(0, n)
    if TWO:
        ax.set_ylim(ylim_worm[0] - shift, ylim_worm[1] + shift)
    else:
        ax.set_ylim(ylim_worm[0] - shift * (not pos_dir), ylim_worm[1] + shift * pos_dir)

    # TODO: confirm whether _npos / _nneg should be used (R barcodeplot.R:290-292
    # uses these to draw coloured background rectangles; not yet ported)
    _npos = int(np.sum(stat > quantiles[1]))
    _nneg = int(np.sum(stat < quantiles[0]))

    # Compute worm (main numeric substrate output)
    worm1 = None
    # TODO: confirm whether _worm2 should be used (R barcodeplot.R:344,356,386
    # compute worm2 and plot worm2.scale; the second-set worm-plot branch is
    # not yet ported)
    _worm2 = None
    if worm:
        idx_sorted = set1["idx"].astype(np.float64)
        if not WTS:
            ave_enrich1 = len(r) / n
            worm1 = tricube_moving_average(idx_sorted, span=span_worm) / ave_enrich1
            if TWO:
                ave_enrich2 = len(r2) / n
                _worm2 = (
                    tricube_moving_average(-set2["idx"].astype(np.float64), span=span_worm)
                    / ave_enrich2
                )
        else:
            ave_enrich1 = np.mean(set1["wt"])
            worm1 = tricube_moving_average(set1["wt"], span=span_worm) / ave_enrich1
            if TWO:
                ave_enrich2 = np.mean(set2["wt"])
                _worm2 = tricube_moving_average(-set2["wt"], span=span_worm) / ave_enrich2

        def rescale(x, new_lo, new_hi, old_lo, old_hi):
            if old_hi == old_lo:
                return np.full_like(np.asarray(x, dtype=np.float64), new_lo)
            return new_lo + (x - old_lo) / (old_hi - old_lo) * (new_hi - new_lo)

        max_w1 = float(np.max(worm1))
        worm1_scale = rescale(worm1, 1.1 + shift * pos_dir, 2.1 + shift * pos_dir, 0.0, max_w1)
        ax.plot(np.arange(1, n + 1), worm1_scale, color=col_bars[0], linewidth=2)

    ax.set_xlabel(xlab)
    return ax


def _as_logical(index, n):
    """Coerce R-style index (logical or integer) to a boolean mask of length n."""
    arr = np.asarray(index)
    if arr.dtype == bool:
        if len(arr) != n:
            out = np.zeros(n, dtype=bool)
            out[: len(arr)] = arr
            return out
        return arr.copy()
    mask = np.zeros(n, dtype=bool)
    # Integer positions: R uses 1-based positive indices
    pos = arr.astype(np.int64)
    # Accept either 1-based (R) or 0-based (Python) via heuristic:
    # if min > 0 and max <= n, assume 1-based.
    if len(pos) and pos.min() >= 1 and pos.max() <= n:
        pos = pos - 1
    mask[pos] = True
    return mask


# ===========================================================================
# Additional port additions (Phase E.5 of the bug-hunt follow-up plan)
# ===========================================================================


def plotlines(
    x,
    first_column_origin: bool = False,
    xlab: str = "Column",
    ylab: str = "x",
    col="black",
    lwd: float = 1.0,
    ax=None,
    **kwargs,
):
    """
    Time-course-style per-row line plot.

    Port of R limma's ``plotlines`` (``plotlines.R``). Draws one line
    per row of ``x`` along the columns.
    """
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if first_column_origin:
        x = x - x[:, 0:1]
    ngenes, ntime = x.shape
    if ax is None:
        _fig, ax = plt.subplots()
    if isinstance(col, (list, tuple, np.ndarray)) and len(col) == ngenes:
        colours = list(col)
    else:
        colours = [col] * ngenes
    t = np.arange(1, ntime + 1)
    for i in range(ngenes):
        ax.plot(t, x[i, :], color=colours[i], linewidth=lwd, **kwargs)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    return ax


def mdplot(
    x, columns=(0, 1), xlab: str = "Mean", ylab: str = "Difference", main=None, ax=None, **kwargs
):
    """
    Mean-difference plot of two columns of a matrix.

    Port of R limma's ``mdplot`` (``plotMD.R``). pylimma uses 0-based
    column indexing: pass ``columns=(0, 1)`` for R's ``c(1, 2)``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("mdplot requires a 2-D matrix")
    i, j = int(columns[0]), int(columns[1])
    d = x[:, j] - x[:, i]
    m = (x[:, i] + x[:, j]) / 2.0
    if main is None:
        main = f"Column {j + 1} vs Column {i + 1}"
    return plot_with_highlights(x=m, y=d, xlab=xlab, ylab=ylab, main=main, ax=ax, **kwargs)


def heat_diagram(
    results,
    coef,
    primary: int = 0,
    names=None,
    orientation: str = "landscape",
    limit=None,
    ncolors: int = 123,
    ax=None,
):
    """
    Heat diagram of fold-changes across conditions.

    Port of R limma's ``heatDiagram`` (``plots-fit.R``). ``primary`` is
    0-based.
    """
    import matplotlib.pyplot as plt

    results = np.asarray(results, dtype=np.float64)
    results = np.where(np.isnan(results), 0.0, results)
    coef_mat = np.asarray(coef, dtype=np.float64).copy()
    if results.shape != coef_mat.shape:
        raise ValueError("results and coef must be the same size")

    DE = np.abs(results[:, primary]) > 0.5
    if not DE.any():
        import warnings as _w

        _w.warn("Nothing significant to plot")
        return None

    results = results[DE]
    coef_mat = coef_mat[DE]
    coef_mat[np.abs(results) < 0.5] = np.nan
    if names is None:
        names = [str(i + 1) for i in range(coef_mat.shape[0])]
    else:
        original_names = list(names)
        names = [str(n)[:15] for i, n in enumerate(original_names) if DE[i]]

    if limit is not None and limit > 0:
        coef_mat = np.clip(coef_mat, -limit, limit)

    order = np.argsort(-coef_mat[:, primary])
    coef_mat = coef_mat[order]
    names = [names[i] for i in order]

    if ax is None:
        _fig, ax = plt.subplots()
    cmap = plt.get_cmap("RdYlGn_r", ncolors)
    data_to_plot = coef_mat if orientation == "portrait" else coef_mat.T
    im = ax.imshow(data_to_plot, aspect="auto", cmap=cmap)
    return {"coef": coef_mat, "names": names, "im": im, "ax": ax}


def plot_rldf(
    y,
    design=None,
    z=None,
    nprobes: int = 100,
    plot: bool = True,
    labels_y=None,
    labels_z=None,
    pch_y=None,
    pch_z=None,
    col_y="black",
    col_z="black",
    show_dimensions=(0, 1),
    ndim=None,
    var_prior=None,
    df_prior=None,
    trend: bool = False,
    robust: bool = False,
    ax=None,
):
    """
    Regularised linear discriminant function plot.

    Port of R limma's ``plotRLDF`` (``plotrldf.R``). ``show_dimensions``
    is 0-based; pass ``(0, 1)`` for R's ``c(1, 2)``.

    Parameters
    ----------
    y : ndarray
        Expression matrix (n_probes, n_samples).
    design : ndarray, optional
        Design matrix (n_samples, p). If None, built from ``labels_y``.
    z : ndarray, optional
        Second expression matrix of the same probe dimension; scored
        with the same metagenes as ``y``.
    nprobes : int, default 100
        Number of top probes by moderated F.
    plot : bool, default True
        Emit a scatter of the two selected discriminant functions.
    labels_y, labels_z : array-like, optional
        Sample labels for y/z; used as scatter annotations.
    pch_y, pch_z : str or int, optional
        Matplotlib marker for scatter (character-style R pch is mapped
        loosely).
    col_y, col_z : str, default "black"
        Colours for scatter points.
    show_dimensions : tuple, default (0, 1)
        Zero-based discriminant indices to plot (R: 1-based).
    ndim : int, optional
        Number of discriminant functions to retain. Defaults to the
        max of ``show_dimensions`` + 1.
    var_prior, df_prior : array-like, optional
        Hyperparameters for the within-gene variance shrinkage. When
        None, estimated via ``squeeze_var``.
    trend : bool, default False
        Pass the rowwise mean of y to ``squeeze_var`` as a covariate.
    robust : bool, default False
        Use robust variance shrinkage.
    ax : matplotlib Axes, optional
        Target axes for the plot.

    Returns
    -------
    dict with keys training (n_samples, ndim) discriminant scores,
    top (indices of the selected probes), metagenes (n_probes_kept, ndim),
    singular_values, rank, var_prior, df_prior. If ``z`` is given,
    ``predicting`` is also populated. Backwards-compatible aliases
    ``training_scores``, ``top_probes`` and ``test_scores`` are kept.
    """
    import matplotlib.pyplot as plt

    from .squeeze_var import squeeze_var

    y = np.asarray(y, dtype=np.float64)
    g, n = y.shape
    if design is None:
        if labels_y is None:
            raise ValueError("groups not specified")
        labs = np.asarray(labels_y)
        if len(np.unique(labs)) == len(labs):
            raise ValueError("design not specified and all labels_y are different")
        # Build ~ f design
        levels, inv = np.unique(labs, return_inverse=True)
        design = np.column_stack(
            [
                np.ones(n),
                (inv[:, None] == np.arange(1, len(levels))).astype(float),
            ]
        )
    design = np.asarray(design, dtype=np.float64)
    if design.shape[0] != n:
        raise ValueError("nrow(design) doesn't match ncol(y)")
    if nprobes < 1:
        raise ValueError("'nprobes' must be at least 1")
    if ndim is None:
        ndim = max(show_dimensions) + 1

    # Project onto between / within spaces via full QR of design
    # (plotrldf.R:59-65). U = Q.T @ y.T shape (n, g). Drop first column
    # as intercept. UB is rows 2:p of U (between space, p-1 rows). UW
    # is rows p+1:n (residual/within space, n-p rows).
    Q_full, R_full = np.linalg.qr(design, mode="complete")
    # R rank = number of non-zero diag entries
    diag_R = np.abs(np.diag(R_full)) if R_full.ndim == 2 else np.array([abs(R_full[0, 0])])
    rank_d = int(np.sum(diag_R > 1e-10))
    p = rank_d
    df_residual = n - p
    if df_residual == 0:
        raise ValueError("No residual degrees of freedom")
    U = Q_full.T @ y.T  # (n, g)
    UB = U[1:p, :]  # rows 2:p, shape (p-1, g); R's 1-based U[2:p,] -> Python U[1:p,]
    UW = U[p:, :]  # rows p+1:n, shape (n-p, g)
    s2 = np.mean(UW**2, axis=0)  # (g,)

    # Prior variance / df (plotrldf.R:68-79)
    if var_prior is None or df_prior is None:
        covariate = np.mean(y, axis=1) if trend else None
        sv = squeeze_var(s2, df=float(df_residual), covariate=covariate, robust=robust)
        var_prior = sv["var_prior"]
        df_prior = sv["df_prior"]
    var_prior = np.atleast_1d(np.asarray(var_prior, dtype=np.float64))
    df_prior = np.atleast_1d(np.asarray(df_prior, dtype=np.float64))
    df_prior = np.minimum(df_prior, (g - 1) * df_residual)
    df_prior = np.maximum(df_prior, 1.0)
    # Broadcast length-1 priors over the probe axis.
    if var_prior.size == 1 and g > 1:
        var_prior = np.full(g, var_prior.item())
    if df_prior.size == 1 and g > 1:
        df_prior = np.full(g, df_prior.item())

    # Select top probes by moderated F (plotrldf.R:82-95)
    if g > nprobes:
        modF = np.mean(UB**2, axis=0) / (s2 + df_prior * var_prior)
        order = np.argsort(-modF, kind="stable")
        top = order[:nprobes]
        y_sel = y[top, :]
        z_sel = None if z is None else np.asarray(z, dtype=np.float64)[top, :]
        UB_sel = UB[:, top]
        UW_sel = UW[:, top]
        if df_prior.size > 1:
            df_prior = df_prior[top]
        if var_prior.size > 1:
            var_prior = var_prior[top]
        g = nprobes
    else:
        top = np.arange(min(g, nprobes))
        y_sel = y
        z_sel = None if z is None else np.asarray(z, dtype=np.float64)
        UB_sel = UB
        UW_sel = UW

    # Within-group SS and regularised within covariance
    # (plotrldf.R:97-110).
    W = UW_sel.T @ UW_sel  # (g, g)
    Wreg = W.copy()
    diag_reg = np.diag(Wreg) + df_prior * var_prior
    np.fill_diagonal(Wreg, diag_reg)
    df_total = df_prior + df_residual
    if df_total.size > 1:
        df_total_sqrt = np.sqrt(df_total)
        Wreg = Wreg / df_total_sqrt[:, None]
        Wreg = Wreg.T / df_total_sqrt[:, None]
    else:
        Wreg = Wreg / float(df_total[0])

    # Cholesky + backsolve (plotrldf.R:112-113).
    # R: WintoB <- backsolve(chol(Wreg), t(UB), transpose=TRUE). R's
    # chol returns upper-triangular R_u with Wreg = R_u^T R_u; numpy's
    # cholesky returns lower-triangular L with Wreg = L L^T, and
    # L = R_u^T. backsolve(R_u, x, transpose=TRUE) solves R_u^T y = x,
    # i.e. L y = x in Python. t(UB) has shape (g, p-1), so WintoB
    # has shape (g, p-1) - matching the metagenes axis expected below.
    L = np.linalg.cholesky(Wreg)
    WintoB = np.linalg.solve(L, UB_sel.T)  # shape (g, p-1)
    # SVD: columns of U are metagenes (per-gene), singular values are
    # the discriminant strengths.
    U_svd, s_vals, _ = np.linalg.svd(WintoB, full_matrices=False)
    metagenes = U_svd[:, :ndim]

    # LDF scores for the training set: d.y = t(y) %*% metagenes
    # (plotrldf.R:125). t(y) is (n, g) so d_y has shape (n, ndim).
    d_y = y_sel.T @ metagenes
    rank_out = min(WintoB.shape)

    result = {
        "training": d_y,
        "training_scores": d_y[:, list(show_dimensions)],
        "top": np.asarray(top),
        "top_probes": np.asarray(top),
        "metagenes": metagenes,
        "singular_values": s_vals,
        "rank": rank_out,
        "var_prior": var_prior,
        "df_prior": df_prior,
    }
    if z_sel is not None:
        d_z = z_sel.T @ metagenes
        result["predicting"] = d_z
        result["test_scores"] = d_z[:, list(show_dimensions)]

    if plot:
        if ax is None:
            _fig, ax = plt.subplots()
        d1 = show_dimensions[0]
        d2 = show_dimensions[1]
        if pch_y is None and labels_y is not None:
            for xi, yi, lab in zip(d_y[:, d1], d_y[:, d2], labels_y):
                ax.text(xi, yi, str(lab), color=col_y)
        else:
            ax.scatter(
                d_y[:, d1], d_y[:, d2], c=col_y, marker=str(pch_y) if pch_y is not None else "o"
            )
        if z_sel is not None:
            d_z = result["predicting"]
            if pch_z is None and labels_z is not None:
                for xi, yi, lab in zip(d_z[:, d1], d_z[:, d2], labels_z):
                    ax.text(xi, yi, str(lab), color=col_z)
            else:
                ax.scatter(
                    d_z[:, d1], d_z[:, d2], c=col_z, marker=str(pch_z) if pch_z is not None else "x"
                )
        ax.set_xlabel(f"Discriminant Function {d1 + 1}")
        ax.set_ylabel(f"Discriminant Function {d2 + 1}")
    return result


def plot_exons(
    fit,
    coef=None,
    geneid=None,
    genecolname: str = "GeneID",
    exoncolname=None,
    rank: int = 1,
    fdr: float = 0.05,
    ax=None,
):
    """
    Plot log-fold-changes of exons for a single gene.

    Port of R limma's ``plotExons`` (``plotExons.R``).
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    from .utils import p_adjust

    if fit.get("p_value") is None:
        raise ValueError("fit must contain p_value from e_bayes")
    genes = fit.get("genes")
    if genes is None:
        raise ValueError("fit must contain genes")
    if not isinstance(genes, pd.DataFrame):
        genes = pd.DataFrame(genes)
    if genecolname not in genes.columns:
        raise ValueError(f"genecolname {genecolname} not found")
    coef_idx = fit["coefficients"].shape[1] - 1 if coef is None else int(coef)
    p_col = np.asarray(fit["p_value"])[:, coef_idx]
    fc = np.asarray(fit["coefficients"])[:, coef_idx]

    if geneid is None:
        if rank == 1:
            gi = int(np.argmin(p_col))
        else:
            order = np.argsort(p_col)
            gi = int(order[rank - 1])
        gene = str(genes.iloc[gi][genecolname])
    else:
        gene = str(geneid)
    mask = genes[genecolname].astype(str).values == gene
    if not mask.any():
        raise ValueError(f"gene {gene} not found")

    adj = p_adjust(p_col, method="BH")
    de_exon = adj[mask] < fdr
    x = np.arange(int(mask.sum()))
    y = fc[mask]
    if ax is None:
        _fig, ax = plt.subplots()
    ax.scatter(x[~de_exon], y[~de_exon], color="grey")
    ax.scatter(x[de_exon], y[de_exon], color="red")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("Exon")
    ax.set_ylabel("log2 FC")
    ax.set_title(gene)
    return ax


def plot_exon_junc(
    fit, coef=None, geneid=None, genecolname=None, fdr: float = 0.05, annotation=None, ax=None
):
    """
    Exon / junction plot for a diff_splice fit.

    Port of R limma's ``plotExonJunc`` (``plotExonJunc.R``). Junctions
    are distinguished from exons by ``genes["Length"] == 1``.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    from .utils import p_adjust

    if genecolname is None:
        genecolname = fit.get("genecolname")
    if genecolname is None:
        raise ValueError("genecolname must be provided")
    genes = fit.get("genes")
    if genes is None:
        raise ValueError("fit must contain genes")
    if not isinstance(genes, pd.DataFrame):
        genes = pd.DataFrame(genes)

    gene = str(geneid)
    mask = genes[genecolname].astype(str).values == gene
    if not mask.any():
        raise ValueError(f"{gene} not found.")
    coef_idx = fit["coefficients"].shape[1] - 1 if coef is None else int(coef)
    p_col = np.asarray(fit["p_value"])[:, coef_idx]
    fc = np.asarray(fit["coefficients"])[:, coef_idx]

    adj = p_adjust(p_col, method="BH")
    genes_g = genes[mask].reset_index(drop=True)
    adj_g = adj[mask]
    fc_g = fc[mask]

    length = genes_g.get("Length")
    if length is None:
        raise ValueError("genes must contain Length to distinguish exons from junctions")
    is_exon = length.values > 1
    if ax is None:
        _fig, ax = plt.subplots()
    ax.scatter(
        np.where(is_exon)[0],
        fc_g[is_exon],
        color=np.where(adj_g[is_exon] < fdr, "red", "grey"),
        marker="s",
        label="Exon",
    )
    ax.scatter(
        np.where(~is_exon)[0],
        fc_g[~is_exon],
        color=np.where(adj_g[~is_exon] < fdr, "orange", "black"),
        marker="x",
        label="Junction",
    )
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("Feature")
    ax.set_ylabel("log2 FC")
    ax.legend()
    return ax


def plot_ma_3by2(
    object,
    prefix: str = "MA",
    path: str | None = None,
    main: list[str] | None = None,
    zero_weights: bool = False,
    common_lim: bool = True,
    device: str = "png",
    **kwargs,
) -> list[str]:
    """
    Write MA plots to disk, six per page in a 3x2 grid.

    Port of R limma's ``plotMA3by2``. Writes one file per page; for a
    matrix with ``n_arrays`` columns the function emits
    ``ceil(n_arrays / 6)`` files named
    ``<path>/<prefix>-<i1>-<i2>.<ext>``.

    Parameters
    ----------
    object : ndarray, EList, or MArrayLM
        Matrix-like or pylimma fit. ``ndarray`` and ``EList`` go through
        ``plot_ma``'s matrix path; ``MArrayLM`` plots ``coefficients``
        column by column. RGList / MAList / EListRaw inputs (R's
        two-colour S4 wrappers) are out of scope.
    prefix : str, default "MA"
        Filename prefix. Files are written as
        ``{path}/{prefix}-{i1}-{i2}.{ext}`` where ``i1``..``i2`` are
        1-based array indices for that page (matching R).
    path : str, optional
        Output directory. Defaults to the current working directory.
    main : list of str, optional
        Per-array title. Defaults to column names where available.
    zero_weights : bool, default False
        Forwarded to ``plot_ma``.
    common_lim : bool, default True
        If True, all pages share the same x and y limits computed from
        the full matrix (matching R's default).
    device : str, default "png"
        Output format. One of ``"png"``, ``"jpeg"``, ``"pdf"``,
        ``"postscript"`` (passed to matplotlib's ``savefig``; matplotlib
        chooses the backend from the file extension).
    **kwargs
        Forwarded to ``plot_ma``.

    Returns
    -------
    list of str
        Absolute paths of the written files (in page order).
    """
    import os

    import matplotlib.pyplot as plt

    if device not in ("png", "jpeg", "pdf", "postscript"):
        raise ValueError(
            f"device {device!r} not recognised; must be one of 'png', 'jpeg', 'pdf', 'postscript'"
        )
    ext = "ps" if device == "postscript" else device
    if path is None:
        path = "."

    # Resolve substrate matrix and per-column labels.
    if isinstance(object, MArrayLM):
        if object.get("coefficients") is None:
            raise ValueError("fit must contain coefficients")
        if object.get("Amean") is None:
            raise ValueError("Amean component is absent.")
        coefs = np.asarray(object["coefficients"])
        if coefs.ndim == 1:
            coefs = coefs[:, np.newaxis]
        n_arrays = coefs.shape[1]
        col_names = object.get("contrast_names") or object.get("coef_names")
    else:
        E, _ = _get_matrix_E(object, want_weights=True)
        n_arrays = E.shape[1]
        col_names = None
        if hasattr(object, "var_names"):
            pass  # AnnData has var_names but we want sample/array names
        if hasattr(object, "obs_names"):
            col_names = list(getattr(object, "obs_names"))

    if main is None:
        main = (
            list(col_names)
            if col_names is not None
            else [f"Array {i + 1}" for i in range(n_arrays)]
        )
    elif len(main) != n_arrays:
        raise ValueError(f"len(main)={len(main)} does not match n_arrays={n_arrays}")

    # Common x/y limits computed once over the whole matrix.
    common_xlim = common_ylim = None
    if common_lim:
        if isinstance(object, MArrayLM):
            x_all = np.asarray(object["Amean"], dtype=np.float64)
            y_all = coefs
        else:
            E, w = _get_matrix_E(object, want_weights=True)
            # A and M for each column: A_i = mean of remaining cols,
            # M_i = E[:, i] - A_i. Vectorised.
            n = E.shape[1]
            row_sum = np.nansum(E, axis=1, keepdims=True)
            A_all = (row_sum - E) / max(n - 1, 1)
            M_all = E - A_all
            if not zero_weights and w is not None:
                M_all = np.where(w > 0, M_all, np.nan)
            x_all = A_all
            y_all = M_all
        common_xlim = (np.nanmin(x_all), np.nanmax(x_all))
        common_ylim = (np.nanmin(y_all), np.nanmax(y_all))

    written: list[str] = []
    n_pages = -(-n_arrays // 6)  # ceil
    for ipage in range(n_pages):
        i1 = ipage * 6 + 1
        i2 = min((ipage + 1) * 6, n_arrays)

        # R uses width=6.5 inches, height=10 inches for vector
        # devices; PNG/JPEG bumped to 6.5*140 x 10*140 px @ default
        # ~72 dpi historic legacy. We pin the PDF/PS aspect and let
        # matplotlib's savefig handle DPI for raster outputs.
        fig, axes = plt.subplots(3, 2, figsize=(6.5, 10.0))
        axes = axes.flatten()

        for slot in range(6):
            array_idx = ipage * 6 + slot
            ax = axes[slot]
            if array_idx >= n_arrays:
                ax.set_axis_off()
                continue
            plot_ma(
                object,
                array=array_idx,
                coef=array_idx if isinstance(object, MArrayLM) else None,
                main=main[array_idx],
                zero_weights=zero_weights,
                ax=ax,
                **kwargs,
            )
            if common_lim:
                ax.set_xlim(common_xlim)
                ax.set_ylim(common_ylim)

        out_file = os.path.join(path, f"{prefix}-{i1}-{i2}.{ext}")
        fig.savefig(out_file, format=ext if ext != "ps" else "ps")
        plt.close(fig)
        written.append(out_file)

    return written

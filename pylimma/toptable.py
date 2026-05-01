# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   toptable.R                 Copyright (C) 2003-2023 Gordon Smyth
#   topTableF.R                Copyright (C) 2006-2022 Gordon Smyth
# Python port: Copyright (C) 2026 John Mulvey
"""
Top table output for pylimma.

Implements:
- top_table(): extract and format results as a DataFrame
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from .utils import p_adjust
from .classes import _resolve_fit_input

if TYPE_CHECKING:
    from anndata import AnnData


def _handle_duplicated_rownames(
    genes: list, genelist_df: pd.DataFrame | None
) -> tuple[list, pd.DataFrame | None]:
    """
    Handle duplicated gene names following R limma's logic.

    If gene names have duplicates:
    - Move original names to genelist_df as 'ID' column (or 'ID0' if 'ID' exists)
    - Replace gene names with 1-indexed integers

    Parameters
    ----------
    genes : list
        Gene names/identifiers
    genelist_df : DataFrame or None
        Gene annotation DataFrame

    Returns
    -------
    tuple
        (genes, genelist_df) - updated gene names and genelist DataFrame
    """
    # Check for duplicates
    if len(genes) != len(set(genes)):
        # Duplicates found - move original names to a column
        if genelist_df is None:
            genelist_df = pd.DataFrame({"ID": genes})
        elif "ID" in genelist_df.columns:
            genelist_df = genelist_df.copy()
            genelist_df["ID0"] = genes
        else:
            genelist_df = genelist_df.copy()
            genelist_df["ID"] = genes
        # Replace with 1-indexed integers
        genes = [i + 1 for i in range(len(genes))]

    return genes, genelist_df


def top_table(
    data,
    coef: int | str | list | None = None,
    number: int = 10,
    genelist: pd.DataFrame | list | np.ndarray | None = None,
    adjust_method: str = "BH",
    sort_by: str = "B",
    resort_by: str | None = None,
    p_value: float = 1.0,
    fc: float | None = None,
    lfc: float = 0.0,
    confint: bool | float = False,
    key: str = "pylimma",
) -> pd.DataFrame:
    """
    Extract a table of top-ranked genes from a linear model fit.

    Parameters
    ----------
    data : AnnData or dict
        Either an AnnData object with fit results in adata.uns[key],
        or a dict from e_bayes().
    coef : int, str, list, or None
        Coefficient(s) to extract. If None or multiple coefficients,
        returns F-statistic results. If single coefficient (int or str),
        returns t-statistic results for that coefficient.
    number : int, default 10
        Maximum number of genes to return. Use np.inf for all genes.
    genelist : DataFrame, list, or array, optional
        Gene annotations to include in output. If a DataFrame, columns
        are merged with results. If a list/array, used as gene identifiers.
        If None (default), uses fit["genes"] if available.

        For AnnData input, fit["genes"] only carries var_names; annotation
        columns in ``adata.var`` (e.g. ``symbol``, ``chromosome``) are not
        duplicated into the fit. Pass ``genelist=adata.var`` explicitly
        to merge those columns into the output table.
    adjust_method : str, default "BH"
        Multiple testing adjustment method. Options: "BH", "bonferroni",
        "holm", "none".
    sort_by : str, default "B"
        Column to sort by. Options:
        - "B" or "b": B-statistic (log-odds of DE)
        - "p" or "P": p-value
        - "t" or "T": t-statistic (absolute value)
        - "logFC" or "log_fc": log fold-change (absolute value)
        - "AveExpr" or "ave_expr": average expression
        - "F": F-statistic (for multiple coefficients)
        - "none": no sorting
    resort_by : str, optional
        Secondary sort column applied after filtering by p_value/lfc.
        Same options as sort_by. Useful when you want to filter by
        p-value but display sorted by fold-change.
    p_value : float, default 1.0
        Adjusted p-value cutoff. Only genes with adj_p_value <= p_value
        are returned.
    fc : float, optional
        Minimum fold-change required (linear scale, must be >= 1).
        Only genes with fold-change >= fc are returned. This is an
        alternative to `lfc`; if both are specified, `fc` takes precedence.
    lfc : float, default 0.0
        Log2 fold-change cutoff. Only genes with \|log_fc\| >= lfc are
        returned. Equivalent to log2(fc).
    confint : bool or float, default False
        If True, compute 95% confidence intervals for log fold-changes.
        If a float, specifies the confidence level (e.g., 0.99 for 99%).
        Only applies for single coefficient tests.
    key : str, default "pylimma"
        Key for fit results in adata.uns (AnnData input only).

    Returns
    -------
    DataFrame
        Table of top genes with columns:
        - gene: gene identifier (if available)
        - log_fc: log fold-change
        - ci_l, ci_r: confidence interval bounds (if confint=True)
        - ave_expr: average expression
        - t: moderated t-statistic
        - p_value: raw p-value
        - adj_p_value: adjusted p-value
        - b: B-statistic (log-odds)

        For multiple coefficients (F-test):
        - One log_fc column per coefficient
        - F: F-statistic
        - p_value: F-test p-value
        - adj_p_value: adjusted p-value

    Notes
    -----
    This function must be called after e_bayes(). The results are sorted
    and filtered according to the specified parameters.

    Examples
    --------
    >>> fit = lm_fit(expr, design)
    >>> fit = e_bayes(fit)
    >>> top_table(fit, coef=1, number=20)  # Top 20 genes for coefficient 1
    >>> top_table(fit, coef=None)  # F-test across all coefficients
    >>> top_table(fit, coef=1, confint=True)  # With 95% CI
    """
    fit, _adata, _adata_key = _resolve_fit_input(data, key)

    # Validate fit
    if "t" not in fit and "F" not in fit:
        raise ValueError("Need to run e_bayes() first")
    if "coefficients" not in fit:
        raise ValueError("coefficients not found in fit")

    coefficients = fit["coefficients"]
    n_genes, n_coefs = coefficients.shape

    # Determine which coefficients to use
    is_treat = "treat_lfc" in fit

    # R's topTreat defaults to sort.by="p", unlike topTable which defaults to "B"
    # Match this behaviour when treat results are detected
    if is_treat and sort_by == "B":
        sort_by = "P"
    if coef is None:
        if is_treat:
            # treat results: default to last coefficient only (R parity)
            coef_idx = [n_coefs - 1]
        elif n_coefs > 1:
            # Use all coefficients, but remove intercept if present (like R)
            coef_idx = list(range(n_coefs))
            coef_names_check = fit.get("contrast_names") or fit.get("coef_names")
            if coef_names_check is not None:
                try:
                    intercept_idx = coef_names_check.index("(Intercept)")
                    coef_idx.remove(intercept_idx)
                    import warnings
                    warnings.warn("Removing intercept from test coefficients")
                except ValueError:
                    pass  # No intercept found
        else:
            coef_idx = [n_coefs - 1]  # Use last coefficient (like R)
    elif isinstance(coef, (list, tuple)):
        coef_idx = list(coef)
    else:
        coef_idx = [coef]

    # Convert string coefficient/contrast names to indices
    # Use contrast_names if available (after contrasts_fit), otherwise coef_names
    coef_names = fit.get("contrast_names") or fit.get("coef_names")
    if coef_names is not None:
        new_coef_idx = []
        for c in coef_idx:
            if isinstance(c, str):
                if c not in coef_names:
                    raise ValueError(
                        f"Coefficient/contrast '{c}' not found. "
                        f"Available: {coef_names}"
                    )
                new_coef_idx.append(coef_names.index(c))
            else:
                new_coef_idx.append(c)
        coef_idx = new_coef_idx

    # Get genelist from fit if not provided. Track whether the value
    # came from the user (explicit) or from `fit["genes"]` (auto-default
    # from rownames in pylimma). R only wraps explicit vectors into an
    # ID column (toptable.R:168); rownames are kept as row index.
    genelist_explicit = genelist is not None
    if genelist is None:
        genelist = fit.get("genes")

    # Handle fc parameter (convert to lfc, fc takes precedence)
    if fc is not None:
        if fc < 1:
            raise ValueError("fc must be >= 1 (fold-change cannot be less than 1)")
        lfc = np.log2(fc)

    # Dispatch to appropriate function
    if len(coef_idx) > 1:
        if is_treat:
            raise ValueError(
                "Treat p-values can only be displayed for single coefficients. "
                "Specify a single coef or use e_bayes() instead of treat()."
            )
        if confint:
            import warnings
            warnings.warn("confint is ignored for F-test (multiple coefficients)")
        # R toptable.R:46: `if(sort.by=="B") sort.by <- "F"` - silent
        # translation when routing to .topTableF.
        if sort_by == "B":
            sort_by = "F"
        return top_table_f(
            fit,
            coef_idx=coef_idx,
            number=number,
            genelist=genelist,
            adjust_method=adjust_method,
            sort_by=sort_by,
            resort_by=resort_by,
            p_value=p_value,
            lfc=lfc,
            _genelist_explicit=genelist_explicit,
        )
    else:
        return _top_table_t(
            fit,
            coef=coef_idx[0],
            number=number,
            genelist=genelist,
            adjust_method=adjust_method,
            sort_by=sort_by,
            resort_by=resort_by,
            p_value=p_value,
            lfc=lfc,
            confint=confint,
            _genelist_explicit=genelist_explicit,
        )


def _top_table_t(
    fit: dict,
    coef: int,
    number: int,
    genelist,
    adjust_method: str,
    sort_by: str,
    resort_by: str | None,
    p_value: float,
    lfc: float,
    confint: bool | float,
    _genelist_explicit: bool = False,
) -> pd.DataFrame:
    """Top table for single coefficient (t-statistics)."""
    coefficients = fit["coefficients"]
    n_genes = coefficients.shape[0]

    # Extract statistics. R toptable.R:273-275 conditions the AveExpr
    # and B columns on slot presence (`if(!is.null(A))`,
    # `if(include.B)`). Track presence so the DataFrame builder can
    # omit the columns when their source is missing - matching R's
    # `data.frame$col <- NULL` no-op behaviour.
    log_fc = coefficients[:, coef]
    t_stat = fit["t"][:, coef]
    p_val = fit["p_value"][:, coef]
    lods = fit.get("lods")
    has_lods = lods is not None
    if has_lods:
        b_stat = lods[:, coef] if lods.ndim > 1 else lods
    else:
        # R toptable.R:209-211: missing lods + sort/resort.by="B" raises.
        if sort_by == "B":
            raise ValueError(
                "Trying to sort.by B, but B-statistic (lods) "
                "not found in MArrayLM object"
            )
        if resort_by == "B":
            raise ValueError(
                "Trying to resort.by B, but B-statistic (lods) "
                "not found in MArrayLM object"
            )
        b_stat = np.full(n_genes, np.nan)
    has_amean = "Amean" in fit and fit["Amean"] is not None
    if has_amean:
        ave_expr = np.asarray(fit["Amean"])
        # R: `if(NCOL(A)>1) A <- rowMeans(A, na.rm=TRUE)`.
        if ave_expr.ndim > 1 and ave_expr.shape[1] > 1:
            ave_expr = np.nanmean(ave_expr, axis=1)
    else:
        ave_expr = np.full(n_genes, np.nan)

    # Handle genelist. R toptable.R:168 wraps an explicit vector
    # genelist as `data.frame(ID=genelist)` so it appears as an ID
    # column. pylimma additionally auto-defaults `genelist` to
    # `fit["genes"]` (set from rownames in lm_fit) - that auto-default
    # is treated as the row index, NOT an ID column, to preserve
    # rownames flow through the pipeline.
    if genelist is None:
        genes = [f"gene{i+1}" for i in range(n_genes)]
        genelist_df = None
    elif isinstance(genelist, pd.DataFrame):
        genes = list(genelist.index) if len(genelist.index) == n_genes else list(range(n_genes))
        genelist_df = genelist.copy()
    elif isinstance(genelist, (list, np.ndarray)):
        if len(genelist) == n_genes:
            if _genelist_explicit:
                # Explicit vector → ID column, row index from fit rownames
                # (or 1..N fallback matching R toptable.R:170-172).
                genelist_df = pd.DataFrame({"ID": list(genelist)})
                genes = list(range(1, n_genes + 1))
            else:
                # Auto-default from fit["genes"] → use as row index
                genes = list(genelist)
                genelist_df = None
        else:
            genes = [f"gene{i}" for i in range(n_genes)]
            genelist_df = None
    else:
        genes = [f"gene{i+1}" for i in range(n_genes)]
        genelist_df = None

    # Handle duplicated gene names (DataFrame index must be unique)
    genes, genelist_df = _handle_duplicated_rownames(genes, genelist_df)

    # Confidence intervals
    margin_error = None
    if confint:
        stdev_unscaled = fit.get("stdev_unscaled")
        s2_post = fit.get("s2_post")
        df_total = fit.get("df_total")
        if stdev_unscaled is None or s2_post is None or df_total is None:
            raise ValueError("Need stdev_unscaled, s2_post, df_total in fit for confidence intervals")
        if isinstance(confint, float):
            alpha = (1 + confint) / 2
        else:
            alpha = 0.975  # 95% CI
        margin_error = np.sqrt(s2_post) * stdev_unscaled[:, coef] * stats.t.ppf(alpha, df_total)

    # Multiple testing adjustment
    adj_p_val = p_adjust(p_val, method=adjust_method)

    # Filter by p-value and lfc thresholds.
    # R toptable.R:233-246 gates filtering on `p.value < 1 | lfc > 0`,
    # masking NaN sig rows to FALSE only inside that block. With default
    # arguments R does no filtering and preserves NaN-p-value rows.
    keep = np.ones(n_genes, dtype=bool)
    if p_value < 1 or lfc > 0:
        if p_value < 1:
            keep &= adj_p_val <= p_value
        if lfc > 0:
            # `abs(M) >= lfc` (toptable.R:234) - inclusive boundary.
            keep &= np.abs(log_fc) >= lfc
        keep &= ~np.isnan(p_val)  # R: sig[is.na(sig)] <- FALSE

    if not np.any(keep):
        return pd.DataFrame()

    # Apply filter
    idx = np.where(keep)[0]
    log_fc = log_fc[idx]
    t_stat = t_stat[idx]
    p_val = p_val[idx]
    adj_p_val = adj_p_val[idx]
    b_stat = b_stat[idx]
    ave_expr = ave_expr[idx]
    genes = [genes[i] for i in idx]
    if genelist_df is not None:
        genelist_df = genelist_df.iloc[idx].reset_index(drop=True)
    if margin_error is not None:
        margin_error = margin_error[idx]

    # Helper function for sorting
    # use_abs: primary sort uses abs() for t and logFC; resort uses signed values (R behaviour)
    def _get_sort_order(sort_col, data_dict, default_order, use_abs=True):
        # R toptable.R:188: `sort.by="A" || sort.by=="Amean"` -> "AveExpr".
        sort_by_map = {
            "B": "b", "b": "b",
            "P": "p", "p": "p",
            "T": "t", "t": "t",
            "logFC": "logFC", "log_fc": "logFC", "M": "logFC",
            "AveExpr": "AveExpr", "ave_expr": "AveExpr",
            "A": "AveExpr", "Amean": "AveExpr",
            "none": "none",
        }
        col = sort_by_map.get(sort_col, sort_col)
        # Use stable descending sort (np.argsort on negated values with
        # kind='stable') to match R's order(-x) tie-break behaviour.
        if col == "b":
            return np.argsort(-data_dict["b"], kind="stable")
        elif col == "p":
            return np.argsort(data_dict["p"], kind="stable")
        elif col == "t":
            vals = np.abs(data_dict["t"]) if use_abs else data_dict["t"]
            return np.argsort(-vals, kind="stable")
        elif col == "logFC":
            vals = np.abs(data_dict["logFC"]) if use_abs else data_dict["logFC"]
            return np.argsort(-vals, kind="stable")
        elif col == "AveExpr":
            return np.argsort(-data_dict["AveExpr"], kind="stable")
        else:
            return default_order

    data_dict = {"b": b_stat, "p": p_val, "t": t_stat, "logFC": log_fc, "AveExpr": ave_expr}

    # Sort by primary column (uses absolute values for t and logFC)
    order = _get_sort_order(sort_by, data_dict, np.arange(len(log_fc)), use_abs=True)

    # Limit number
    if number < len(order):
        order = order[:number]

    # Apply primary sort
    log_fc = log_fc[order]
    t_stat = t_stat[order]
    p_val = p_val[order]
    adj_p_val = adj_p_val[order]
    b_stat = b_stat[order]
    ave_expr = ave_expr[order]
    genes = [genes[i] for i in order]
    if genelist_df is not None:
        genelist_df = genelist_df.iloc[order].reset_index(drop=True)
    if margin_error is not None:
        margin_error = margin_error[order]

    # Resort if requested (uses signed values for t and logFC, matching R)
    if resort_by is not None:
        data_dict = {"b": b_stat, "p": p_val, "t": t_stat, "logFC": log_fc, "AveExpr": ave_expr}
        resort_order = _get_sort_order(resort_by, data_dict, np.arange(len(log_fc)), use_abs=False)
        log_fc = log_fc[resort_order]
        t_stat = t_stat[resort_order]
        p_val = p_val[resort_order]
        adj_p_val = adj_p_val[resort_order]
        b_stat = b_stat[resort_order]
        ave_expr = ave_expr[resort_order]
        genes = [genes[i] for i in resort_order]
        if genelist_df is not None:
            genelist_df = genelist_df.iloc[resort_order].reset_index(drop=True)
        if margin_error is not None:
            margin_error = margin_error[resort_order]

    # Build DataFrame. R toptable.R:268-280 puts AveExpr / B in the
    # output only when their source slots are non-NULL. Mirror that.
    cols: dict[str, np.ndarray] = {"log_fc": log_fc}
    if confint and margin_error is not None:
        cols["ci_l"] = log_fc - margin_error
        cols["ci_r"] = log_fc + margin_error
    if has_amean:
        cols["ave_expr"] = ave_expr
    cols["t"] = t_stat
    cols["p_value"] = p_val
    cols["adj_p_value"] = adj_p_val
    if has_lods:
        cols["b"] = b_stat
    df = pd.DataFrame(cols)

    # Add genelist columns if DataFrame
    if genelist_df is not None:
        for col in genelist_df.columns:
            if col not in df.columns:
                df[col] = genelist_df[col].values

    df.index = genes
    df.index.name = "gene"

    return df


def top_table_f(
    fit: dict,
    number: int = 10,
    genelist=None,
    adjust_method: str = "BH",
    sort_by: str = "F",
    p_value: float = 1.0,
    fc: float | None = None,
    lfc: float | None = None,
    *,
    coef_idx: list[int] | None = None,
    resort_by: str | None = None,
    _genelist_explicit: bool | None = None,
) -> pd.DataFrame:
    """
    Top table for multiple coefficients ranked by F-statistic.

    Port of R's ``limma::topTableF``. Positional arguments follow R's
    order exactly; ``coef_idx`` and ``resort_by`` are pylimma-only
    extensions and keyword-only.

    Parameters
    ----------
    fit : dict
        Fit object from :func:`e_bayes` containing F-statistics.
    number : int, default 10
        Maximum number of genes to return.
    genelist : DataFrame, list, or array, optional
        Gene annotations to include in output. When None, uses
        ``fit.get("genes")`` (matches R's ``genelist=fit$genes``
        default).

        For AnnData input, fit["genes"] only carries var_names; pass
        ``genelist=adata.var`` explicitly to merge annotation columns
        (e.g. ``symbol``, ``chromosome``) into the output table.
    adjust_method : str, default "BH"
        Multiple-testing adjustment method.
    sort_by : str, default "F"
        Column to sort by. ``"F"`` or ``"none"`` (R).
    p_value : float, default 1.0
        Adjusted p-value cutoff.
    fc : float, optional
        Fold-change cutoff. If given, sets ``lfc = log2(fc)``.
        Must be >= 1.
    lfc : float, optional
        Log fold-change cutoff. Defaults to 0 when both ``fc`` and
        ``lfc`` are None (R's NULL).
    coef_idx : list of int, optional (keyword-only)
        Indices of coefficients to include. Defaults to all columns
        of ``fit["coefficients"]``, matching R's ``topTableF`` which
        always uses the whole coefficient matrix.
    resort_by : str, optional (keyword-only)
        pylimma extension: secondary sort column applied after
        ``sort_by`` + truncation.

    Returns
    -------
    DataFrame
        Table of top genes ranked by F-statistic.
    """
    # R topTableF.R:7 emits a deprecation message on every call. Mirror
    # it with DeprecationWarning so downstream tooling (pytest's
    # warning capture, IDE linters) sees an equivalent signal.
    import warnings as _warnings
    _warnings.warn(
        "top_table_f is obsolete and will be removed in a future "
        "version of pylimma. Please consider using top_table instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Resolve lfc from fc / lfc exactly as R does (toptable.R:33-39):
    # when fc is supplied, lfc is silently overridden with log2(fc) -
    # any user-supplied lfc value is discarded without warning.
    if fc is not None:
        if fc < 1:
            raise ValueError("fc must be greater than or equal to 1")
        lfc = float(np.log2(fc))
    elif lfc is None:
        lfc = 0.0

    # R topTableF.R:38: `sort.by <- match.arg(sort.by, c("F","none"))`.
    # match.arg errors on anything else; mirror that here.
    if sort_by not in ("F", "none"):
        raise ValueError(
            f"sort_by={sort_by!r} not recognised. Must be one of 'F', 'none'."
        )

    # R topTableF.R:12: `rn <- rownames(M)`. When fit["coefficients"]
    # is a DataFrame, surface its index as `coef_rownames` so duplicate
    # detection (R-B11a topTableF.R:27-28) can promote them to an ID
    # column. Plain ndarray inputs leave coef_rownames at None.
    coef_obj = fit["coefficients"]
    coef_rownames = None
    if isinstance(coef_obj, pd.DataFrame):
        coef_rownames = list(coef_obj.index)
    coef_matrix = np.asarray(coef_obj)
    if coef_idx is None:
        # R's topTableF uses the whole coefficient matrix.
        coef_idx = list(range(coef_matrix.shape[1]))
    coefficients = coef_matrix[:, coef_idx]
    n_genes = coefficients.shape[0]

    # Default genelist from fit, matching R's genelist=fit$genes.
    # _genelist_explicit is 3-state: True/False from the wrapper
    # (which has already handled auto-default), None when the user
    # invoked top_table_f directly. In the direct-call case we need
    # to auto-detect: a non-None genelist at entry IS explicit.
    if _genelist_explicit is None:
        if genelist is not None:
            _genelist_explicit = True
        else:
            _genelist_explicit = False
            genelist = fit.get("genes")
    elif genelist is None:
        # Wrapper passed an explicit flag but no genelist - this can
        # happen if the wrapper's auto-default also returned None.
        genelist = fit.get("genes")

    # F-statistics
    f_stat = fit.get("F")
    f_p_val = fit.get("F_p_value")

    if f_stat is None or f_p_val is None:
        raise ValueError("F-statistics not found. Run e_bayes() with multiple coefficients.")

    # R topTableF.R:89: `tab$AveExpr <- Amean[o]`. When Amean is NULL,
    # R's `data.frame$col <- NULL` is a no-op, so the column is absent.
    # Track absence explicitly so the DataFrame builder can omit it.
    has_amean = "Amean" in fit and fit["Amean"] is not None
    if has_amean:
        ave_expr = np.asarray(fit["Amean"])
        # R: `if(NCOL(A)>1) A <- rowMeans(A, na.rm=TRUE)`.
        if ave_expr.ndim > 1 and ave_expr.shape[1] > 1:
            ave_expr = np.nanmean(ave_expr, axis=1)
    else:
        ave_expr = np.full(n_genes, np.nan)

    # Handle genelist. R topTableF.R:85 wraps an explicit vector as
    # `data.frame(ProbeID=genelist)` so it appears as a ProbeID column.
    # pylimma's auto-default from `fit["genes"]` is used as the row
    # index instead - see `_top_table_t` for the same pattern.
    if genelist is None:
        genes = (
            list(coef_rownames) if coef_rownames is not None
            else [f"gene{i+1}" for i in range(n_genes)]
        )
        genelist_df = None
    elif isinstance(genelist, pd.DataFrame):
        genes = list(genelist.index) if len(genelist.index) == n_genes else list(range(n_genes))
        genelist_df = genelist.copy()
    elif isinstance(genelist, (list, np.ndarray)):
        if len(genelist) == n_genes:
            if _genelist_explicit:
                genelist_df = pd.DataFrame({"ProbeID": list(genelist)})
                genes = list(range(1, n_genes + 1))
            else:
                genes = list(genelist)
                genelist_df = None
        else:
            genes = [f"gene{i}" for i in range(n_genes)]
            genelist_df = None
    else:
        genes = [f"gene{i+1}" for i in range(n_genes)]
        genelist_df = None

    # R topTableF.R:87-100: when rownames(M) (i.e. fit$coefficients
    # rownames) has duplicates, promote them to an ID/ID0 column and
    # reset rn to 1..N. coef_rownames is None for ndarray fits;
    # _handle_duplicated_rownames() catches the simpler case of
    # duplicates inside `genes` itself.
    if coef_rownames is not None and len(coef_rownames) != len(set(coef_rownames)):
        if genelist_df is None:
            genelist_df = pd.DataFrame({"ID": list(coef_rownames)})
        elif "ID" in genelist_df.columns:
            genelist_df = genelist_df.copy()
            genelist_df["ID0"] = list(coef_rownames)
        else:
            genelist_df = genelist_df.copy()
            genelist_df["ID"] = list(coef_rownames)
        genes = list(range(1, n_genes + 1))
    else:
        # Handle duplicated gene names (DataFrame index must be unique)
        genes, genelist_df = _handle_duplicated_rownames(genes, genelist_df)

    # Multiple testing adjustment
    adj_p_val = p_adjust(f_p_val, method=adjust_method)

    # Filter (R topTableF.R: NaN sig dropped only when filtering is active).
    keep = np.ones(n_genes, dtype=bool)
    if p_value < 1 or lfc > 0:
        if p_value < 1:
            keep &= adj_p_val <= p_value
        if lfc > 0:
            keep &= np.any(np.abs(coefficients) > lfc, axis=1)
        keep &= ~np.isnan(f_p_val)

    if not np.any(keep):
        return pd.DataFrame()

    idx = np.where(keep)[0]
    coefficients = coefficients[idx]
    f_stat = f_stat[idx]
    f_p_val = f_p_val[idx]
    adj_p_val = adj_p_val[idx]
    ave_expr = ave_expr[idx]
    genes = [genes[i] for i in idx]
    if genelist_df is not None:
        genelist_df = genelist_df.iloc[idx].reset_index(drop=True)

    # Helper function for sorting
    def _get_sort_order(sort_col, f_stat, f_p_val, ave_expr, default_order):
        sort_by_map = {
            "F": "F", "f": "F",
            "P": "p", "p": "p",
            "AveExpr": "AveExpr", "ave_expr": "AveExpr", "A": "AveExpr",
            "none": "none",
        }
        col = sort_by_map.get(sort_col, "F" if sort_col == "B" else sort_col)
        if col == "F":
            return np.argsort(f_p_val, kind="stable")  # ascending p for F
        elif col == "p":
            return np.argsort(f_p_val, kind="stable")
        elif col == "AveExpr":
            return np.argsort(-ave_expr, kind="stable")
        else:
            return default_order

    # Sort by primary column
    order = _get_sort_order(sort_by, f_stat, f_p_val, ave_expr, np.arange(len(f_stat)))

    if number < len(order):
        order = order[:number]

    # Apply primary sort
    coefficients = coefficients[order]
    f_stat = f_stat[order]
    f_p_val = f_p_val[order]
    adj_p_val = adj_p_val[order]
    ave_expr = ave_expr[order]
    genes = [genes[i] for i in order]
    if genelist_df is not None:
        genelist_df = genelist_df.iloc[order].reset_index(drop=True)

    # Resort if requested
    if resort_by is not None:
        resort_order = _get_sort_order(resort_by, f_stat, f_p_val, ave_expr, np.arange(len(f_stat)))
        coefficients = coefficients[resort_order]
        f_stat = f_stat[resort_order]
        f_p_val = f_p_val[resort_order]
        adj_p_val = adj_p_val[resort_order]
        ave_expr = ave_expr[resort_order]
        genes = [genes[i] for i in resort_order]
        if genelist_df is not None:
            genelist_df = genelist_df.iloc[resort_order].reset_index(drop=True)

    # Build DataFrame with coefficient columns
    # Use contrast_names if available, otherwise generate default names
    stored_names = fit.get("contrast_names") or fit.get("coef_names")
    # R topTableF.R:13: default colnames are `Coef1, Coef2, ...`
    # (1-based, no separator) when fit$coefficients has none.
    if stored_names is not None:
        coef_names = [stored_names[i] for i in coef_idx]
    else:
        coef_names = [f"Coef{i+1}" for i in coef_idx]

    # R topTableF.R:88 builds the table with genelist FIRST, then
    # coefficient columns: `tab <- data.frame(genelist[o,,drop=FALSE],
    # M[o,,drop=FALSE])`. Mirror that column order.
    cols: dict[str, np.ndarray] = {}
    if genelist_df is not None:
        for col in genelist_df.columns:
            cols[col] = genelist_df[col].values
    for j, name in enumerate(coef_names):
        cols[name] = coefficients[:, j]
    if has_amean:
        cols["ave_expr"] = ave_expr
    cols["F"] = f_stat
    cols["p_value"] = f_p_val
    cols["adj_p_value"] = adj_p_val
    df = pd.DataFrame(cols)

    df.index = genes
    df.index.name = "gene"

    return df

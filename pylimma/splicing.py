# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   diffSplice.R                          Copyright (C) 2013-2025 Gordon Smyth,
#                                                                 Charity Law
#   topSplice.R                           Copyright (C) 2013-2025 Gordon Smyth,
#                                                                 Yunshun Chen
#   plotSplice.R                          Copyright (C) 2014-2025 Gordon Smyth,
#                                                                 Yifang Hu,
#                                                                 Yunshun Chen
# Python port: Copyright (C) 2026 John Mulvey
"""
Differential exon-usage testing for pylimma.

Faithful port of R limma's differential-splicing trio:

- ``diff_splice``: per-exon and per-gene tests from an exon-level fit.
- ``top_splice``: collate ``diff_splice`` output into a ranked DataFrame.
- ``plot_splice``: plot exons of one gene from ``diff_splice`` output.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from scipy.stats import f as f_dist

from .classes import MArrayLM, _resolve_fit_input
from .squeeze_var import squeeze_var
from .utils import p_adjust


def _rowsum_sorted(values: np.ndarray, group_codes: np.ndarray,
                   n_groups: int) -> np.ndarray:
    """Port of R's ``rowsum(values, group, reorder=FALSE)`` for the case
    where group codes are already in order-of-first-appearance.

    ``values`` has shape (nrows,) or (nrows, ncols). ``group_codes`` is
    an integer array mapping each row to a group code in [0, n_groups).
    Returns an array of shape (n_groups,) or (n_groups, ncols).
    """
    values = np.asarray(values)
    if values.ndim == 1:
        out = np.zeros(n_groups, dtype=np.float64)
        np.add.at(out, group_codes, values)
        return out
    out = np.zeros((n_groups, values.shape[1]), dtype=np.float64)
    np.add.at(out, group_codes, values)
    return out


def diff_splice(
    fit,
    geneid,
    exonid=None,
    robust: bool = False,
    legacy: bool = False,
    verbose: bool = True,
    *,
    key: str = "pylimma",
) -> MArrayLM:
    """Test for differential exon usage from an exon-level fit.

    Port of R limma's ``diffSplice.MArrayLM`` (Gordon Smyth and Charity
    Law, 2013-2025). The input ``fit`` must be an exon-resolution fit -
    either an :class:`MArrayLM` / dict, or an :class:`AnnData` whose
    ``adata.uns[key]`` holds the fit (as produced by
    ``pylimma.lm_fit(adata_exon, ...)``).

    Gene and exon identifiers are supplied via ``geneid`` and
    ``exonid`` (column names of ``fit['genes']``, or row-aligned
    arrays). For AnnData-stored fits, ``geneid`` / ``exonid`` may also
    be column names of ``adata.var`` since ``lm_fit`` propagates
    ``adata.var`` into ``fit['genes']`` via ``get_eawp``.

    The splice output has a different row count (genes with a single
    exon are dropped) from the input and carries a mix of per-exon and
    per-gene slots; it therefore isn't written back to the AnnData.
    The returned :class:`MArrayLM` is the input to :func:`top_splice` /
    :func:`plot_splice`.

    Parameters
    ----------
    fit : AnnData, MArrayLM, or dict
        Exon-level fit.
    geneid, exonid, robust, legacy, verbose :
        See R's ``diffSplice``.
    key : str, default "pylimma"
        AnnData ``adata.uns`` key holding the fit.
    """
    # Accept AnnData-stored fits (same contract as top_table /
    # decide_tests / genas). For MArrayLM / dict input this is a no-op.
    fit, _adata, _adata_key = _resolve_fit_input(fit, key)
    if not isinstance(fit, dict):
        raise ValueError("fit must be an MArrayLM, dict, or AnnData")

    # For AnnData input, prefer adata.var as the exon-annotation table
    # so string lookups like ``geneid="gene"`` find ``adata.var["gene"]``
    # directly. fit["genes"] is a list of var_names on the AnnData path
    # (per the no-duplication decision for #2 in the audit), which would
    # otherwise fail a column-name lookup.
    n_exons = np.asarray(fit["coefficients"]).shape[0]
    if _adata is not None and _adata.var is not None \
            and len(_adata.var.columns):
        exon_genes = _adata.var.copy()
    else:
        exon_genes = fit.get("genes")
        if exon_genes is None:
            exon_genes = pd.DataFrame({"ExonID": np.arange(1, n_exons + 1)})
        else:
            exon_genes = (
                exon_genes.copy() if isinstance(exon_genes, pd.DataFrame)
                else pd.DataFrame(exon_genes).copy()
            )

    # Extract geneid and exonid
    if isinstance(geneid, str):
        genecolname = geneid
        geneid_arr = exon_genes[genecolname].values
    else:
        geneid_arr = np.asarray(geneid)
        exon_genes["GeneID"] = geneid_arr
        genecolname = "GeneID"

    if exonid is None:
        exoncolname = None
        exonid_arr = None
    else:
        if isinstance(exonid, str):
            exoncolname = exonid
            exonid_arr = exon_genes[exoncolname].values
        else:
            exonid_arr = np.asarray(exonid)
            exon_genes["ExonID"] = exonid_arr
            exoncolname = "ExonID"

    # Treat NA geneids as genes with one exon each
    geneid_arr = np.array([str(v) for v in geneid_arr.tolist()], dtype=object)
    isna_mask = pd.isna(geneid_arr) | (geneid_arr == "nan") | (geneid_arr == "None")
    if isna_mask.any():
        na_idx = np.where(isna_mask)[0]
        geneid_arr = geneid_arr.copy()
        for k, idx in enumerate(na_idx):
            geneid_arr[idx] = f"NA{k + 1}"

    # Sort by (geneid, exonid). np.lexsort sorts by LAST key as primary,
    # so pass (exonid, geneid) to sort primarily by geneid.
    if exonid_arr is None:
        order = np.argsort(geneid_arr, kind="stable")
    else:
        exonid_str = np.array([str(v) for v in exonid_arr.tolist()])
        order = np.lexsort((exonid_str, geneid_arr))

    geneid_arr = geneid_arr[order]
    exon_genes = exon_genes.iloc[order].reset_index(drop=True)

    coefficients = np.asarray(fit["coefficients"], dtype=np.float64)
    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(-1, 1)
    coefficients = coefficients[order, :]

    stdev_unscaled = np.asarray(fit["stdev_unscaled"], dtype=np.float64)
    if stdev_unscaled.ndim == 1:
        stdev_unscaled = stdev_unscaled.reshape(-1, 1)
    stdev_unscaled = stdev_unscaled[order, :]

    df_residual = np.asarray(fit["df_residual"], dtype=np.float64)
    if df_residual.ndim == 0:
        df_residual = np.full(n_exons, float(df_residual))
    df_residual = df_residual[order]

    sigma = np.asarray(fit["sigma"], dtype=np.float64)
    exon_s2 = sigma[order] ** 2

    # Genes with zero df get s2 = 0 (match R line 50)
    if df_residual.min() < 1e-6:
        exon_s2 = exon_s2.copy()
        exon_s2[df_residual < 1e-6] = 0.0

    # Build group codes in order-of-first-appearance (reorder=FALSE).
    unique_genes, first_idx = np.unique(geneid_arr, return_index=True)
    # Sort unique by first_idx to preserve order-of-first-appearance
    order_first = np.argsort(first_idx, kind="stable")
    unique_genes_ord = unique_genes[order_first]
    gene_to_code = {g: i for i, g in enumerate(unique_genes_ord.tolist())}
    geneid_codes = np.array([gene_to_code[g] for g in geneid_arr.tolist()],
                            dtype=np.int64)
    n_genes_total = len(unique_genes_ord)

    # Count exons per gene, sum df.residual, sum df.residual * s2
    gene_nexons = _rowsum_sorted(np.ones(n_exons), geneid_codes,
                                 n_genes_total)
    gene_df_residual = _rowsum_sorted(df_residual, geneid_codes,
                                      n_genes_total)
    gene_df_res_s2 = _rowsum_sorted(df_residual * exon_s2, geneid_codes,
                                    n_genes_total)
    # Pooled gene variance
    with np.errstate(invalid="ignore", divide="ignore"):
        gene_s2 = np.where(gene_df_residual > 0,
                           gene_df_res_s2 / gene_df_residual, 0.0)

    if verbose:
        one_exon = int(np.sum(gene_nexons == 1))
        warnings.warn(
            f"diff_splice: {n_exons} exons, "
            f"{n_genes_total} genes, "
            f"{one_exon} with 1 exon, "
            f"mean {round(float(gene_nexons.mean()))} exons/gene, "
            f"max {int(gene_nexons.max())}",
            UserWarning,
            stacklevel=2,
        )

    # R diffSplice defaults to ``legacy=FALSE``; pylimma.squeeze_var
    # auto-picks legacy=True for all-equal df, so forward the flag
    # explicitly to match R in both branches.
    squeeze = squeeze_var(var=gene_s2, df=gene_df_residual,
                          robust=robust, legacy=legacy)

    # Keep only genes with > 1 exon
    gene_keep = gene_nexons > 1
    ngenes = int(gene_keep.sum())
    if ngenes == 0:
        raise ValueError("No genes with more than one exon")

    # Repeat gene_keep across each gene's exons. Because exons are sorted
    # by gene (order-of-first-appearance), we can tile gene_keep by
    # gene_nexons.
    exon_keep = np.repeat(gene_keep, gene_nexons.astype(np.int64))

    geneid_arr = geneid_arr[exon_keep]
    exon_genes = exon_genes.iloc[exon_keep].reset_index(drop=True)
    coefficients = coefficients[exon_keep, :]
    stdev_unscaled = stdev_unscaled[exon_keep, :]
    df_residual = df_residual[exon_keep]

    gene_nexons_kept = gene_nexons[gene_keep]
    gene_df_test = gene_nexons_kept - 1
    gene_df_residual_kept = gene_df_residual[gene_keep]

    df_prior = squeeze["df_prior"]
    df_prior_arr = np.atleast_1d(np.asarray(df_prior, dtype=np.float64))
    if df_prior_arr.size > 1:
        df_prior_arr = df_prior_arr[gene_keep]
    gene_df_total = gene_df_residual_kept + df_prior_arr
    # Cap df_total at sum of residual dfs
    gene_df_total = np.minimum(gene_df_total, float(gene_df_residual_kept.sum()))

    var_post = np.asarray(squeeze["var_post"])
    gene_s2_post = var_post[gene_keep]

    # Rebuild group codes on kept exons
    unique_genes_kept = unique_genes_ord[gene_keep]
    kept_gene_to_code = {g: i for i, g in enumerate(unique_genes_kept.tolist())}
    geneid_codes_kept = np.array([kept_gene_to_code[g] for g in geneid_arr.tolist()],
                                 dtype=np.int64)

    # Per-gene weighted mean beta
    u2 = 1.0 / (stdev_unscaled ** 2)
    u2_rowsum = _rowsum_sorted(u2, geneid_codes_kept, ngenes)
    gene_betabar = _rowsum_sorted(coefficients * u2,
                                  geneid_codes_kept, ngenes) / u2_rowsum

    # t-statistics and F-statistics
    g = geneid_codes_kept
    coefficients_centred = coefficients - gene_betabar[g, :]
    exon_t = coefficients_centred / stdev_unscaled / np.sqrt(gene_s2_post[g])[:, None]
    exon_t_sq = exon_t ** 2
    gene_F = _rowsum_sorted(exon_t_sq, geneid_codes_kept,
                            ngenes) / gene_df_test[:, None]

    exon_1m_leverage = 1.0 - (u2 / u2_rowsum[g, :])
    coefficients_final = coefficients_centred / exon_1m_leverage
    exon_t_final = exon_t / np.sqrt(exon_1m_leverage)

    with np.errstate(invalid="ignore"):
        gene_df_total_tiled = gene_df_total[g][:, None]
        # Broadcast exon_t_final (n_exons, n_coefs) vs df (n_exons, 1).
        exon_p_value = 2.0 * t_dist.sf(np.abs(exon_t_final),
                                       df=gene_df_total_tiled)

    gene_F_p_value = f_dist.sf(gene_F,
                               dfn=gene_df_test[:, None],
                               dfd=gene_df_total[:, None])

    # Assemble exon-level output
    out = MArrayLM()
    out["genes"] = exon_genes
    out["genecolname"] = genecolname
    out["exoncolname"] = exoncolname
    out["coefficients"] = coefficients_final
    out["t"] = exon_t_final
    out["p_value"] = exon_p_value

    # Gene-level output
    out["gene_df_prior"] = df_prior_arr
    out["gene_df_residual"] = gene_df_residual_kept
    out["gene_df_total"] = gene_df_total
    out["gene_s2"] = gene_s2[gene_keep]
    out["gene_s2_post"] = gene_s2_post
    out["gene_F"] = gene_F
    out["gene_F_p_value"] = gene_F_p_value

    # Identify gene-level annotation columns (those constant within each
    # gene). First and last exon indices (0-based).
    gene_lastexon = np.cumsum(gene_nexons_kept.astype(np.int64)) - 1
    gene_firstexon = gene_lastexon - gene_nexons_kept.astype(np.int64) + 1

    # Per-column: is this column constant within each gene?
    n_kept = len(exon_genes)
    gene_genes = exon_genes.iloc[gene_lastexon].reset_index(drop=True)
    is_gene_level = []
    for col in exon_genes.columns:
        col_vals = exon_genes[col].values
        constant = True
        # Check that within each gene, all values equal the last-exon value
        for gi in range(ngenes):
            first = int(gene_firstexon[gi])
            last = int(gene_lastexon[gi])
            if last > first:
                ref = col_vals[last]
                for k in range(first, last):
                    a = col_vals[k]
                    b = ref
                    # NaN-aware equality
                    if pd.isna(a) and pd.isna(b):
                        continue
                    if a != b:
                        constant = False
                        break
            if not constant:
                break
        is_gene_level.append(constant)
    gene_level_cols = [c for c, keep in zip(exon_genes.columns, is_gene_level)
                       if keep]
    gene_genes = gene_genes[gene_level_cols].copy()
    gene_genes.index = gene_genes[genecolname].astype(str).values
    gene_genes["NExons"] = gene_nexons_kept.astype(np.int64)
    out["gene_genes"] = gene_genes
    out["gene_firstexon"] = gene_firstexon
    out["gene_lastexon"] = gene_lastexon

    # Simes adjustment - for each coefficient column independently
    n_coefs = coefficients_final.shape[1]
    gene_simes = gene_F_p_value.copy()

    # Build the penalty vector (R lines 138-141):
    # penalty[i] = (within-gene rank) starting at 1; then inverse-ranked
    # by gene size: penalty = n_in_gene / within_gene_rank
    within_gene_rank = np.zeros(len(g), dtype=np.int64)
    for gi in range(ngenes):
        first = int(gene_firstexon[gi])
        last = int(gene_lastexon[gi])
        within_gene_rank[first:last + 1] = np.arange(1, last - first + 2)
    gene_nexons_per_exon = gene_nexons_kept[g].astype(np.float64)
    penalty = gene_nexons_per_exon / within_gene_rank

    for j in range(n_coefs):
        # Sort exons within gene by p-value ascending; multiply by penalty
        # in rank order; take min-adjusted p per gene.
        pj = exon_p_value[:, j]
        # lexsort: primary = g (gene), secondary = pj (within-gene p)
        # np.lexsort sorts by LAST key as primary.
        order_p = np.lexsort((pj, g))
        p_ordered = pj[order_p]
        adj = p_ordered * penalty
        # Now take the minimum adjusted p per gene. Within each gene in
        # the sorted order, the minimum is the first since penalty is
        # monotonically non-increasing? Not quite - penalty = n / rank,
        # rank 1..n so penalty is DECREASING. p_ordered is INCREASING.
        # The product is not monotone. R's logic:
        #   o <- order(g, p.adj)
        #   out$gene.simes.p.value[,j] <- p.adj[o][gene.firstexon]
        # i.e., re-sort by (gene, adj) and take the first (smallest) per
        # gene. Mirror that.
        # We need the positions in the ORIGINAL (kept) exon ordering.
        # After order_p, group positions = g[order_p]. Re-sort by
        # (g_sorted, adj) and take per-gene minimum.
        g_sorted = g[order_p]
        order2 = np.lexsort((adj, g_sorted))
        adj_final = adj[order2]
        # Per-gene first position in this double-sorted order
        g_double = g_sorted[order2]
        first_positions = np.zeros(ngenes, dtype=np.int64)
        # Walk through g_double to find first occurrence of each gene.
        # g_double is not sorted (g_sorted was sorted first, then ties
        # broken by adj). But gene groups are still contiguous because
        # we sorted on (g_sorted, adj).
        cur_gene = -1
        for k in range(len(g_double)):
            if g_double[k] != cur_gene:
                first_positions[g_double[k]] = k
                cur_gene = int(g_double[k])
        gene_simes[:, j] = adj_final[first_positions]

    out["gene_simes_p_value"] = gene_simes

    # Bonferroni adjustment
    gene_bonferroni = gene_F_p_value.copy()
    for j in range(n_coefs):
        pj = exon_p_value[:, j]
        order_p = np.lexsort((pj, g))
        p_ordered = pj[order_p]
        # For each gene, take the first (smallest) p-value and multiply
        # by its n_exons. Bonferroni cap at 1.
        adj_bon = np.minimum(
            p_ordered[gene_firstexon] * gene_nexons_kept.astype(np.float64),
            1.0,
        )
        gene_bonferroni[:, j] = adj_bon
    out["gene_bonferroni_p_value"] = gene_bonferroni

    return out


def top_splice(
    fit,
    coef: int = -1,
    test: str = "F",
    number: int | float = 10,
    fdr: float = 1.0,
    sort_by: str = "p",
) -> pd.DataFrame:
    """Top-ranked splicing results from a ``diff_splice`` fit.

    Port of R limma's ``topSplice``. ``number=np.inf`` returns all rows.
    """
    if fit.get("gene_genes") is None or "NExons" not in fit["gene_genes"].columns:
        raise ValueError("fit should be a fit object produced by diff_splice")

    # Normalise test
    test_low = test.lower()
    if test_low == "f":
        test_upper = "F"
    elif test_low == "simes":
        test_upper = "simes"
    elif test_low == "t":
        test_upper = "t"
    else:
        raise ValueError(
            f"test must be 'simes', 'F', or 't', got {test!r}"
        )

    if sort_by not in ("p", "none", "logFC", "NExons"):
        raise ValueError(
            f"sort_by must be 'p', 'none', 'logFC', or 'NExons', "
            f"got {sort_by!r}"
        )
    if sort_by == "logFC" and test_upper != "t":
        raise ValueError("Sorting by logFC only available with t test")
    if sort_by == "NExons" and test_upper == "t":
        raise ValueError("Sorting by NExons only available with gene-level tests")

    # Resolve coef. R default: ncol(fit), i.e. the last column (1-based
    # ncol = last column). In pylimma 0-based convention, coef=-1 maps
    # to last column.
    n_coefs = np.asarray(fit["coefficients"]).shape[1]
    if coef == -1:
        c = n_coefs - 1
    elif isinstance(coef, str):
        names_list = fit.get("contrast_names") or fit.get("coef_names")
        if names_list is None or coef not in list(names_list):
            raise ValueError(f"coef {coef!r} not found")
        c = list(names_list).index(coef)
    else:
        c = int(coef)

    if test_upper == "t":
        out = fit["genes"].copy()
        out["logFC"] = np.asarray(fit["coefficients"])[:, c]
        out["t"] = np.asarray(fit["t"])[:, c]
        out["P.Value"] = np.asarray(fit["p_value"])[:, c]
    elif test_upper == "F":
        out = fit["gene_genes"].copy()
        out["F"] = np.asarray(fit["gene_F"])[:, c]
        out["P.Value"] = np.asarray(fit["gene_F_p_value"])[:, c]
    else:  # simes
        out = fit["gene_genes"].copy()
        out["P.Value"] = np.asarray(fit["gene_simes_p_value"])[:, c]

    out["FDR"] = p_adjust(out["P.Value"].values, method="BH")

    if fdr < 1.0:
        out = out[out["FDR"] <= fdr]

    number = min(int(number) if number != np.inf else len(out), len(out))
    if number <= 1:
        return out

    if sort_by == "p":
        o = np.argsort(out["P.Value"].values, kind="stable")
    elif sort_by == "logFC":
        o = np.argsort(-np.abs(out["logFC"].values), kind="stable")
    elif sort_by == "NExons":
        # R: order(out$NExons, -out$P.Value, decreasing=TRUE)
        # decreasing=TRUE applies to BOTH keys via negation.
        # Primary: -NExons ascending  = NExons descending
        # Secondary: -(-P.Value) ascending = P.Value ascending
        o = np.lexsort((out["P.Value"].values,
                        -out["NExons"].values.astype(np.float64)))
    else:  # none
        o = np.arange(len(out))

    o = o[:number]
    return out.iloc[o].reset_index(drop=True)


def plot_splice(
    fit,
    coef: int = -1,
    geneid: str | None = None,
    genecolname: str | None = None,
    rank: int = 1,
    fdr: float = 0.05,
    xlab: str = "Exon",
    ax=None,
    **kwargs,
):
    """Plot exons or isoforms of a chosen gene. Port of R limma's
    ``plotSplice``.
    """
    from .plotting import _require_matplotlib
    plt = _require_matplotlib()

    if genecolname is None:
        genecolname = fit.get("genecolname")
    else:
        genecolname = str(genecolname)

    n_coefs = np.asarray(fit["coefficients"]).shape[1]
    c = n_coefs - 1 if coef == -1 else int(coef)

    gene_F_p = np.asarray(fit["gene_F_p_value"])
    gene_genes = fit["gene_genes"]

    if geneid is None:
        if rank == 1:
            i = int(np.argmin(gene_F_p[:, c]))
        else:
            i = int(np.argsort(gene_F_p[:, c], kind="stable")[rank - 1])
        geneid = str(gene_genes.iloc[i][genecolname])
    else:
        geneid = str(geneid)
        matches = np.where(gene_genes[genecolname].astype(str).values == geneid)[0]
        if len(matches) == 0:
            raise ValueError(f"geneid {geneid} not found")
        i = int(matches[0])

    first = int(fit["gene_firstexon"][i])
    last = int(fit["gene_lastexon"][i])
    j = np.arange(first, last + 1)

    exoncolname = fit.get("exoncolname")
    ylab = f"logFC (this {xlab.lower()} vs rest)"

    coefs = np.asarray(fit["coefficients"])[j, c]

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(np.arange(1, len(j) + 1), coefs, "o-", color="black")
    ax.set_ylabel(ylab)
    ax.set_title(geneid)

    if exoncolname is None:
        ax.set_xlabel(xlab)
    else:
        exon_ids = fit["genes"].iloc[j][exoncolname].astype(str).values
        ax.set_xticks(np.arange(1, len(j) + 1))
        ax.set_xticklabels(exon_ids, rotation=90, fontsize=6)
        ax.set_xlabel(f"{xlab} {exoncolname}")

    # Mark significant exons
    top = top_splice(fit, coef=c, number=np.inf, test="t", sort_by="none")
    gene_label = gene_genes.iloc[i][genecolname]
    mask = top[genecolname].astype(str).values == str(gene_label)
    fdr_vals = top["FDR"].values[mask]
    sig = fdr_vals < fdr
    if np.any(sig):
        fdr_sig = fdr_vals[sig]
        if len(np.unique(fdr_sig)) == 1:
            cex = 1.5
        else:
            abs_fdr = np.abs(np.log10(fdr_sig))
            lo, hi = abs_fdr.min(), abs_fdr.max()
            cex = (abs_fdr - lo) / (hi - lo) * 1.0 + 1.0
        x_positions = np.arange(1, len(j) + 1)[sig]
        y_positions = coefs[sig]
        sizes = np.atleast_1d(cex) * 25
        ax.scatter(x_positions, y_positions, c="red",
                   s=sizes if len(sizes) == len(x_positions) else sizes[0],
                   marker="o", zorder=5)

    ax.axhline(0.0, linestyle="--", color="gray")
    return ax

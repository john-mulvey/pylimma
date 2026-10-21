# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   R/goana.R       Copyright (C) 2014-2022 Gordon Smyth, Yifang Hu
#   R/kegga.R       Copyright (C) 2015-2022 Gordon Smyth, Yifang Hu
#   R/goanaTrend.R  Copyright (C) 2022      Gordon Smyth
# Python port: Copyright (C) 2026 John Mulvey
"""
Gene-ontology and KEGG pathway over-representation analysis.

Public API
----------
goana, top_go            Gene-ontology over-representation, port of R
                         limma's ``goana`` and ``topGO``.
kegga, top_kegg          KEGG pathway over-representation, port of R
                         limma's ``kegga`` and ``topKEGG``.
goana_trend              Per-gene DE-probability estimate from a covariate
                         (used internally for length / abundance bias
                         correction), port of R limma's ``goanaTrend``.

Scope
-----
The port covers the database-free path. Two differences from R limma:

1. **Bioconductor database lookups are not ported.** R limma's
   ``goana.default`` reaches into ``GO.db`` / ``org.<species>.eg.db``;
   ``kegga.default`` reaches into the live KEGG REST API. pylimma does
   not wrap Bioconductor annotation databases. Both ports require the
   caller to supply a ``gene_pathway`` data frame mapping gene IDs to
   pathway IDs (the universal interface, fully database-free). The
   ``getGeneKEGGLinks`` and ``getKEGGPathwayNames`` helpers, which exist
   in R only to fetch from the KEGG REST API, are not ported.

2. **The bias-corrected test is not yet ported.** R's ``trend=TRUE``,
   ``covariate`` and ``null.prob`` paths perform Wallenius' noncentral
   hypergeometric test via ``BiasedUrn::pWNCHypergeo`` /
   ``dWNCHypergeo`` for length / abundance bias correction. pylimma
   raises ``NotImplementedError`` whenever ``trend`` is truthy or a
   ``covariate`` or ``prior_prob`` would be used, rather than silently
   falling back to the plain hypergeometric test, which would give
   plausible-looking but wrong results. As in R, ``covariate`` and
   ``prior_prob`` are ignored (with a warning) when ``universe`` is not
   given, and a ``covariate`` is ignored when no gene is DE.

The plain-hypergeometric branch (``trend=False``) ports verbatim and is
numerically validated against R limma fixtures.

gene_pathway format
-------------------
A pandas ``DataFrame`` (or anything ``DataFrame(...)`` accepts) with at
least two columns:

* column 0 - gene id
* column 1 - pathway id (GO term, KEGG pathway, ...)
* column 2 *(optional, goana only)* - GO ontology, one of "BP" / "CC"
  / "MF". Surfaces in the output's ``ontology`` column and lets
  ``top_go`` filter by ontology.
* column 3 *(optional, goana only)* - human-readable term name.
  Surfaces in the output's ``term`` column. Without this column the
  ``term`` column repeats the pathway id verbatim, since pylimma has
  no GO.db lookup to fall back to.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .classes import (
    MArrayLM,
    _fit_var,
    _is_anndata,
    _resolve_fit_input,
)
from .geneset import _is_list_of_sets
from .toptable import _top_row_count
from .utils import _match_arg, p_adjust, tricube_moving_average

# ---------------------------------------------------------------------------
# Public dispatchers
# ---------------------------------------------------------------------------


def goana(
    de,
    gene_pathway,
    universe=None,
    *,
    species=None,
    prior_prob=None,
    covariate=None,
    plot=False,
    fdr=0.05,
    trend=False,
    **kwargs,
) -> pd.DataFrame:
    """
    Gene-ontology over-representation analysis.

    Port of R limma's ``goana`` (``goana.R``). See module docstring for
    the gene_pathway format and the scope notes.

    Parameters
    ----------
    de : MArrayLM, AnnData, dict-of-vectors, or 1-D vector of gene IDs
        Differentially-expressed genes. A vector (or a single gene ID) is
        one set named ``"DE"``, a list of vectors is several sets named
        ``"DE1"``, ``"DE2"``, ... and a DataFrame raises, as in R. When a fit
        object is supplied,
        the up- and down-regulated gene sets are derived internally
        (BH adjustment at level ``fdr``).
    gene_pathway : DataFrame
        Mapping from gene IDs to GO terms. See module docstring.
    universe : sequence of str, optional
        Background gene universe. Defaults to all gene IDs in
        ``gene_pathway``. As in R, genes with no annotation (not in
        ``gene_pathway``) are dropped from the universe, so they do not
        count towards the number of genes. A fit's universe is its genes;
        passing ``universe`` (or ``covariate``) with a fit raises, as R.
    species : str, optional
        Accepted for R-API compatibility; ignored without GO.db.
    prior_prob : sequence of float, optional
        Per-gene null DE probability (R name: ``null.prob``), aligned to
        the universe. Not yet supported (it selects Wallenius' test):
        raises ``NotImplementedError``. Ignored, with a warning, when
        ``universe`` is not given.
    covariate : sequence of float, optional
        Covariate values aligned to the universe, from which
        ``goana_trend`` computes ``prior_prob``. Not yet supported:
        raises ``NotImplementedError``. Ignored, with a warning, when
        ``universe`` is not given or no gene is DE.
    plot : bool, default False
        Forwarded to ``goana_trend`` when applicable.
    fdr : float, default 0.05
        FDR threshold for selecting up/down DE genes from a fit object
        (R name: ``FDR``). Only used in the MArrayLM branch.
    trend : bool, numeric, or str, default False
        Use length / abundance bias correction. Not yet supported:
        raises ``NotImplementedError`` for any truthy / numeric /
        character value.
    **kwargs
        ``coef``, ``geneid`` and ``key`` for the MArrayLM branch. ``key``
        (default ``"pylimma"``) names the ``adata.uns`` entry holding the
        fit when ``de`` is an AnnData. For AnnData input a string
        ``geneid`` is looked up in ``adata.var``. The stored fit is only
        used while ``var_names`` still match the genes that were fitted;
        otherwise a ValueError is raised and ``lm_fit`` must be re-run.
    """
    if _is_dispatch_marraylm(de):
        return _goana_marraylm(
            de,
            gene_pathway=gene_pathway,
            universe=universe,
            species=species,
            prior_prob=prior_prob,
            covariate=covariate,
            plot=plot,
            fdr=fdr,
            trend=trend,
            **kwargs,
        )
    return _goana_default(
        de,
        gene_pathway=gene_pathway,
        universe=universe,
        species=species,
        prior_prob=prior_prob,
        covariate=covariate,
        plot=plot,
        trend=trend,
    )


def kegga(
    de,
    gene_pathway,
    pathway_names=None,
    universe=None,
    *,
    restrict_universe=False,
    species=None,
    prior_prob=None,
    covariate=None,
    plot=False,
    fdr=0.05,
    trend=False,
    **kwargs,
) -> pd.DataFrame:
    """
    KEGG pathway over-representation analysis.

    Port of R limma's ``kegga`` (``kegga.R``). See module docstring for
    the gene_pathway format and the scope notes.

    All parameters except ``pathway_names`` have identical semantics to
    :func:`goana`; see that function's docstring for full descriptions.

    Parameters
    ----------
    pathway_names : DataFrame, optional
        Two-column mapping from pathway id (column 0) to human-readable
        description (column 1). If absent, the output's ``pathway``
        column repeats the pathway id verbatim.
    restrict_universe : bool, default False
        Drop universe genes that are not in any pathway of
        ``gene_pathway`` (R: ``restrict.universe``). ``goana`` always
        does this.
    """
    if _is_dispatch_marraylm(de):
        return _kegga_marraylm(
            de,
            gene_pathway=gene_pathway,
            pathway_names=pathway_names,
            universe=universe,
            restrict_universe=restrict_universe,
            species=species,
            prior_prob=prior_prob,
            covariate=covariate,
            plot=plot,
            fdr=fdr,
            trend=trend,
            **kwargs,
        )
    return _kegga_default(
        de,
        gene_pathway=gene_pathway,
        pathway_names=pathway_names,
        universe=universe,
        restrict_universe=restrict_universe,
        species=species,
        prior_prob=prior_prob,
        covariate=covariate,
        plot=plot,
        trend=trend,
    )


def _is_dispatch_marraylm(de) -> bool:
    if _is_anndata(de):
        return True
    if isinstance(de, MArrayLM):
        return True
    if isinstance(de, dict) and "coefficients" in de and "p_value" in de:
        return True
    return False


# ---------------------------------------------------------------------------
# MArrayLM dispatch helpers (port of goana.MArrayLM, kegga.MArrayLM)
# ---------------------------------------------------------------------------


def _select_de_from_fit(
    fit, geneid, coef, fdr, adata=None
) -> tuple[list[str], list[str], list[str]]:
    """Return (universe, up_ids, down_ids) from an MArrayLM-shaped fit.

    Mirrors lines 13-77 of goana.R / kegga.R: validate the fit, resolve
    geneid (vector vs. column-name reference), select FDR-significant
    up/down genes via BH at level ``fdr``.
    """
    if fit.get("coefficients") is None:
        raise ValueError("de does not appear to be a valid MArrayLM fit object.")
    if fit.get("p_value") is None:
        raise ValueError("p.value not found in fit object, perhaps need to run e_bayes first.")

    coefs = np.asarray(fit["coefficients"])
    if coefs.ndim == 1:
        coefs = coefs.reshape(-1, 1)
    pvals = np.asarray(fit["p_value"])
    if pvals.ndim == 1:
        pvals = pvals.reshape(-1, 1)
    ngenes, ncoef = coefs.shape

    if coef is None:
        coef = ncoef - 1
    # R: length(coef) != 1, where a column name has length 1.
    if not isinstance(coef, str) and hasattr(coef, "__len__"):
        if len(coef) != 1:
            raise ValueError("Only one coef can be specified.")
        coef = coef[0]
    if isinstance(coef, str):
        coef_names = list(fit.get("contrast_names") or fit.get("coef_names") or [])
        if not coef_names:
            # Try column index fallback via design.
            design = fit.get("design")
            if design is not None and hasattr(design, "columns"):
                coef_names = list(design.columns)
        if coef not in coef_names:
            raise ValueError(f"coef {coef!r} not found in fit")
        coef_idx = coef_names.index(coef)
    else:
        coef_idx = int(coef)
        if coef_idx < 0 or coef_idx >= ncoef:
            raise ValueError(f"coef index {coef_idx} out of range")

    # Resolve geneid: either a length-ngenes vector or a single column name
    # in fit['genes']. R's default is geneid = rownames(de); pylimma keeps
    # row names in fit['genes'] as a list (ndarray after an h5ad round
    # trip) or as the index of a genes DataFrame.
    genes = fit.get("genes")
    if geneid is None:
        if isinstance(genes, pd.DataFrame):
            rownames = genes.index
        elif genes is not None and np.ndim(genes) == 1:
            rownames = genes
        else:
            rownames = []
        universe = [str(v) for v in rownames]
        if len(universe) != ngenes:
            # R: as.character(rownames(de)) has length 0 when absent.
            raise ValueError("geneid of incorrect length")
    elif isinstance(geneid, str) or (
        hasattr(geneid, "__len__")
        and len(geneid) == 1
        and not isinstance(geneid, (np.ndarray, pd.Series))
    ):
        col = geneid if isinstance(geneid, str) else geneid[0]
        if adata is not None:
            # AnnData: fit['genes'] only carries var_names, so column-name
            # lookups go to adata.var, reduced to the fit's rows
            # (_resolve_fit_input has checked that it still describes them).
            genes = _fit_var(fit, adata)
        if not isinstance(genes, pd.DataFrame) or col not in genes.columns:
            raise ValueError(f"Column {col} not found in de$genes")
        universe = [str(v) for v in genes[col].values]
    else:
        gid_arr = np.asarray(geneid).ravel()
        if gid_arr.size != ngenes:
            raise ValueError("geneid of incorrect length")
        universe = [str(v) for v in gid_arr]

    if not isinstance(fdr, (int, float, np.integer, np.floating)):
        raise ValueError("FDR must be numeric and of length 1.")
    fdr = float(fdr)
    if fdr < 0 or fdr > 1:
        raise ValueError("FDR should be between 0 and 1.")

    pvec = pvals[:, coef_idx]
    cvec = coefs[:, coef_idx]

    # Drop rows where geneid is NA (R's anyNA branch).
    isna = np.array(
        [
            v is None
            or (isinstance(v, float) and np.isnan(v))
            or v == ""
            or v == "nan"
            or v == "NA"
            for v in universe
        ]
    )
    if isna.all():
        raise ValueError("Gene IDs are all NA")
    if isna.any():
        keep = ~isna
        universe = [u for u, k in zip(universe, keep) if k]
        pvec = pvec[keep]
        cvec = cvec[keep]

    fdr_coef = p_adjust(pvec, method="BH")
    up_mask = (fdr_coef < fdr) & (cvec > 0)
    dn_mask = (fdr_coef < fdr) & (cvec < 0)
    up = [u for u, m in zip(universe, up_mask) if m]
    dn = [u for u, m in zip(universe, dn_mask) if m]
    return universe, up, dn


def _goana_marraylm(
    de, *, gene_pathway, universe, species, prior_prob, covariate, plot, fdr, trend, **kwargs
):
    """Port of goana.MArrayLM (goana.R:3-86)."""
    fit, _adata, _adata_key = _resolve_fit_input(de, kwargs.pop("key", "pylimma"))

    # R stops for a universe; a covariate collides with the one passed on.
    if universe is not None:
        raise ValueError("goana.MArrayLM defines its own universe")
    if covariate is not None:
        raise ValueError("goana.MArrayLM defines its own covariate")

    geneid = kwargs.pop("geneid", None)
    coef = kwargs.pop("coef", None)
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")

    if not isinstance(trend, (bool, np.bool_)):
        # R accepts numeric or character trend; both go through the
        # BiasedUrn path which is deferred.
        raise NotImplementedError(
            "trend correction requires the BiasedUrn port; see pylimma roadmap."
        )

    fit_universe, up, dn = _select_de_from_fit(
        fit, geneid, coef, fdr, adata=_adata
    )

    if not up and not dn:
        warnings.warn("No DE genes")
        return pd.DataFrame()

    return _goana_default(
        {"Up": up, "Down": dn},
        gene_pathway=gene_pathway,
        universe=fit_universe,
        species=species,
        prior_prob=prior_prob,
        covariate=covariate,
        plot=plot,
        trend=trend,
    )


def _kegga_marraylm(
    de,
    *,
    gene_pathway,
    pathway_names,
    universe,
    restrict_universe,
    species,
    prior_prob,
    covariate,
    plot,
    fdr,
    trend,
    **kwargs,
):
    """Port of kegga.MArrayLM (kegga.R:3-86)."""
    fit, _adata, _adata_key = _resolve_fit_input(de, kwargs.pop("key", "pylimma"))

    # R stops for a universe; a covariate collides with the one passed on.
    if universe is not None:
        raise ValueError("kegga.MArrayLM defines its own universe")
    if covariate is not None:
        raise ValueError("kegga.MArrayLM defines its own covariate")

    geneid = kwargs.pop("geneid", None)
    coef = kwargs.pop("coef", None)
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")

    if not isinstance(trend, (bool, np.bool_)):
        raise NotImplementedError(
            "trend correction requires the BiasedUrn port; see pylimma roadmap."
        )

    fit_universe, up, dn = _select_de_from_fit(
        fit, geneid, coef, fdr, adata=_adata
    )

    if not up and not dn:
        warnings.warn("No DE genes")
        return pd.DataFrame()

    return _kegga_default(
        {"Up": up, "Down": dn},
        gene_pathway=gene_pathway,
        pathway_names=pathway_names,
        universe=fit_universe,
        restrict_universe=restrict_universe,
        species=species,
        prior_prob=prior_prob,
        covariate=covariate,
        plot=plot,
        trend=trend,
    )


# ---------------------------------------------------------------------------
# default-method helpers
# ---------------------------------------------------------------------------


def _ensure_de_lists(de) -> dict[str, np.ndarray]:
    """Port of the R 'ensure de is a list, dedupe, name uniquely' block.

    Mirrors goana.R:160-179 and kegga.R:93-112.
    """
    if isinstance(de, pd.DataFrame):
        raise ValueError("de should be a list of character vectors. It should not be a data.frame.")
    if isinstance(de, dict):
        names = list(de.keys())
        sets = list(de.values())
    elif _is_list_of_sets(de):
        # R's unnamed list of sets: named DE1, DE2, ... below.
        names = [None] * len(de)
        sets = list(de)
    else:
        # R: de <- list(DE = de), so a single gene ID is a set of one.
        names = ["DE"]
        sets = [de]

    gene_sets: list[np.ndarray] = []
    cleaned_names: list[str] = []
    for nm, vec in zip(names, sets):
        if isinstance(vec, pd.DataFrame):
            raise ValueError(
                "de should be a list of character vectors. It should not be a data.frame."
            )
        if vec is None:
            raise ValueError("components of de should be vectors")
        array_like = (str, bytes, np.ndarray, pd.Series, pd.Index)
        if not isinstance(vec, array_like) and hasattr(vec, "__iter__"):
            # Python sets, dict views and generators, as well as lists.
            vec = list(vec)
        arr = np.asarray(vec, dtype=object).ravel()
        # R: unique(as.character(...)), in order of first appearance.
        gene_sets.append(np.array(list(dict.fromkeys(str(v) for v in arr)), dtype=object))
        cleaned_names.append("" if nm is None else str(nm).strip())

    # Replace empty / NA-like names with DE<i> and disambiguate duplicates.
    final_names: list[str] = []
    for i, nm in enumerate(cleaned_names, start=1):
        if nm == "" or nm.lower() == "nan":
            final_names.append(f"DE{i}")
        else:
            final_names.append(nm)

    seen_count: dict[str, int] = {}
    for nm in final_names:
        seen_count[nm] = seen_count.get(nm, 0) + 1
    width: dict[str, int] = {
        nm: 1 + int(np.floor(np.log10(c))) if c > 1 else 0 for nm, c in seen_count.items()
    }
    occ: dict[str, int] = {nm: 0 for nm in seen_count}
    disambiguated: list[str] = []
    for nm in final_names:
        if seen_count[nm] > 1:
            occ[nm] += 1
            disambiguated.append(f"{nm}{occ[nm]:0{width[nm]}d}")
        else:
            disambiguated.append(nm)

    return dict(zip(disambiguated, gene_sets))


def _normalise_universe(universe, prior_prob, covariate, annotated_genes, restrict):
    """Apply R's NA-removal, dedup and (optional) restrict-to-annotated logic.

    Mirrors goana.R:122-156 and kegga.R:155-202 for the explicit-universe
    path (the implicit / database-built path doesn't apply here). With
    ``restrict``, genes not in ``annotated_genes`` are dropped, as goana
    always does and kegga does with ``restrict.universe=TRUE``.
    """
    universe = np.asarray(universe).astype(str).ravel()
    lu = universe.size
    if prior_prob is not None:
        prior_prob = np.asarray(prior_prob, dtype=np.float64).ravel().copy()
        if prior_prob.size != lu:
            raise ValueError("universe and null.prob must have same length")
    if covariate is not None:
        covariate = np.asarray(covariate, dtype=np.float64).ravel().copy()
        if covariate.size != lu:
            raise ValueError("universe and covariate must have same length")

    # NA filtering
    isna = np.isin(universe, ["", "nan", "NA", "None"])
    if covariate is not None:
        isna |= np.isnan(covariate)
    if prior_prob is not None:
        isna |= np.isnan(prior_prob)
    if isna.any():
        if isna.all():
            raise ValueError("Gene IDs are all NA")
        keep = ~isna
        universe = universe[keep]
        if covariate is not None:
            covariate = covariate[keep]
        if prior_prob is not None:
            prior_prob = prior_prob[keep]

    # Dedup universe (keep first occurrence). R takes !duplicated
    # (so first occurrence is kept), and slices covariate / prior_prob by
    # the same mask.
    _, first_idx = np.unique(universe, return_index=True)
    if first_idx.size != universe.size:
        keep = np.zeros(universe.size, dtype=bool)
        keep[first_idx] = True
        universe = universe[keep]
        if covariate is not None:
            covariate = covariate[keep]
        if prior_prob is not None:
            prior_prob = prior_prob[keep]

    if restrict:
        keep = np.isin(universe, np.asarray(annotated_genes, dtype=str))
        universe = universe[keep]
        if covariate is not None:
            covariate = covariate[keep]
        if prior_prob is not None:
            prior_prob = prior_prob[keep]

    return universe, prior_prob, covariate


def _normalise_gene_pathway(gene_pathway, *, optional_extra_cols=False):
    """Validate and dedupe a gene_pathway DataFrame.

    Mirrors kegga.R:122-138 / goana.R:153-156:
    - require 2 cols
    - drop rows where either of the first two cols is NA
    - drop duplicate (col1, col2) rows

    Returns a fresh DataFrame keyed off the *first two* columns
    (renamed gene_id, pathway_id) with optional extra columns
    preserved when ``optional_extra_cols`` is True (used by goana to
    carry ontology / term lookups through).
    """
    if gene_pathway is None:
        raise ValueError(
            "gene_pathway is required; pylimma does not look up GO.db or the "
            "KEGG REST API, so supply a gene-to-pathway mapping."
        )
    df = pd.DataFrame(gene_pathway).copy()
    if df.shape[1] < 2:
        raise ValueError("gene.pathway must have at least 2 columns")
    cols = list(df.columns)
    df = df.rename(columns={cols[0]: "gene_id", cols[1]: "pathway_id"})
    df["gene_id"] = df["gene_id"].astype(str)
    df["pathway_id"] = df["pathway_id"].astype(str)

    # Drop rows where either of the first two cols is NA
    isna = (
        df[["gene_id", "pathway_id"]].isna().any(axis=1)
        | (df["gene_id"].isin(["", "nan", "NA"]))
        | (df["pathway_id"].isin(["", "nan", "NA"]))
    )
    df = df.loc[~isna].copy()

    # Drop duplicate (gene_id, pathway_id) rows
    df = df.drop_duplicates(subset=["gene_id", "pathway_id"], keep="first")

    if not optional_extra_cols and df.shape[1] > 2:
        df = df.iloc[:, :2]
    return df


def _check_null_prob(de_lists, prior_prob, covariate) -> None:
    """Raise for the Wallenius test that a null probability selects.

    R computes ``null.prob`` from a covariate with ``goanaTrend`` unless no
    gene is DE (message "No DE genes", then the plain test), and a
    ``null.prob`` switches the test to Wallenius' noncentral
    hypergeometric distribution (``BiasedUrn``), which is not yet ported.
    """
    if covariate is not None and not any(genes.size for genes in de_lists.values()):
        warnings.warn("No DE genes")
        return
    if prior_prob is not None or covariate is not None:
        raise NotImplementedError(
            "prior_prob and covariate need Wallenius' noncentral hypergeometric "
            "test, which requires the BiasedUrn port; see pylimma roadmap."
        )


def _hypergeometric_pvalues(
    S_counts: np.ndarray, S_N: np.ndarray, nde: np.ndarray, NGenes: int
) -> np.ndarray:
    """Vectorised port of the phyper(...,lower.tail=FALSE) loop.

    R: ``PValue[,j] <- phyper(S[,1L+j]-0.5, nde[j], NGenes-nde[j], S[,N], lower.tail=FALSE)``

    R's ``phyper(q, m, n, k, lower.tail=FALSE)`` is ``P(X > q)`` for a
    hypergeometric draw of ``k`` from a pool of ``m`` successes and
    ``n`` failures. With ``q = S - 0.5`` and integer ``S``, this is
    ``P(X >= S)``. ``scipy.stats.hypergeom.sf(k, M, n, N)`` is
    ``P(X > k)``; we set ``k = S - 1`` so it computes ``P(X >= S)``.
    Verified against R numerically.
    """
    # S_counts: (n_pathways, n_de_lists)
    # S_N: (n_pathways,)
    # nde: (n_de_lists,)
    npw, nsets = S_counts.shape
    out = np.zeros_like(S_counts, dtype=np.float64)
    for j in range(nsets):
        # P(X >= s) where X ~ Hypergeom(M=NGenes, n_success=nde[j], N_sample=S_N[i])
        out[:, j] = stats.hypergeom.sf(S_counts[:, j] - 1, NGenes, int(nde[j]), S_N)
    return out


def _pathway_counts(gp_in_universe, de_lists):
    """Genes and DE genes per pathway: R's ``rowsum(X, group, reorder=FALSE)``
    of the incidence matrix ``X`` (a column of ones, then one column per DE
    set), in order of first appearance of each pathway."""
    gene_ids = gp_in_universe["gene_id"]
    X = np.column_stack(
        [np.ones(len(gene_ids), dtype=np.int64)]
        + [
            gene_ids.isin(set(genes.tolist())).to_numpy(dtype=np.int64)
            for genes in de_lists.values()
        ]
    )
    S = pd.DataFrame(X).groupby(gp_in_universe["pathway_id"].to_numpy(), sort=False).sum()
    return S.index.to_numpy(), S[0].to_numpy(dtype=np.int64), S.iloc[:, 1:].to_numpy(dtype=np.int64)


def _add_set_columns(out, set_names, S_counts, PValue) -> pd.DataFrame:
    """Append each set's count and p-value columns (R: ``S`` and ``PValue``).

    Columns are the lower-case set name and ``p_`` plus it. Two sets, or a
    set and a fixed column such as ``n``, that give the same column raise
    an error; R's set names keep their case, so it has no such clash.
    """
    count_cols = [nm.lower() for nm in set_names]
    p_cols = [f"p_{c}" for c in count_cols]
    columns = list(out.columns) + count_cols + p_cols
    clashes = sorted({c for c in columns if columns.count(c) > 1})
    if clashes:
        raise ValueError(
            f"gene set names {list(set_names)} give the same output column(s) "
            f"{clashes}; rename the sets"
        )
    counts = pd.DataFrame(S_counts, index=out.index, columns=count_cols)
    pvalues = pd.DataFrame(PValue, index=out.index, columns=p_cols)
    return pd.concat([out, counts, pvalues], axis=1)


def _goana_default(de, *, gene_pathway, universe, species, prior_prob, covariate, plot, trend):
    """Port of goana.default (goana.R:88-260)."""
    if not isinstance(trend, (bool, np.bool_)) or trend:
        raise NotImplementedError(
            "trend correction requires the BiasedUrn port; see pylimma roadmap."
        )

    # gene_pathway with optional ontology (col 3) + term (col 4) for
    # the goana output.
    gp = _normalise_gene_pathway(gene_pathway, optional_extra_cols=True)
    has_ontology = gp.shape[1] >= 3
    has_term = gp.shape[1] >= 4
    ont_col = gp.columns[2] if has_ontology else None
    term_col = gp.columns[3] if has_term else None

    de_lists = _ensure_de_lists(de)

    if universe is None:
        universe = pd.unique(gp["gene_id"])
        if prior_prob is not None or covariate is not None:
            warnings.warn("Ignoring covariate and null.prob because universe not set")
            prior_prob = None
            covariate = None
    # R: "Make universe and set of all annotated genes agree".
    universe, prior_prob, covariate = _normalise_universe(
        universe, prior_prob, covariate, gp["gene_id"], restrict=True
    )
    NGenes = universe.size
    if NGenes < 1:
        raise ValueError("No annotated genes found in universe")

    # Restrict DE genes to universe.
    universe_set = set(universe.tolist())
    for nm in list(de_lists.keys()):
        de_lists[nm] = np.array([g for g in de_lists[nm] if g in universe_set], dtype=object)

    # Restrict pathways to universe.
    gp_in_universe = gp[gp["gene_id"].isin(universe_set)].reset_index(drop=True)
    if gp_in_universe.empty:
        raise ValueError("Pathways do not overlap with universe")
    _check_null_prob(de_lists, prior_prob, covariate)

    set_names = list(de_lists.keys())
    pathway_ids, S_N, S_counts = _pathway_counts(gp_in_universe, de_lists)
    nde = np.array([de_lists[nm].size for nm in set_names], dtype=np.int64)

    PValue = _hypergeometric_pvalues(S_counts, S_N, nde, NGenes)

    # Term and ontology lookups: use the first occurrence of each
    # pathway_id in the (possibly extended) gene_pathway table.
    first_rows = gp.drop_duplicates(subset=["pathway_id"], keep="first").set_index("pathway_id")
    term_lookup = first_rows[term_col] if has_term else None
    ont_lookup = first_rows[ont_col] if has_ontology else None

    out = pd.DataFrame(index=pathway_ids)
    if has_term:
        out["term"] = [term_lookup.get(p, p) for p in pathway_ids]
    else:
        out["term"] = list(pathway_ids)
    if has_ontology:
        out["ontology"] = [ont_lookup.get(p, np.nan) for p in pathway_ids]
    else:
        out["ontology"] = np.nan
    out["n"] = S_N
    return _add_set_columns(out, set_names, S_counts, PValue)


def _kegga_default(
    de, *, gene_pathway, pathway_names, universe, restrict_universe, species, prior_prob, covariate, plot, trend
):
    """Port of kegga.default (kegga.R:88-280)."""
    if not isinstance(trend, (bool, np.bool_)) or trend:
        raise NotImplementedError(
            "trend correction requires the BiasedUrn port; see pylimma roadmap."
        )

    de_lists = _ensure_de_lists(de)
    gp = _normalise_gene_pathway(gene_pathway, optional_extra_cols=False)

    # Pathway names lookup
    if pathway_names is not None:
        pn = pd.DataFrame(pathway_names).copy()
        if pn.shape[1] < 2:
            raise ValueError("pathway.names must have at least 2 columns")
        pn = pn.rename(columns={pn.columns[0]: "_path_id", pn.columns[1]: "_description"})
        pn = pn.loc[pn[["_path_id", "_description"]].notna().all(axis=1)].astype(str)
        # R: match(g, PathID.PathName[,1]) takes the first row of an ID.
        pn = pn.drop_duplicates(subset="_path_id", keep="first")
        pn_lookup = dict(zip(pn["_path_id"], pn["_description"]))
    else:
        pn_lookup = None

    if universe is None:
        universe = pd.unique(gp["gene_id"])
        if prior_prob is not None or covariate is not None:
            warnings.warn("Ignoring covariate and null.prob because universe not set")
            prior_prob = None
            covariate = None
    elif len(universe) == 0:
        raise ValueError("No genes in universe")
    universe, prior_prob, covariate = _normalise_universe(
        universe, prior_prob, covariate, gp["gene_id"], restrict=restrict_universe
    )
    NGenes = universe.size
    if NGenes < 1:
        raise ValueError("No annotated genes found in universe")

    universe_set = set(universe.tolist())
    for nm in list(de_lists.keys()):
        de_lists[nm] = np.array([g for g in de_lists[nm] if g in universe_set], dtype=object)

    gp_in_universe = gp[gp["gene_id"].isin(universe_set)].reset_index(drop=True)
    if gp_in_universe.empty:
        raise ValueError("Pathways do not overlap with universe")
    _check_null_prob(de_lists, prior_prob, covariate)

    set_names = list(de_lists.keys())
    pathway_ids, S_N, S_counts = _pathway_counts(gp_in_universe, de_lists)
    nde = np.array([de_lists[nm].size for nm in set_names], dtype=np.int64)

    PValue = _hypergeometric_pvalues(S_counts, S_N, nde, NGenes)

    out = pd.DataFrame(index=pathway_ids)
    if pn_lookup is not None:
        out["pathway"] = [pn_lookup.get(p, np.nan) for p in pathway_ids]
    else:
        out["pathway"] = list(pathway_ids)
    out["n"] = S_N
    return _add_set_columns(out, set_names, S_counts, PValue)


# ---------------------------------------------------------------------------
# top_go / top_kegg
# ---------------------------------------------------------------------------


def _order_p_n_name(P, N, names) -> np.ndarray:
    """R's ``order(P, N, name)``: stable, with missing values last. Names
    compare by code point, as R in the C locale (R's order follows the
    user's locale; known_differences.rst)."""
    keys = pd.DataFrame(
        {"p": np.asarray(P), "n": np.asarray(N), "name": np.asarray(names, dtype=object)}
    )
    return keys.sort_values(["p", "n", "name"], kind="stable", na_position="last").index.to_numpy()


def top_go(
    results, ontology=("BP", "CC", "MF"), sort=None, number=20, truncate_term=None, p_value=1.0
) -> pd.DataFrame:
    """
    Extract the top GO terms from a :func:`goana` result.

    Port of R limma's ``topGO``. ``results`` must be a DataFrame in the
    shape returned by :func:`goana` (columns ``term``, ``ontology``,
    ``n``, then one count column per DE list, then one ``p_*`` column
    per DE list).

    ``ontology`` is one or more of ``"BP"``, ``"CC"`` and ``"MF"``, which
    may be abbreviated as in R; unlike R, an element that matches none of
    them raises an error instead of being dropped. ``number`` is capped at
    the number of rows, so ``np.inf`` returns every row.
    """
    if not isinstance(results, pd.DataFrame):
        raise ValueError("results should be a data.frame.")

    # R: match.arg(unique(ontology), several.ok = TRUE), except that an
    # element matching nothing raises rather than being dropped.
    ontology = [ontology] if isinstance(ontology, str) else list(ontology)
    if not ontology:
        raise ValueError("'arg' must be of length >= 1")
    ontology_arr = list(dict.fromkeys(_match_arg(o, ("BP", "CC", "MF"), "arg") for o in ontology))

    if len(ontology_arr) < 3 and "ontology" in results.columns:
        results = results.loc[results["ontology"].isin(ontology_arr)]

    dimres = results.shape

    if isinstance(number, bool) or not isinstance(number, (int, np.integer, float, np.floating)):
        raise ValueError("number should be a positive integer")
    number = _top_row_count(number, dimres[0])
    if number < 1:
        return results.iloc[0:0]

    nsets = (dimres[1] - 3) // 2
    if nsets < 1:
        raise ValueError("results has wrong number of columns")
    setnames = list(results.columns[3 : 3 + nsets])

    if sort is None:
        isort = list(range(nsets))
    else:
        sort_arr = [str(s).lower() for s in ([sort] if isinstance(sort, str) else list(sort))]
        isort = [i for i, nm in enumerate(setnames) if nm.lower() in sort_arr]
        if not isort:
            raise ValueError("sort column not found in results")

    p_cols = [results.columns[3 + nsets + i] for i in isort]
    if len(p_cols) == 1:
        P = results[p_cols[0]].to_numpy(dtype=np.float64)
    else:
        P = results[p_cols].min(axis=1, skipna=False).to_numpy(dtype=np.float64)  # R's pmin

    if p_value < 1:
        number = min(number, int(np.sum(P <= p_value)))
    if number < 1:
        return results.iloc[0:0]

    keep = _order_p_n_name(P, results["n"], results["term"])[:number]
    tab = results.iloc[keep].copy()

    if truncate_term is not None:
        tt = int(np.atleast_1d(truncate_term).ravel()[0])
        tt = max(tt, 5)
        tt = min(tt, 1000)
        tm2 = tt - 3
        new_terms = []
        for v in tab["term"]:
            sv = str(v) if v is not None else ""
            if len(sv) > tm2:
                sv = sv[:tm2] + "..."
            new_terms.append(sv)
        tab["term"] = new_terms

    return tab


def top_kegg(results, sort=None, number=20, truncate_path=None, p_value=1.0) -> pd.DataFrame:
    """
    Extract the top KEGG pathways from a :func:`kegga` result.

    Port of R limma's ``topKEGG``.
    """
    if not isinstance(results, pd.DataFrame):
        raise ValueError("results should be a data.frame.")
    dimres = results.shape

    if isinstance(number, bool) or not isinstance(number, (int, np.integer, float, np.floating)):
        raise ValueError("number should be a positive integer")
    number = _top_row_count(number, dimres[0])
    if number < 1:
        return results.iloc[0:0]

    nsets = (dimres[1] - 2) // 2
    if nsets < 1:
        raise ValueError("results has wrong number of columns")
    setnames = list(results.columns[2 : 2 + nsets])

    if sort is None:
        isort = list(range(nsets))
    else:
        sort_arr = [str(s).lower() for s in ([sort] if isinstance(sort, str) else list(sort))]
        isort = [i for i, nm in enumerate(setnames) if nm.lower() in sort_arr]
        if not isort:
            raise ValueError("sort column not found in results")

    p_cols = [results.columns[2 + nsets + i] for i in isort]
    if len(p_cols) == 1:
        P = results[p_cols[0]].to_numpy(dtype=np.float64)
    else:
        P = results[p_cols].min(axis=1, skipna=False).to_numpy(dtype=np.float64)  # R's pmin

    if p_value < 1:
        number = min(number, int(np.sum(P <= p_value)))
    if number < 1:
        return results.iloc[0:0]

    keep = _order_p_n_name(P, results["n"], results["pathway"])[:number]
    tab = results.iloc[keep].copy()

    if truncate_path is not None:
        tt = int(np.atleast_1d(truncate_path).ravel()[0])
        tt = max(tt, 5)
        tt = min(tt, 1000)
        tm2 = tt - 3
        new_paths = []
        for v in tab["pathway"]:
            sv = str(v) if v is not None else ""
            if len(sv) > tm2:
                sv = sv[:tm2] + "..."
            new_paths.append(sv)
        tab["pathway"] = new_paths

    return tab


# ---------------------------------------------------------------------------
# goana_trend (port of goanaTrend.R)
# ---------------------------------------------------------------------------


def goana_trend(
    index_de,
    covariate,
    n_prior=10,
    plot=False,
    xlab="Covariate Rank",
    ylab="Probability gene is DE",
    main="DE status vs covariate",
) -> np.ndarray:
    """
    Estimate per-gene DE probability from a covariate.

    Port of R limma's ``goanaTrend`` (``goanaTrend.R``). Used internally
    by :func:`goana` and :func:`kegga` for length / abundance bias
    correction. Stand-alone use is supported and validated against R.

    Parameters
    ----------
    index_de : array_like
        The DE genes: a boolean mask the length of ``covariate``, or
        0-based positions (whole numbers between 0 and ``len(covariate) -
        1``). Unlike R, negative, fractional or out-of-range positions
        raise an error.
    covariate : array_like
        Genewise covariate, such as gene length or average log-expression.
    n_prior : float, default 10
        Prior number of genes moderating the trend towards a constant.
    plot : bool, default False
        Draw the trend against covariate rank, with a tick at each DE
        gene, as R (requires matplotlib).
    xlab, ylab, main : str
        Axis labels and title of the plot.
    """
    index_de = np.asarray(index_de)
    if index_de.dtype != bool and np.any(pd.isna(index_de)):
        raise ValueError("index.de should not contain missing values")

    covariate = np.asarray(covariate, dtype=np.float64).copy()
    ngenes = covariate.size
    if ngenes == 0:
        return np.zeros(0, dtype=np.float64)

    p = np.zeros(ngenes, dtype=np.float64)
    isDE = np.zeros(ngenes, dtype=np.float64)
    if index_de.dtype == bool:
        if index_de.size != ngenes:
            raise ValueError("index.de length must match covariate length")
        isDE[index_de] = 1.0
    else:
        # 0-based positions. R excludes a negative position, truncates a
        # fraction and extends isDE past the end; pylimma raises instead.
        positions = index_de.ravel()
        if positions.size and (
            not np.issubdtype(positions.dtype, np.number)
            or np.any(positions != np.round(positions))
        ):
            raise ValueError("index_de positions must be whole numbers (or a boolean mask)")
        if np.any((positions < 0) | (positions >= ngenes)):
            raise ValueError(f"index_de positions must be between 0 and {ngenes - 1}")
        isDE[positions.astype(np.int64)] = 1.0

    nDE = isDE.sum()
    p_mean = max(nDE / ngenes, 1e-5)

    # NA branch in covariate.
    if np.any(np.isnan(covariate)):
        isna = np.isnan(covariate)
        p[isna] = p_mean
        sub_cov = covariate[~isna]
        sub_isde = isDE[~isna].astype(bool)
        # Recurse with bool index aligned to the sub-vector.
        p[~isna] = goana_trend(
            sub_isde, sub_cov, n_prior=n_prior, plot=plot, xlab=xlab, ylab=ylab, main=main
        )
        return p

    # R: o <- order(covariate). Stable for ties (ascending).
    o = np.argsort(covariate, kind="stable")
    isDE_o = isDE[o]
    # span = approx(c(20,200), c(1,0.5), xout=nDE, rule=2, ties=...)$y
    # rule=2 -> clamp at boundaries, NOT linear extrapolation.
    if nDE <= 20:
        span = 1.0
    elif nDE >= 200:
        span = 0.5
    else:
        # Linear interpolation between (20, 1) and (200, 0.5).
        span = 1.0 + (0.5 - 1.0) * (nDE - 20) / (200 - 20)
    p_o = tricube_moving_average(isDE_o, span=span)
    p_o = (p_mean * n_prior + p_o * nDE) / (n_prior + nDE)
    p[o] = p_o
    if plot:
        ylim_top = np.ceil(p.max() * ngenes) / ngenes
        _draw_goana_trend(p_o, isDE_o, ylim_top=ylim_top, xlab=xlab, ylab=ylab, main=main)

    return p


def _draw_goana_trend(p_o, isDE_o, *, ylim_top, xlab, ylab, main) -> None:
    """R's goanaTrend plot: the trend against covariate rank, and a tick
    of height ylim / 9 at the rank of each DE gene."""
    from .plotting import _require_matplotlib

    plt = _require_matplotlib()
    ranks = np.arange(1, p_o.size + 1)
    de_ranks = ranks[isDE_o.astype(bool)]
    _fig, ax = plt.subplots()
    ax.plot(ranks, p_o, color="black", linewidth=2)
    ax.set_ylim(0, ylim_top)
    ax.vlines(de_ranks, 0, ylim_top / 9, colors="black", linewidth=2 if de_ranks.size < 100 else 1)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(main)

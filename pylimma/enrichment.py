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

Phase-1 scope
-------------
This first-cut port focuses on the database-free path. Two cuts versus
R limma:

1. **Bioconductor database lookups dropped.** R limma's ``goana.default``
   reaches into ``GO.db`` / ``org.<species>.eg.db``; ``kegga.default``
   reaches into the live KEGG REST API. pylimma deliberately does not
   wrap Bioconductor annotation databases (see
   ``policy_data_class_wrappers``). Both ports require the caller to
   supply a ``gene_pathway`` data frame mapping gene IDs to pathway IDs
   (the universal interface, fully database-free). The
   ``getGeneKEGGLinks`` and ``getKEGGPathwayNames`` helpers, which exist
   in R only to fetch from the KEGG REST API, are not ported.

2. **BiasedUrn deferred.** R's ``trend=TRUE`` path performs Wallenius'
   noncentral hypergeometric test via ``BiasedUrn::pWNCHypergeo`` /
   ``dWNCHypergeo`` for length / abundance bias correction. BiasedUrn
   is GPL-2-or-later (compatible) but a substantial side-quest and is
   staged to a separate Phase 2. Phase 1 raises
   ``NotImplementedError`` whenever ``trend`` is truthy or a covariate
   is supplied to a ``*.default`` method - silently falling back to
   plain hypergeometric would be wrong-but-plausible-looking.

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
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .classes import MArrayLM, _is_anndata, _resolve_fit_input
from .utils import p_adjust, tricube_moving_average


# ---------------------------------------------------------------------------
# Public dispatchers
# ---------------------------------------------------------------------------


def goana(de, gene_pathway, universe=None, *,
          species=None, prior_prob=None,
          covariate=None, plot=False,
          fdr=0.05, trend=False, **kwargs) -> pd.DataFrame:
    """
    Gene-ontology over-representation analysis.

    Port of R limma's ``goana`` (``goana.R``). See module docstring for
    the gene_pathway format and the Phase-1 scope cuts.

    Parameters
    ----------
    de : MArrayLM, AnnData, dict-of-vectors, or 1-D vector of gene IDs
        Differentially-expressed genes. When a fit object is supplied,
        the up- and down-regulated gene sets are derived internally
        (BH adjustment at level ``fdr``).
    gene_pathway : DataFrame
        Mapping from gene IDs to GO terms. See module docstring.
    universe : sequence of str, optional
        Background gene universe. Defaults to all gene IDs in
        ``gene_pathway``. Required - without GO.db there is no
        meaningful default beyond the supplied gene_pathway.
    species : str, optional
        Accepted for R-API compatibility; ignored without GO.db.
    prior_prob : sequence of float, optional
        Per-gene null DE probability (R name: ``null.prob``).
    covariate : sequence of float, optional
        Covariate values aligned to the universe; if supplied,
        ``trend`` becomes implicit and ``goana_trend`` is run.
        Phase 1 raises ``NotImplementedError``.
    plot : bool, default False
        Forwarded to ``goana_trend`` when applicable.
    fdr : float, default 0.05
        FDR threshold for selecting up/down DE genes from a fit object
        (R name: ``FDR``). Only used in the MArrayLM branch.
    trend : bool, numeric, or str, default False
        Use length / abundance bias correction. Phase 1 raises
        ``NotImplementedError`` for any truthy / numeric / character
        value.
    **kwargs
        ``coef`` and ``geneid`` for the MArrayLM branch.
    """
    if _is_dispatch_marraylm(de):
        return _goana_marraylm(de, gene_pathway=gene_pathway,
                               universe=universe, species=species,
                               prior_prob=prior_prob, covariate=covariate,
                               plot=plot, fdr=fdr, trend=trend, **kwargs)
    return _goana_default(de, gene_pathway=gene_pathway,
                          universe=universe, species=species,
                          prior_prob=prior_prob, covariate=covariate,
                          plot=plot, trend=trend)


def kegga(de, gene_pathway, pathway_names=None, universe=None, *,
          species=None, prior_prob=None,
          covariate=None, plot=False,
          fdr=0.05, trend=False, **kwargs) -> pd.DataFrame:
    """
    KEGG pathway over-representation analysis.

    Port of R limma's ``kegga`` (``kegga.R``). See module docstring for
    the gene_pathway format and the Phase-1 scope cuts.

    Parameters
    ----------
    de, gene_pathway, universe, species, prior_prob, covariate, plot,
    fdr, trend, **kwargs
        See :func:`goana`. Identical semantics.
    pathway_names : DataFrame, optional
        Two-column mapping from pathway id (column 0) to human-readable
        description (column 1). If absent, the output's ``pathway``
        column repeats the pathway id verbatim.
    """
    if _is_dispatch_marraylm(de):
        return _kegga_marraylm(de, gene_pathway=gene_pathway,
                               pathway_names=pathway_names,
                               universe=universe, species=species,
                               prior_prob=prior_prob, covariate=covariate,
                               plot=plot, fdr=fdr, trend=trend, **kwargs)
    return _kegga_default(de, gene_pathway=gene_pathway,
                          pathway_names=pathway_names,
                          universe=universe, species=species,
                          prior_prob=prior_prob, covariate=covariate,
                          plot=plot, trend=trend)


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


def _select_de_from_fit(fit, geneid, coef, fdr) -> tuple[list[str], list[str], list[str]]:
    """Return (universe, up_ids, down_ids) from an MArrayLM-shaped fit.

    Mirrors lines 13-77 of goana.R / kegga.R: validate the fit, resolve
    geneid (vector vs. column-name reference), select FDR-significant
    up/down genes via BH at level ``fdr``.
    """
    if fit.get("coefficients") is None:
        raise ValueError("de does not appear to be a valid MArrayLM fit object.")
    if fit.get("p_value") is None:
        raise ValueError(
            "p.value not found in fit object, perhaps need to run e_bayes first."
        )

    coefs = np.asarray(fit["coefficients"])
    if coefs.ndim == 1:
        coefs = coefs.reshape(-1, 1)
    pvals = np.asarray(fit["p_value"])
    if pvals.ndim == 1:
        pvals = pvals.reshape(-1, 1)
    ngenes, ncoef = coefs.shape

    if coef is None:
        coef = ncoef - 1
    if hasattr(coef, "__len__") and len(coef) != 1:
        raise ValueError("Only one coef can be specified.")
    if hasattr(coef, "__len__"):
        coef = coef[0]
    if isinstance(coef, str):
        coef_names = list(fit.get("coef_names") or [])
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
    # in fit['genes'].
    if geneid is None:
        genes = fit.get("genes")
        if genes is not None and hasattr(genes, "index"):
            universe = [str(v) for v in genes.index]
        else:
            universe = [str(i) for i in range(ngenes)]
    elif isinstance(geneid, str) or (
        hasattr(geneid, "__len__") and len(geneid) == 1
        and not isinstance(geneid, (np.ndarray, pd.Series))
    ):
        col = geneid if isinstance(geneid, str) else geneid[0]
        genes = fit.get("genes")
        if genes is None or col not in getattr(genes, "columns", []):
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
    isna = np.array([
        v is None or (isinstance(v, float) and np.isnan(v))
        or v == "" or v == "nan" or v == "NA"
        for v in universe
    ])
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


def _goana_marraylm(de, *, gene_pathway, universe, species,
                    prior_prob, covariate, plot, fdr, trend, **kwargs):
    """Port of goana.MArrayLM (goana.R:3-86)."""
    fit, _adata, _adata_key = _resolve_fit_input(de, kwargs.pop("key", "fit"))

    if "universe" in kwargs:
        raise ValueError("goana.MArrayLM defines its own universe")
    if (not isinstance(trend, bool) or trend) and "covariate" in kwargs:
        raise ValueError("goana.MArrayLM defines its own covariate")

    geneid = kwargs.pop("geneid", None)
    coef = kwargs.pop("coef", None)
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")

    if not isinstance(trend, bool):
        # R accepts numeric or character trend; both go through the
        # BiasedUrn path which is deferred.
        raise NotImplementedError(
            "trend correction requires the BiasedUrn port; "
            "see pylimma roadmap."
        )

    fit_universe, up, dn = _select_de_from_fit(fit, geneid, coef, fdr)

    if not up and not dn:
        warnings.warn("No DE genes")
        return pd.DataFrame()

    return _goana_default({"Up": up, "Down": dn},
                          gene_pathway=gene_pathway,
                          universe=fit_universe,
                          species=species,
                          prior_prob=prior_prob,
                          covariate=covariate,
                          plot=plot,
                          trend=trend)


def _kegga_marraylm(de, *, gene_pathway, pathway_names, universe, species,
                    prior_prob, covariate, plot, fdr, trend, **kwargs):
    """Port of kegga.MArrayLM (kegga.R:3-86)."""
    fit, _adata, _adata_key = _resolve_fit_input(de, kwargs.pop("key", "fit"))

    if "universe" in kwargs:
        raise ValueError("kegga.MArrayLM defines its own universe")
    if (not isinstance(trend, bool) or trend) and "covariate" in kwargs:
        raise ValueError("kegga.MArrayLM defines its own covariate")

    geneid = kwargs.pop("geneid", None)
    coef = kwargs.pop("coef", None)
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")

    if not isinstance(trend, bool):
        raise NotImplementedError(
            "trend correction requires the BiasedUrn port; "
            "see pylimma roadmap."
        )

    fit_universe, up, dn = _select_de_from_fit(fit, geneid, coef, fdr)

    if not up and not dn:
        warnings.warn("No DE genes")
        return pd.DataFrame()

    return _kegga_default({"Up": up, "Down": dn},
                          gene_pathway=gene_pathway,
                          pathway_names=pathway_names,
                          universe=fit_universe,
                          species=species,
                          prior_prob=prior_prob,
                          covariate=covariate,
                          plot=plot,
                          trend=trend)


# ---------------------------------------------------------------------------
# default-method helpers
# ---------------------------------------------------------------------------


def _ensure_de_lists(de) -> dict[str, np.ndarray]:
    """Port of the R 'ensure de is a list, dedupe, name uniquely' block.

    Mirrors goana.R:160-179 and kegga.R:93-112.
    """
    if isinstance(de, dict):
        names = list(de.keys())
        sets = list(de.values())
    elif hasattr(de, "__iter__") and not isinstance(de, (str, bytes)):
        names = ["DE"]
        sets = [list(de)]
    else:
        raise ValueError("components of de should be vectors")

    out: dict[str, np.ndarray] = {}
    cleaned_names: list[str] = []
    for nm, vec in zip(names, sets):
        if isinstance(vec, pd.DataFrame):
            raise ValueError(
                "de should be a list of character vectors. "
                "It should not be a data.frame."
            )
        try:
            arr = np.asarray(vec).ravel()
        except Exception as e:
            raise ValueError("components of de should be vectors") from e
        # Unique-stable, R semantics: order of first appearance, cast to str.
        seen = set()
        uniq = []
        for v in arr:
            sv = str(v)
            if sv not in seen:
                seen.add(sv)
                uniq.append(sv)
        cleaned_names.append("" if nm is None else str(nm).strip())
        out[nm] = np.array(uniq, dtype=object)

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
        nm: 1 + int(np.floor(np.log10(c))) if c > 1 else 0
        for nm, c in seen_count.items()
    }
    occ: dict[str, int] = {nm: 0 for nm in seen_count}
    disambiguated: list[str] = []
    for nm in final_names:
        if seen_count[nm] > 1:
            occ[nm] += 1
            disambiguated.append(f"{nm}{occ[nm]:0{width[nm]}d}")
        else:
            disambiguated.append(nm)

    keyed = {dn: out[on] for dn, on in zip(disambiguated, list(out.keys()))}
    return keyed


def _normalise_universe(universe, prior_prob, covariate, gene_pathway_first_col):
    """Apply R's NA-removal, dedup, and (optional) restrict-to-pathway logic.

    Mirrors goana.R:122-156 and kegga.R:155-202 for the explicit-universe
    path (the implicit / database-built path doesn't apply here).
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
    isna = np.array([u in ("", "nan", "NA", "None") for u in universe]) | (universe == "")
    # numpy strings compare equal to "nan" already; explicit check above is
    # for object-dtype inputs.
    if covariate is not None:
        isna |= np.isnan(covariate)
    if prior_prob is not None:
        isna |= np.isnan(prior_prob)
    if isna.all():
        raise ValueError("Gene IDs are all NA")
    if isna.any():
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
            "gene_pathway is required; pylimma does not bundle GO.db / "
            "KEGG REST lookups (see policy_data_class_wrappers)."
        )
    df = pd.DataFrame(gene_pathway).copy()
    if df.shape[1] < 2:
        raise ValueError("gene.pathway must have at least 2 columns")
    cols = list(df.columns)
    df = df.rename(columns={cols[0]: "gene_id", cols[1]: "pathway_id"})
    df["gene_id"] = df["gene_id"].astype(str)
    df["pathway_id"] = df["pathway_id"].astype(str)

    # Drop rows where either of the first two cols is NA
    isna = df[["gene_id", "pathway_id"]].isna().any(axis=1) | \
        (df["gene_id"].isin(["", "nan", "NA"])) | \
        (df["pathway_id"].isin(["", "nan", "NA"]))
    df = df.loc[~isna].copy()

    # Drop duplicate (gene_id, pathway_id) rows
    df = df.drop_duplicates(subset=["gene_id", "pathway_id"], keep="first")

    if not optional_extra_cols and df.shape[1] > 2:
        df = df.iloc[:, :2]
    return df


def _hypergeometric_pvalues(S_counts: np.ndarray, S_N: np.ndarray,
                            nde: np.ndarray, NGenes: int) -> np.ndarray:
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
        out[:, j] = stats.hypergeom.sf(
            S_counts[:, j] - 1, NGenes, int(nde[j]), S_N
        )
    return out


def _goana_default(de, *, gene_pathway, universe, species, prior_prob,
                   covariate, plot, trend):
    """Port of goana.default (goana.R:88-260)."""
    if not isinstance(trend, bool) or trend:
        raise NotImplementedError(
            "trend correction requires the BiasedUrn port; "
            "see pylimma roadmap."
        )
    if covariate is not None:
        raise NotImplementedError(
            "trend correction requires the BiasedUrn port; "
            "see pylimma roadmap."
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
    universe, prior_prob, _covariate = _normalise_universe(
        universe, prior_prob, covariate, gp["gene_id"]
    )
    NGenes = universe.size
    if NGenes < 1:
        raise ValueError("No annotated genes found in universe")

    # Restrict DE genes to universe.
    universe_set = set(universe.tolist())
    for nm in list(de_lists.keys()):
        de_lists[nm] = np.array(
            [g for g in de_lists[nm] if g in universe_set], dtype=object
        )

    # Restrict pathways to universe.
    gp_in_universe = gp[gp["gene_id"].isin(universe_set)].reset_index(drop=True)
    if gp_in_universe.empty:
        raise ValueError("Pathways do not overlap with universe")

    # Build incidence matrix in R's column order: N, then one column per
    # DE list. ``rowsum(X, group, reorder=FALSE)`` in pandas is a groupby
    # on the pathway id with sort=False to preserve first-occurrence
    # order.
    set_names = list(de_lists.keys())
    nsets = len(set_names)
    columns = ["N"] + set_names
    X = pd.DataFrame(0, index=gp_in_universe.index, columns=columns,
                     dtype=np.int64)
    X["N"] = 1
    for nm in set_names:
        members = set(de_lists[nm].tolist())
        X[nm] = gp_in_universe["gene_id"].isin(members).astype(np.int64)

    grouped = X.groupby(gp_in_universe["pathway_id"].values, sort=False).sum()
    pathway_ids = grouped.index.to_numpy()

    S_N = grouped["N"].to_numpy(dtype=np.int64)
    S_counts = grouped[set_names].to_numpy(dtype=np.int64)
    nde = np.array([de_lists[nm].size for nm in set_names], dtype=np.int64)

    PValue = _hypergeometric_pvalues(S_counts, S_N, nde, NGenes)

    # Term and ontology lookups: use the first occurrence of each
    # pathway_id in the (possibly extended) gene_pathway table.
    first_rows = (
        gp.drop_duplicates(subset=["pathway_id"], keep="first")
          .set_index("pathway_id")
    )
    term_lookup = (
        first_rows[term_col] if has_term else None
    )
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
    for j, nm in enumerate(set_names):
        out[nm.lower()] = S_counts[:, j]
    for j, nm in enumerate(set_names):
        out[f"p_{nm.lower()}"] = PValue[:, j]
    return out


def _kegga_default(de, *, gene_pathway, pathway_names, universe, species,
                   prior_prob, covariate, plot, trend):
    """Port of kegga.default (kegga.R:88-280)."""
    if not isinstance(trend, bool) or trend:
        raise NotImplementedError(
            "trend correction requires the BiasedUrn port; "
            "see pylimma roadmap."
        )
    if covariate is not None:
        raise NotImplementedError(
            "trend correction requires the BiasedUrn port; "
            "see pylimma roadmap."
        )

    de_lists = _ensure_de_lists(de)
    gp = _normalise_gene_pathway(gene_pathway, optional_extra_cols=False)

    # Pathway names lookup
    if pathway_names is not None:
        pn = pd.DataFrame(pathway_names).copy()
        if pn.shape[1] < 2:
            raise ValueError("pathway.names must have at least 2 columns")
        pn = pn.rename(columns={pn.columns[0]: "_path_id",
                                pn.columns[1]: "_description"})
        pn["_path_id"] = pn["_path_id"].astype(str)
        pn["_description"] = pn["_description"].astype(str)
        pn_isna = pn[["_path_id", "_description"]].isna().any(axis=1)
        pn = pn.loc[~pn_isna].copy()
        pn_lookup = dict(zip(pn["_path_id"], pn["_description"]))
    else:
        pn_lookup = None

    if universe is None:
        universe = pd.unique(gp["gene_id"])
        if prior_prob is not None or covariate is not None:
            warnings.warn("Ignoring covariate and null.prob because universe not set")
            prior_prob = None
            covariate = None
    universe, prior_prob, _covariate = _normalise_universe(
        universe, prior_prob, covariate, gp["gene_id"]
    )
    NGenes = universe.size
    if NGenes < 1:
        raise ValueError("No annotated genes found in universe")

    universe_set = set(universe.tolist())
    for nm in list(de_lists.keys()):
        de_lists[nm] = np.array(
            [g for g in de_lists[nm] if g in universe_set], dtype=object
        )

    gp_in_universe = gp[gp["gene_id"].isin(universe_set)].reset_index(drop=True)
    if gp_in_universe.empty:
        raise ValueError("Pathways do not overlap with universe")

    set_names = list(de_lists.keys())
    nsets = len(set_names)
    columns = ["N"] + set_names
    X = pd.DataFrame(0, index=gp_in_universe.index, columns=columns,
                     dtype=np.int64)
    X["N"] = 1
    for nm in set_names:
        members = set(de_lists[nm].tolist())
        X[nm] = gp_in_universe["gene_id"].isin(members).astype(np.int64)
    grouped = X.groupby(gp_in_universe["pathway_id"].values, sort=False).sum()
    pathway_ids = grouped.index.to_numpy()

    S_N = grouped["N"].to_numpy(dtype=np.int64)
    S_counts = grouped[set_names].to_numpy(dtype=np.int64)
    nde = np.array([de_lists[nm].size for nm in set_names], dtype=np.int64)

    PValue = _hypergeometric_pvalues(S_counts, S_N, nde, NGenes)

    out = pd.DataFrame(index=pathway_ids)
    if pn_lookup is not None:
        out["pathway"] = [pn_lookup.get(p, np.nan) for p in pathway_ids]
    else:
        out["pathway"] = list(pathway_ids)
    out["n"] = S_N
    for j, nm in enumerate(set_names):
        out[nm.lower()] = S_counts[:, j]
    for j, nm in enumerate(set_names):
        out[f"p_{nm.lower()}"] = PValue[:, j]
    return out


# ---------------------------------------------------------------------------
# top_go / top_kegg
# ---------------------------------------------------------------------------


def top_go(results, ontology=("BP", "CC", "MF"), sort=None,
           number=20, truncate_term=None, p_value=1.0) -> pd.DataFrame:
    """
    Extract the top GO terms from a :func:`goana` result.

    Port of R limma's ``topGO``. ``results`` must be a DataFrame in the
    shape returned by :func:`goana` (columns ``term``, ``ontology``,
    ``n``, then one count column per DE list, then one ``p_*`` column
    per DE list).
    """
    if not isinstance(results, pd.DataFrame):
        raise ValueError("results should be a data.frame.")

    valid_ont = {"BP", "CC", "MF"}
    ontology_arr = list(dict.fromkeys(ontology))
    for o in ontology_arr:
        if o not in valid_ont:
            raise ValueError(
                f"'arg' should be one of 'BP', 'CC', 'MF': got {o!r}"
            )

    if len(ontology_arr) < 3 and "ontology" in results.columns:
        results = results.loc[results["ontology"].isin(ontology_arr)]

    dimres = results.shape

    if not isinstance(number, (int, np.integer, float, np.floating)):
        raise ValueError("number should be a positive integer")
    number = int(number)
    if number > dimres[0]:
        number = dimres[0]
    if number < 1:
        return results.iloc[0:0]

    nsets = (dimres[1] - 3) // 2
    if nsets < 1:
        raise ValueError("results has wrong number of columns")
    setnames = list(results.columns[3:3 + nsets])

    if sort is None:
        isort = list(range(nsets))
    else:
        sort_arr = [str(s).lower() for s in (
            [sort] if isinstance(sort, str) else list(sort)
        )]
        isort = [i for i, nm in enumerate(setnames) if nm.lower() in sort_arr]
        if not isort:
            raise ValueError("sort column not found in results")

    p_cols = [results.columns[3 + nsets + i] for i in isort]
    if len(p_cols) == 1:
        P = results[p_cols[0]].to_numpy(dtype=np.float64)
    else:
        P = results[p_cols].min(axis=1).to_numpy(dtype=np.float64)

    if p_value < 1:
        number = min(number, int(np.sum(P <= p_value)))
    if number < 1:
        return results.iloc[0:0]

    N_arr = results["n"].to_numpy()
    Term_arr = results["term"].to_numpy()
    # R: order(P, results$N, results$Term) - stable multi-key sort.
    order = np.lexsort((Term_arr, N_arr, P))
    keep = order[:number]
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


def top_kegg(results, sort=None, number=20,
             truncate_path=None, p_value=1.0) -> pd.DataFrame:
    """
    Extract the top KEGG pathways from a :func:`kegga` result.

    Port of R limma's ``topKEGG``.
    """
    if not isinstance(results, pd.DataFrame):
        raise ValueError("results should be a data.frame.")
    dimres = results.shape

    if not isinstance(number, (int, np.integer, float, np.floating)):
        raise ValueError("number should be a positive integer")
    number = int(number)
    if number > dimres[0]:
        number = dimres[0]
    if number < 1:
        return results.iloc[0:0]

    nsets = (dimres[1] - 2) // 2
    if nsets < 1:
        raise ValueError("results has wrong number of columns")
    setnames = list(results.columns[2:2 + nsets])

    if sort is None:
        isort = list(range(nsets))
    else:
        sort_arr = [str(s).lower() for s in (
            [sort] if isinstance(sort, str) else list(sort)
        )]
        isort = [i for i, nm in enumerate(setnames) if nm.lower() in sort_arr]
        if not isort:
            raise ValueError("sort column not found in results")

    p_cols = [results.columns[2 + nsets + i] for i in isort]
    if len(p_cols) == 1:
        P = results[p_cols[0]].to_numpy(dtype=np.float64)
    else:
        P = results[p_cols].min(axis=1).to_numpy(dtype=np.float64)

    if p_value < 1:
        number = min(number, int(np.sum(P <= p_value)))
    if number < 1:
        return results.iloc[0:0]

    N_arr = results["n"].to_numpy()
    Path_arr = results["pathway"].to_numpy()
    order = np.lexsort((Path_arr, N_arr, P))
    keep = order[:number]
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


def goana_trend(index_de, covariate, n_prior=10, plot=False) -> np.ndarray:
    """
    Estimate per-gene DE probability from a covariate.

    Port of R limma's ``goanaTrend`` (``goanaTrend.R``). Used internally
    by :func:`goana` and :func:`kegga` for length / abundance bias
    correction. Stand-alone use is supported and validated against R.
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
        # 0-based integer indices.
        idx_int = np.asarray(index_de, dtype=np.int64).ravel()
        isDE[idx_int] = 1.0

    nDE = isDE.sum()
    p_mean = max(nDE / ngenes, 1e-5)

    # NA branch in covariate.
    if np.any(np.isnan(covariate)):
        isna = np.isnan(covariate)
        p[isna] = p_mean
        sub_cov = covariate[~isna]
        sub_isde = isDE[~isna].astype(bool)
        # Recurse with bool index aligned to the sub-vector.
        p[~isna] = goana_trend(sub_isde, sub_cov, n_prior=n_prior, plot=plot)
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

    return p

"""
Tests for pylimma.classes: EList, MArrayLM, get_eawp, put_eawp.

Test strategy
-------------
The classes are dict subclasses. Python already guarantees that dict
subclasses support []-access, attribute-as-key, and isinstance(_, dict) -
testing those things is rubber-stamping. This file tests only behaviour
that is NOT free from the language:

1. R-parity of [i, j] subsetting, slot-by-slot, against R fixtures
   (Part 1). This is where the EList/MArrayLM slot-classification tables
   earn their keep - and where they can break silently.
2. Polymorphic-input equivalence: voom/normalize/weights/duplicate_correlation
   must produce numerically identical output whether input is ndarray,
   dict, EList, or AnnData. Bugs in get_eawp/put_eawp surface here.
3. Dispatcher behavioural contracts: error paths, write-back locations,
   slot preservation across the put_eawp round-trip.
4. A handful of back-compat / corner-case branches in get_eawp that are
   hit in practice but not by the equivalence tests above.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylimma import EList, MArrayLM, get_eawp, put_eawp

FIXTURES = Path(__file__).parent / "fixtures"


# -----------------------------------------------------------------------------
# Part 1: R-parity for [i, j] subsetting
# -----------------------------------------------------------------------------


def _load_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, index_col=0).values


def _load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def _build_elist_from_fixtures() -> EList:
    E = _load_matrix(FIXTURES / "R_elist_full_E.csv")
    W = _load_matrix(FIXTURES / "R_elist_full_weights.csv")
    genes = _load_df(FIXTURES / "R_elist_full_genes.csv")
    targets = _load_df(FIXTURES / "R_elist_full_targets.csv")
    design = _load_matrix(FIXTURES / "R_elist_full_design.csv")
    gene_names = list(genes.index)
    sample_names = list(targets.index)
    return EList(
        {
            "E": pd.DataFrame(E, index=gene_names, columns=sample_names),
            "weights": pd.DataFrame(W, index=gene_names, columns=sample_names),
            "genes": genes,
            "targets": targets,
            "design": pd.DataFrame(design, index=sample_names, columns=["Intercept", "groupB"]),
        }
    )


ELIST_SUBSET_CASES = [
    ("full", slice(None), slice(None)),
    ("rows", slice(0, 10), slice(None)),
    ("cols", slice(None), slice(0, 4)),
    ("both", slice(0, 10), slice(0, 4)),
    ("rowstr", ["gene3", "gene7", "gene15"], slice(None)),
    (
        "rowbool",
        np.array([False, True, False, True, False, True, False, True, False, True] + [False] * 20),
        slice(None),
    ),
]


@pytest.mark.parametrize("tag,i,j", ELIST_SUBSET_CASES)
def test_elist_subset_parity_numerical(tag, i, j):
    """Slot-by-slot numeric parity of EList[i, j] against R's [.EList."""
    el = _build_elist_from_fixtures()
    sub = el if tag == "full" else el[i, j]

    np.testing.assert_allclose(
        np.asarray(sub.E),
        _load_matrix(FIXTURES / f"R_elist_{tag}_E.csv"),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(sub.weights),
        _load_matrix(FIXTURES / f"R_elist_{tag}_weights.csv"),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(sub.design),
        _load_matrix(FIXTURES / f"R_elist_{tag}_design.csv"),
        rtol=1e-6,
        atol=1e-12,
    )


def test_elist_column_subset_preserves_gene_aligned_slots():
    """IX slots (genes) must be unchanged when only columns are subsetted.

    This is the subsetting-rule that breaks most easily: a naive
    implementation would slice `genes` by j, corrupting the gene annotation.
    """
    el = _build_elist_from_fixtures()
    full_genes = el.genes.copy()
    sub = el[:, 0:4]
    pd.testing.assert_frame_equal(sub.genes, full_genes)
    # And conversely, targets/design should shrink
    assert sub.targets.shape[0] == 4
    assert np.asarray(sub.design).shape == (4, 2)


def test_elist_row_subset_preserves_sample_aligned_slots():
    """JX slots (targets, design) must be unchanged when only rows are
    subsetted."""
    el = _build_elist_from_fixtures()
    full_targets = el.targets.copy()
    full_design = np.asarray(el.design).copy()
    sub = el[0:10, :]
    pd.testing.assert_frame_equal(sub.targets, full_targets)
    np.testing.assert_array_equal(np.asarray(sub.design), full_design)


def test_elist_subset_returns_elist_not_dict():
    """Subsetting must preserve the class - otherwise every downstream
    method (head/tail/further subsetting) breaks."""
    el = _build_elist_from_fixtures()
    assert isinstance(el[0:5, :], EList)
    assert isinstance(el[:, 0:3], EList)
    assert isinstance(el[0:5, 0:3], EList)


def _build_marraylm_from_fixtures() -> MArrayLM:
    coef = _load_matrix(FIXTURES / "R_marraylm_full_coefficients.csv")
    stdev = _load_matrix(FIXTURES / "R_marraylm_full_stdev_unscaled.csv")
    tstat = _load_matrix(FIXTURES / "R_marraylm_full_t.csv")
    pval = _load_matrix(FIXTURES / "R_marraylm_full_p_value.csv")
    lods = _load_matrix(FIXTURES / "R_marraylm_full_lods.csv")
    i_slots = _load_df(FIXTURES / "R_marraylm_full_i_slots.csv")
    genes = _load_df(FIXTURES / "R_marraylm_full_genes.csv")
    gene_names = list(i_slots.index)
    return MArrayLM(
        {
            "coefficients": pd.DataFrame(coef, index=gene_names, columns=["Intercept", "groupB"]),
            "stdev_unscaled": pd.DataFrame(
                stdev, index=gene_names, columns=["Intercept", "groupB"]
            ),
            "t": pd.DataFrame(tstat, index=gene_names, columns=["Intercept", "groupB"]),
            "p_value": pd.DataFrame(pval, index=gene_names, columns=["Intercept", "groupB"]),
            "lods": pd.DataFrame(lods, index=gene_names, columns=["Intercept", "groupB"]),
            "Amean": i_slots["Amean"].values,
            "sigma": i_slots["sigma"].values,
            "df_residual": i_slots["df_residual"].values,
            "df_total": i_slots["df_total"].values,
            "s2_post": i_slots["s2_post"].values,
            "genes": genes,
        }
    )


MARRAYLM_SUBSET_CASES = [
    ("rows", slice(0, 10), slice(None)),
    ("rowstr", ["gene3", "gene7", "gene15"], slice(None)),
    ("cols", slice(None), [1]),
    ("both", slice(0, 10), [1]),
]


@pytest.mark.parametrize("tag,i,j", MARRAYLM_SUBSET_CASES)
def test_marraylm_subset_parity_numerical(tag, i, j):
    """MArrayLM[i, j] slot-by-slot numeric parity against R's [.MArrayLM."""
    m = _build_marraylm_from_fixtures()
    sub = m[i, j]
    np.testing.assert_allclose(
        np.asarray(sub.coefficients),
        _load_matrix(FIXTURES / f"R_marraylm_{tag}_coefficients.csv"),
        rtol=1e-6,
        atol=1e-12,
    )
    exp_i = _load_df(FIXTURES / f"R_marraylm_{tag}_i_slots.csv")
    np.testing.assert_allclose(np.asarray(sub.Amean), exp_i["Amean"].values, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(np.asarray(sub.sigma), exp_i["sigma"].values, rtol=1e-6, atol=1e-12)


def test_marraylm_column_subset_leaves_gene_scalar_slots_unchanged():
    """The I slot class (Amean/sigma/df_residual - one scalar per gene)
    must be untouched by column subsetting. Easy to break by mis-classifying
    these into the IJ group."""
    m = _build_marraylm_from_fixtures()
    full_sigma = np.asarray(m.sigma).copy()
    full_df = np.asarray(m.df_residual).copy()
    sub = m[:, [0]]
    np.testing.assert_array_equal(np.asarray(sub.sigma), full_sigma)
    np.testing.assert_array_equal(np.asarray(sub.df_residual), full_df)
    assert np.asarray(sub.coefficients).shape == (m.nrow, 1)
    assert np.asarray(sub.stdev_unscaled).shape == (m.nrow, 1)


def test_head_returns_correct_rows():
    """Exercises _subset via head(), which should produce identical rows
    to manual slicing."""
    el = _build_elist_from_fixtures()
    h = el.head(5)
    assert isinstance(h, EList)
    np.testing.assert_array_equal(np.asarray(h.E), np.asarray(el.E)[:5])


# -----------------------------------------------------------------------------
# Part 2: polymorphic-input (ndarray / EList / AnnData) equivalence for voom,
# normalize_between_arrays, array_weights and duplicate_correlation
# -----------------------------------------------------------------------------
# If get_eawp/put_eawp have a bug (missing transpose, lost weights,
# clobbered design), the numerics diverge. These are the load-bearing
# tests for the dispatchers.


def _test_counts():
    rng = np.random.default_rng(42)
    counts = rng.poisson(20, (100, 8)).astype(float)
    design = np.column_stack([np.ones(8), np.array([0] * 4 + [1] * 4)])
    return counts, design


def test_voom_ndarray_elist_anndata_numerically_identical():
    pytest.importorskip("anndata")
    from anndata import AnnData

    from pylimma import voom

    counts, design = _test_counts()

    # Three routes to the same computation
    v_arr = voom(counts, design)
    v_el = voom(EList({"E": counts, "design": design}))
    adata = AnnData(X=counts.T)
    voom(adata, design)  # mutates

    # All three must agree to machine precision on both E and weights
    np.testing.assert_allclose(v_el.E, v_arr["E"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v_el.weights, v_arr["weights"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(adata.layers["voom_E"].T, v_arr["E"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        adata.layers["voom_weights"].T, v_arr["weights"], rtol=1e-12, atol=1e-12
    )


def test_voom_anndata_uns_contains_design_and_libsize():
    """AnnData write-back must include ancillary metadata in uns, not just
    the two layer matrices."""
    pytest.importorskip("anndata")
    from anndata import AnnData

    from pylimma import voom

    counts, design = _test_counts()
    adata = AnnData(X=counts.T)
    voom(adata, design)
    uns = adata.uns["voom"]
    assert "design" in uns
    assert "lib_size" in uns
    np.testing.assert_array_equal(uns["design"], design)


def test_normalize_actually_transforms_the_data():
    """Verify that normalize_between_arrays is actually doing something
    non-trivial. Without this, the equivalence tests below could all pass
    trivially if the function returned its input unchanged."""
    from pylimma import normalize_between_arrays

    rng = np.random.default_rng(7)
    E = rng.standard_normal((100, 6))
    E[:, 0] += 5  # skew one sample so quantile normalisation has work to do

    out = normalize_between_arrays(E, method="quantile")
    # Output must differ meaningfully from input
    assert not np.allclose(out, E)
    # Quantile normalisation makes column distributions identical -
    # sorted columns should match exactly
    sorted_cols = np.sort(out, axis=0)
    for j in range(1, out.shape[1]):
        np.testing.assert_allclose(sorted_cols[:, j], sorted_cols[:, 0], rtol=1e-12, atol=1e-12)


def test_normalize_ndarray_elist_anndata_numerically_identical():
    pytest.importorskip("anndata")
    from anndata import AnnData

    from pylimma import normalize_between_arrays

    rng = np.random.default_rng(3)
    E = rng.standard_normal((50, 6))
    E[:, 0] += 2  # make normalisation non-trivial

    out_arr = normalize_between_arrays(E, method="quantile")
    out_el = normalize_between_arrays(EList({"E": E}), method="quantile")
    adata = AnnData(X=E.T)
    result = normalize_between_arrays(adata, method="quantile")

    assert result is None
    np.testing.assert_allclose(out_el.E, out_arr, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(adata.layers["normalized"].T, out_arr, rtol=1e-12, atol=1e-12)


def test_array_weights_equivalence_across_inputs():
    """array_weights must yield identical sample-level weights for
    equivalent ndarray / EList / dict / AnnData input."""
    pytest.importorskip("anndata")
    from anndata import AnnData

    from pylimma import array_weights, voom

    counts, design = _test_counts()
    v = voom(counts, design)  # dict

    aw_dict = array_weights(v)
    aw_elist = array_weights(EList({"E": v["E"], "weights": v["weights"], "design": design}))
    aw_arr = array_weights(v["E"], design=design, weights=v["weights"])

    adata = AnnData(X=v["E"].T)
    adata.layers["weights"] = v["weights"].T
    aw_adata = array_weights(adata, design=design, layer=None, weights=v["weights"])

    np.testing.assert_allclose(aw_elist, aw_dict, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(aw_arr, aw_dict, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(aw_adata, aw_dict, rtol=1e-12, atol=1e-12)


def test_duplicate_correlation_equivalence_across_inputs():
    from pylimma import voom
    from pylimma.dups import duplicate_correlation

    counts, design = _test_counts()
    v = voom(counts, design)
    block = np.array([1, 1, 2, 2, 3, 3, 4, 4])

    dc_arr = duplicate_correlation(v["E"], design=design, block=block)
    dc_elist = duplicate_correlation(EList({"E": v["E"], "design": design}), block=block)
    np.testing.assert_allclose(
        dc_elist["consensus_correlation"],
        dc_arr["consensus_correlation"],
        rtol=1e-12,
        atol=1e-12,
    )


# -----------------------------------------------------------------------------
# Part 3: dispatcher error + write-back contracts
# -----------------------------------------------------------------------------


def test_get_eawp_rejects_unsupported_wrapper_by_name():
    """A class named like a Bioconductor wrapper should raise with a
    pointer to the scope policy, not fall through to the ndarray branch."""

    class RGList:
        pass

    with pytest.raises(TypeError, match="out of scope"):
        get_eawp(RGList())


def test_get_eawp_rejects_none():
    with pytest.raises(TypeError):
        get_eawp(None)


def test_get_eawp_anndata_transposes_and_extracts_metadata():
    """AnnData input has samples as rows (obs) and genes as columns (var).
    get_eawp must transpose to (genes, samples) and lift obs/var into
    targets/probes."""
    pytest.importorskip("anndata")
    from anndata import AnnData

    X = np.random.default_rng(0).standard_normal((6, 10))  # 6 samples x 10 genes
    obs = pd.DataFrame(
        {"group": ["A", "A", "A", "B", "B", "B"]}, index=[f"sample{i}" for i in range(6)]
    )
    var = pd.DataFrame(
        {"symbol": [f"g{i}" for i in range(10)]}, index=[f"gene{i}" for i in range(10)]
    )
    adata = AnnData(X=X, obs=obs, var=var)

    out = get_eawp(adata)
    assert out["exprs"].shape == (10, 6)
    np.testing.assert_allclose(out["exprs"], X.T)
    assert out["targets"] is not None and list(out["targets"]["group"]) == list(obs["group"])
    assert out["probes"] is not None and list(out["probes"]["symbol"]) == list(var["symbol"])


def test_get_eawp_dataframe_with_id_column_branch():
    """DataFrame input with a single non-numeric column should treat that
    column as gene IDs - the branch at classes.py:_parse_design-adjacent."""
    df = pd.DataFrame(
        {
            "id": [f"g{i}" for i in range(5)],
            "s1": np.arange(5, dtype=float),
            "s2": np.arange(5, dtype=float) + 1,
            "s3": np.arange(5, dtype=float) + 2,
        }
    )
    out = get_eawp(df)
    assert out["exprs"].shape == (5, 3)
    assert out["probes"] is not None
    assert "id" in out["probes"].columns


def test_get_eawp_dataframe_preserves_gene_name_index():
    """An all-numeric DataFrame with a non-default row index should
    propagate the row names into probes, matching R's getEAWP which
    turns rownames into a one-column data.frame. Without this, gene
    names are silently dropped and top_table returns integer-indexed
    rows."""
    rng = np.random.default_rng(0)
    M = rng.standard_normal((4, 3))
    df = pd.DataFrame(M, index=["TP53", "MYC", "BRCA1", "KRAS"])
    out = get_eawp(df)
    assert out["probes"] is not None
    assert out["probes"].iloc[:, 0].tolist() == ["TP53", "MYC", "BRCA1", "KRAS"]


def test_get_eawp_dataframe_default_index_no_spurious_probes():
    """A DataFrame with the default RangeIndex must NOT generate
    spurious 0/1/2 probes - the index is meaningless."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((4, 3)))
    out = get_eawp(df)
    assert out["probes"] is None


def test_get_eawp_rejects_layer_for_non_anndata():
    """layer= is AnnData-only. Passing it with ndarray / DataFrame /
    EList / dict inputs silently did nothing before - now raises a
    clear TypeError so users catch misuse early."""
    rng = np.random.default_rng(0)
    M = rng.standard_normal((4, 3))

    with pytest.raises(TypeError, match="layer="):
        get_eawp(M, layer="voom_E")
    with pytest.raises(TypeError, match="layer="):
        get_eawp(pd.DataFrame(M), layer="voom_E")
    with pytest.raises(TypeError, match="layer="):
        get_eawp({"E": M}, layer="voom_E")


def test_get_eawp_dict_with_exprs_key_back_compat():
    """A dict written by a previous get_eawp call (which uses 'exprs'
    rather than 'E') must round-trip cleanly."""
    E = np.random.default_rng(0).standard_normal((4, 3))
    d = {"exprs": E, "weights": np.ones_like(E)}
    out = get_eawp(d)
    np.testing.assert_array_equal(out["exprs"], E)
    np.testing.assert_array_equal(out["weights"], np.ones_like(E))


def test_put_eawp_requires_E_key():
    """Missing 'E' should fail loudly rather than silently returning
    something useless."""
    with pytest.raises(ValueError, match="'E'"):
        put_eawp({"weights": np.ones((5, 3))}, np.zeros((5, 3)))


def test_put_eawp_elist_preserves_unrelated_slots():
    """Slots the caller didn't update must survive the put_eawp round-trip."""
    el = EList(
        {
            "E": np.zeros((5, 3)),
            "design": np.eye(3),
            "genes": pd.DataFrame({"symbol": list("abcde")}),
            "custom_slot": "keep_me",
        }
    )
    out = put_eawp({"E": np.ones((5, 3))}, el)
    assert isinstance(out, EList)
    np.testing.assert_array_equal(out.E, np.ones((5, 3)))
    # All non-updated slots preserved
    np.testing.assert_array_equal(np.asarray(out.design), np.eye(3))
    assert "genes" in out
    assert out["custom_slot"] == "keep_me"


def test_put_eawp_anndata_uns_only_payload_when_weights_layered():
    """When weights_layer is given, weights go to layers and uns holds
    only the non-matrix metadata - not a duplicate of weights."""
    pytest.importorskip("anndata")
    from anndata import AnnData

    adata = AnnData(X=np.random.default_rng(0).standard_normal((4, 3)))
    slots = {
        "E": np.ones((3, 4)),
        "weights": np.full((3, 4), 2.0),
        "design": np.eye(4),
        "lib_size": np.array([1, 2, 3, 4]),
    }
    put_eawp(slots, adata, out_layer="E", weights_layer="W", uns_key="meta")

    assert "E" in adata.layers and "W" in adata.layers
    assert "meta" in adata.uns
    assert "weights" not in adata.uns["meta"]
    assert "E" not in adata.uns["meta"]
    assert "design" in adata.uns["meta"]
    assert "lib_size" in adata.uns["meta"]


# -----------------------------------------------------------------------------
# Part 5: AnnData regressions (2026-09-30 audit)
#
# Each test compares the AnnData route with the equivalent ndarray / EList
# route, which the R-parity suite already validates.
# -----------------------------------------------------------------------------


def _de_adata():
    """Log-expression AnnData with a strong group effect in the first 15
    genes, gene symbols in var and a two-level group in obs."""
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(7)
    n_genes, n_samples = 60, 8
    group = np.repeat(["A", "B"], 4)
    expr = rng.normal(8, 1, (n_genes, n_samples))
    expr[:15, group == "B"] += 3
    obs = pd.DataFrame({"group": pd.Categorical(group)}, index=[f"s{i}" for i in range(n_samples)])
    var = pd.DataFrame(
        {"symbol": [f"SYM{i}" for i in range(n_genes)]}, index=[f"g{i}" for i in range(n_genes)]
    )
    return ad.AnnData(X=expr.T.copy(), obs=obs, var=var)


def _h5ad_roundtrip(adata, tmp_path):
    import anndata as ad

    path = tmp_path / "adata.h5ad"
    adata.write_h5ad(path)
    return ad.read_h5ad(path)


def _gene_pathway(ids):
    return pd.DataFrame({"gene": list(ids[:20]) + list(ids[30:40]), "term": ["T1"] * 20 + ["T2"] * 10})


@pytest.mark.parametrize("fn_name", ["goana", "kegga"])
def test_enrichment_anndata_fit_matches_named_fit(fn_name, tmp_path):
    """goana / kegga on an AnnData fit (default key, and after h5ad) must
    match the same fit supplied as an MArrayLM with row names."""
    import pylimma

    fn = getattr(pylimma, fn_name)
    adata = _de_adata()
    pylimma.lm_fit(adata, "~ group")
    pylimma.e_bayes(adata)
    reference = fn(MArrayLM(adata.uns["pylimma"]), gene_pathway=_gene_pathway(adata.var_names))
    assert (reference["up"] > 0).any()

    pd.testing.assert_frame_equal(fn(adata, gene_pathway=_gene_pathway(adata.var_names)), reference)
    reloaded = _h5ad_roundtrip(adata, tmp_path)
    pd.testing.assert_frame_equal(fn(reloaded, gene_pathway=_gene_pathway(adata.var_names)), reference)


def test_goana_anndata_geneid_column_reads_adata_var():
    import pylimma

    adata = _de_adata()
    pylimma.lm_fit(adata, "~ group")
    pylimma.e_bayes(adata)
    symbols = adata.var["symbol"].to_numpy()
    gp = _gene_pathway(symbols)
    pd.testing.assert_frame_equal(
        pylimma.goana(adata, gene_pathway=gp, geneid="symbol"),
        pylimma.goana(adata, gene_pathway=gp, geneid=symbols),
    )


def test_goana_fit_without_row_names_raises_like_r():
    """R: geneid = rownames(de) is character(0) when absent, so goana
    stops with 'geneid of incorrect length' rather than inventing ids."""
    import pylimma

    adata = _de_adata()
    fit = pylimma.e_bayes(pylimma.lm_fit(adata.X.T, np.column_stack([np.ones(8), np.repeat([0.0, 1.0], 4)])))
    with pytest.raises(ValueError, match="geneid of incorrect length"):
        pylimma.goana(fit, gene_pathway=_gene_pathway(adata.var_names))


def _voomed_adata_and_elist():
    """AnnData after voom(design=formula) plus the equivalent EList built
    from voom on the bare count matrix."""
    ad = pytest.importorskip("anndata")
    from pylimma import voom

    counts, design = _test_counts()
    obs = pd.DataFrame({"group": pd.Categorical(np.repeat(["A", "B"], 4))}, index=[f"s{i}" for i in range(8)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(counts.shape[0])])
    adata = ad.AnnData(X=counts.T.copy(), obs=obs, var=var)
    voom(adata, design="~ group")
    v = voom(counts, design)
    elist = EList({"E": v["E"], "weights": v["weights"], "design": design, "genes": var})
    return adata, elist, design


@pytest.mark.parametrize(
    "fn_name, kwargs",
    [
        ("camera", {}),
        ("fry", {}),
        ("roast", {"rng": 11, "nrot": 199}),
        ("mroast", {"rng": 11, "nrot": 199}),
        ("romer", {"rng": 11, "nrot": 199}),
    ],
)
def test_geneset_layer_matches_voom_elist(fn_name, kwargs):
    """layer='voom_E' must feed the voom expression, weights and design,
    exactly as an EList from voom does."""
    import pylimma

    fn = getattr(pylimma, fn_name)
    adata, elist, design = _voomed_adata_and_elist()
    index = {"set1": np.arange(10), "set2": np.arange(40, 60)}
    if fn_name == "roast":
        index = np.arange(10)
    # R's romer does as.matrix(y), so it never picks up y$design.
    design_kwargs = {"design": design} if fn_name == "romer" else {}
    from_adata = fn(adata, index, layer="voom_E", **design_kwargs, **kwargs)
    from_elist = fn(elist, index, design=design, **kwargs)
    if isinstance(from_elist, pd.DataFrame):
        pd.testing.assert_frame_equal(from_adata, from_elist)
    else:
        pd.testing.assert_frame_equal(from_adata["p_value"], from_elist["p_value"])


def test_camera_weights_layer_is_used():
    import pylimma

    adata, elist, design = _voomed_adata_and_elist()
    adata.layers["custom_w"] = adata.layers["voom_weights"]
    del adata.layers["voom_weights"]
    index = {"set1": np.arange(10)}
    pd.testing.assert_frame_equal(
        pylimma.camera(adata, index, design, layer="voom_E", weights_layer="custom_w"),
        pylimma.camera(elist, index, design),
    )


def test_wsva_layer_matches_matrix():
    import pylimma

    adata, elist, design = _voomed_adata_and_elist()
    np.testing.assert_array_equal(
        pylimma.wsva(adata, design, n_sv=2, layer="voom_E"),
        pylimma.wsva(elist["E"], design, n_sv=2),
    )


@pytest.mark.parametrize("fn_name", ["camera", "wsva"])
def test_geneset_layer_rejected_for_non_anndata(fn_name):
    import pylimma

    _, elist, design = _voomed_adata_and_elist()
    args = (elist["E"], {"set1": np.arange(10)}, design) if fn_name == "camera" else (elist["E"], design)
    with pytest.raises(TypeError, match="only supported for AnnData"):
        getattr(pylimma, fn_name)(*args, layer="voom_E")


@pytest.mark.parametrize("voom_fn", ["voom", "voom_with_quality_weights", "vooma"])
def test_voom_formula_design_names_reach_lm_fit(voom_fn, tmp_path):
    """voom(design=formula) -> lm_fit(layer=...) must keep the patsy column
    names (R keeps colnames(v$design)), including across an h5ad save."""
    import pylimma

    adata = _voomed_adata_and_elist()[0]
    if voom_fn == "vooma":
        adata.X = adata.layers["voom_E"].copy()
    getattr(pylimma, voom_fn)(adata, design="~ group")
    layer = "vooma_E" if voom_fn == "vooma" else "voom_E"
    for obj in (adata, _h5ad_roundtrip(adata, tmp_path)):
        pylimma.lm_fit(obj, layer=layer)
        assert obj.uns["pylimma"]["coef_names"] == ["Intercept", "group[T.B]"]
        pylimma.e_bayes(obj)
        pd.testing.assert_frame_equal(
            pylimma.top_table(obj, coef="group[T.B]", number=np.inf, sort_by="none"),
            pylimma.top_table(obj, coef=1, number=np.inf, sort_by="none"),
        )


def test_vooma_lm_fit_keeps_gene_and_coef_names():
    """vooma_lm_fit must label genes and coefficients as lm_fit does
    (R's final lmFit runs on the EList)."""
    import pylimma

    adata = _de_adata()
    pylimma.vooma_lm_fit(adata, design="~ group")
    fit = adata.uns["pylimma"]
    assert fit["genes"] == list(adata.var_names)
    assert fit["coef_names"] == ["Intercept", "group[T.B]"]

    matrix_fit = pylimma.vooma_lm_fit(adata.X.T, design=np.asarray(fit["design"]))
    np.testing.assert_array_equal(fit["coefficients"], matrix_fit["coefficients"])

    frame = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)
    assert pylimma.vooma_lm_fit(frame, design=np.asarray(fit["design"]))["genes"] == list(adata.var_names)


def test_top_table_f_accepts_anndata():
    import pylimma

    adata = _de_adata()
    adata.obs["batch"] = list("xyxyxyxy")
    pylimma.lm_fit(adata, "~ group + batch")
    pylimma.e_bayes(adata)
    with pytest.warns(DeprecationWarning):
        from_adata = pylimma.top_table_f(adata, number=10)
    with pytest.warns(DeprecationWarning):
        from_fit = pylimma.top_table_f(adata.uns["pylimma"], number=10)
    pd.testing.assert_frame_equal(from_adata, from_fit)


def test_lm_fit_anndata_does_not_duplicate_obs_as_targets():
    """For AnnData the fit sits next to adata.obs, so lm_fit must not store
    a second copy as fit['targets'] (it would be duplicated in memory and
    written again to the h5ad file)."""
    import pylimma

    adata = _de_adata()
    pylimma.lm_fit(adata, "~ group")
    assert "targets" not in adata.uns["pylimma"]


def test_get_fit_matches_matrix_fit_before_and_after_h5ad(tmp_path):
    """get_fit must return the same fit as the matrix route, with names as
    lists and no targets slot, also after an h5ad reload."""
    import pylimma

    adata = _de_adata()
    pylimma.lm_fit(adata, "~ group")
    pylimma.e_bayes(adata)
    frame = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)
    reference = pylimma.e_bayes(pylimma.lm_fit(frame, adata.uns["pylimma"]["design"]))
    for obj in (adata, _h5ad_roundtrip(adata, tmp_path)):
        fit = pylimma.get_fit(obj)
        assert isinstance(fit, MArrayLM)
        assert fit["coef_names"] == ["Intercept", "group[T.B]"]
        # adata.obs may have changed since the fit, so it is not attached.
        assert "targets" not in fit
        pd.testing.assert_frame_equal(
            pylimma.top_table(fit, coef="group[T.B]", number=np.inf, sort_by="none"),
            pylimma.top_table(reference, coef=1, number=np.inf, sort_by="none"),
        )


def test_get_fit_returns_independent_copy():
    import pylimma

    adata = _de_adata()
    pylimma.lm_fit(adata, "~ group")
    fit = pylimma.get_fit(adata)
    fit["coefficients"][0, 0] = np.nan
    assert not np.isnan(adata.uns["pylimma"]["coefficients"][0, 0])


def test_get_fit_rejects_non_anndata():
    import pylimma

    with pytest.raises(TypeError, match="expects an AnnData"):
        pylimma.get_fit({"coefficients": np.zeros((2, 1))})


@pytest.mark.parametrize("wrap", [MArrayLM, dict])
def test_fit_taken_from_reloaded_uns_by_hand_supports_name_lookup(wrap, tmp_path):
    """Fits taken out of a reloaded adata.uns by hand (h5ad stores the name
    slots as arrays) must still accept coefficient names, without the
    caller's object being modified."""
    import pylimma

    adata = _de_adata()
    pylimma.lm_fit(adata, "~ group")
    pylimma.e_bayes(adata)
    expected = pylimma.top_table(adata, coef="group[T.B]", number=np.inf, sort_by="none")
    fit = wrap(_h5ad_roundtrip(adata, tmp_path).uns["pylimma"])
    pd.testing.assert_frame_equal(
        pylimma.top_table(fit, coef="group[T.B]", number=np.inf, sort_by="none"), expected
    )
    assert isinstance(fit["coef_names"], np.ndarray)


@pytest.mark.parametrize("change", ["subset", "reorder", "rename"])
def test_goana_geneid_column_refuses_var_changed_after_fit(change):
    """adata.uns is copied unchanged when genes are subset / reordered, so
    a string geneid must not read an adata.var that no longer describes
    the fitted genes; a vector geneid aligned to the fit still works."""
    import pylimma

    adata = _de_adata()
    pylimma.lm_fit(adata, "~ group")
    pylimma.e_bayes(adata)
    symbols = adata.var["symbol"].to_numpy()
    gp = _gene_pathway(symbols)
    expected = pylimma.goana(adata, gene_pathway=gp, geneid="symbol")

    if change == "subset":
        changed = adata[:, 5:].copy()
    elif change == "reorder":
        changed = adata[:, ::-1].copy()
    else:
        changed = adata.copy()
        changed.var_names = [f"renamed{i}" for i in range(changed.n_vars)]
    with pytest.raises(ValueError, match="does not match the genes in the stored fit"):
        pylimma.goana(changed, gene_pathway=gp, geneid="symbol")
    pd.testing.assert_frame_equal(pylimma.goana(changed, gene_pathway=gp, geneid=symbols), expected)


@pytest.mark.parametrize(
    "call",
    [
        lambda pl, a: pl.vooma_by_group(a, group=a.obs["group"].to_numpy(), design="~ group"),
        lambda pl, a: pl.remove_batch_effect(a, batch=np.tile(["x", "y"], 4), design="~ group"),
        lambda pl, a: pl.wsva(a, "~ group"),
        lambda pl, a: pl.camera(a, {"s": np.arange(10)}, "~ group"),
        lambda pl, a: pl.roast(a, np.arange(10), "~ group", nrot=9),
        lambda pl, a: pl.mroast(a, {"s": np.arange(10)}, "~ group", nrot=9),
        lambda pl, a: pl.fry(a, {"s": np.arange(10)}, "~ group"),
        lambda pl, a: pl.romer(a, {"s": np.arange(10)}, "~ group", nrot=9),
        lambda pl, a: pl.inter_gene_correlation(a.X.T, "~ group"),
    ],
    ids=["vooma_by_group", "remove_batch_effect", "wsva", "camera", "roast", "mroast", "fry", "romer", "inter_gene_correlation"],
)
def test_matrix_only_functions_reject_formula_with_clear_error(call):
    """Formula strings are limited to lm_fit and the voom family; the
    matrix-only functions must say so rather than fail inside numpy."""
    import pylimma

    with pytest.raises(ValueError, match="formula strings are only supported by lm_fit"):
        call(pylimma, _de_adata())


def _transposed_arg_adata():
    """AnnData (8 samples x 60 genes) whose X is usable as counts, plus a
    positive matrix in AnnData orientation (samples x genes)."""
    adata = _de_adata()
    adata.X = np.round(2 ** adata.X)
    samples_by_genes = np.random.default_rng(9).uniform(0.5, 2.0, adata.shape)
    return adata, samples_by_genes


@pytest.mark.parametrize(
    "name, call",
    [
        ("weights", lambda pl, a, m: pl.lm_fit(a, "~ group", weights=m)),
        ("offset", lambda pl, a, m: pl.voom(a, design="~ group", offset=m)),
        ("offset_prior", lambda pl, a, m: pl.voom(a, design="~ group", offset_prior=m)),
        ("predictor", lambda pl, a, m: pl.vooma(a, design="~ group", predictor=m)),
        ("predictor", lambda pl, a, m: pl.vooma_lm_fit(a, design="~ group", predictor=m)),
        ("weights", lambda pl, a, m: pl.vooma_lm_fit(a, design="~ group", prior_weights=m)),
        ("background", lambda pl, a, m: pl.background_correct(a, background=m, method="subtract")),
        ("weights", lambda pl, a, m: pl.array_weights(a, design=np.ones((8, 1)), weights=m)),
        ("weights", lambda pl, a, m: pl.duplicate_correlation(a, np.ones((8, 1)), ndups=1, block=np.repeat([0, 1, 2, 3], 2), weights=m)),
        ("weights", lambda pl, a, m: pl.camera(a, {"s": np.arange(10)}, np.ones((8, 1)), weights=m)),
    ],
    ids=["lm_fit", "voom-offset", "voom-offset_prior", "vooma", "vooma_lm_fit-predictor",
         "vooma_lm_fit-prior_weights", "background_correct", "array_weights", "duplicate_correlation", "camera"],
)
def test_matrix_argument_in_anndata_orientation_gets_transpose_hint(name, call):
    """Explicit matrix arguments are genes x samples (limma orientation);
    one supplied in AnnData orientation must get an error that says so."""
    import pylimma

    adata, samples_by_genes = _transposed_arg_adata()
    with pytest.raises(ValueError, match=rf"{name} has shape \(8, 60\) .* looks transposed"):
        call(pylimma, adata, samples_by_genes)


def test_matrix_argument_wrong_shape_without_transpose_has_no_hint():
    import pylimma

    adata, _ = _transposed_arg_adata()
    with pytest.raises(ValueError, match="weights is of unexpected shape$"):
        pylimma.lm_fit(adata, "~ group", weights=np.ones((59, 8)))


def _rich_adata():
    """AnnData with every slot populated: two layers (one sparse), obsm /
    varm / obsp / varp and a stored fit in uns."""
    import scipy.sparse as sp

    import pylimma

    adata = _de_adata()
    rng = np.random.default_rng(11)
    adata.layers["counts"] = sp.csr_matrix(np.round(2 ** adata.X))
    adata.layers["w"] = rng.uniform(0.5, 2.0, adata.shape)
    adata.obsm["X_pca"] = rng.normal(size=(adata.n_obs, 2))
    adata.varm["loadings"] = rng.normal(size=(adata.n_vars, 2))
    adata.obsp["dist"] = rng.uniform(size=(adata.n_obs, adata.n_obs))
    adata.varp["cor"] = rng.uniform(size=(adata.n_vars, adata.n_vars))
    pylimma.lm_fit(adata, "~ group")
    return adata


def _dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _assert_same_adata(actual, expected):
    np.testing.assert_array_equal(actual.X, expected.X)
    assert list(actual.layers) == list(expected.layers)
    for name in expected.layers:
        np.testing.assert_array_equal(_dense(actual.layers[name]), _dense(expected.layers[name]))
    pd.testing.assert_frame_equal(actual.obs, expected.obs)
    pd.testing.assert_frame_equal(actual.var, expected.var)


def test_avereps_anndata_mirrors_elist_method():
    """R's avereps.EList averages every matrix, keeps the first occurrence
    of each ID's annotation and leaves the sample axis and other slots
    alone; the input must not be modified."""
    import pylimma

    adata = _rich_adata()
    before = adata.copy()
    ID = np.array(["z", "a", "z", "m"] * (adata.n_vars // 4))
    out = pylimma.avereps(adata, ID=ID)

    _assert_same_adata(adata, before)
    first = np.sort(np.unique(ID, return_index=True)[1])
    assert list(out.var_names) == ["z", "a", "m"]
    np.testing.assert_array_equal(out.X.T, pylimma.avereps(adata.X.T, ID=ID))
    np.testing.assert_array_equal(out.layers["counts"].T, pylimma.avereps(adata.layers["counts"].toarray().T, ID=ID))
    np.testing.assert_array_equal(out.layers["w"].T, pylimma.avereps(adata.layers["w"].T, ID=ID))
    assert list(out.var["symbol"]) == list(adata.var["symbol"].iloc[first])
    np.testing.assert_array_equal(out.varm["loadings"], adata.varm["loadings"][first])
    np.testing.assert_array_equal(out.varp["cor"], adata.varp["cor"][np.ix_(first, first)])
    pd.testing.assert_frame_equal(out.obs, adata.obs)
    np.testing.assert_array_equal(out.obsm["X_pca"], adata.obsm["X_pca"])
    np.testing.assert_array_equal(_dense(out.obsp["dist"]), adata.obsp["dist"])
    # uns carried as is (R's y <- x): the stored fit still describes the
    # original probes.
    assert out.uns["pylimma"]["genes"] == list(adata.var_names)
    np.testing.assert_array_equal(out.uns["pylimma"]["coefficients"], adata.uns["pylimma"]["coefficients"])


@pytest.mark.parametrize("weighted", [False, True])
def test_aver_arrays_anndata_mirrors_elist_method(weighted):
    """R's avearrays.EList averages every matrix with the same weights,
    keeps the first occurrence of each id's sample annotation and leaves
    the gene axis and other slots alone; the input must not be modified."""
    import pylimma

    adata = _rich_adata()
    before = adata.copy()
    ids = ["z", "a", "z", "a", "m", "m", "b", "b"]
    weights = adata.layers["w"].T if weighted else None
    out = pylimma.aver_arrays(adata, id=ids, weights=weights)

    _assert_same_adata(adata, before)
    first = np.array([0, 1, 4, 6])
    assert list(out.obs_names) == ["z", "a", "m", "b"]
    for name, matrix in (("X", adata.X), ("counts", adata.layers["counts"].toarray()), ("w", adata.layers["w"])):
        averaged = out.X if name == "X" else out.layers[name]
        np.testing.assert_array_equal(averaged.T, pylimma.aver_arrays(matrix.T, id=ids, weights=weights))
    assert list(out.obs["group"]) == list(adata.obs["group"].iloc[first])
    np.testing.assert_array_equal(out.obsm["X_pca"], adata.obsm["X_pca"][first])
    pd.testing.assert_frame_equal(out.var, adata.var)
    np.testing.assert_array_equal(out.varm["loadings"], adata.varm["loadings"])
    assert out.uns["pylimma"]["genes"] == list(adata.var_names)

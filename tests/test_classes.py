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
    return EList({
        "E": pd.DataFrame(E, index=gene_names, columns=sample_names),
        "weights": pd.DataFrame(W, index=gene_names, columns=sample_names),
        "genes": genes,
        "targets": targets,
        "design": pd.DataFrame(design, index=sample_names,
                               columns=["Intercept", "groupB"]),
    })


ELIST_SUBSET_CASES = [
    ("full", slice(None), slice(None)),
    ("rows", slice(0, 10), slice(None)),
    ("cols", slice(None), slice(0, 4)),
    ("both", slice(0, 10), slice(0, 4)),
    ("rowstr", ["gene3", "gene7", "gene15"], slice(None)),
    ("rowbool",
     np.array([False, True, False, True, False, True, False, True,
               False, True] + [False] * 20), slice(None)),
]


@pytest.mark.parametrize("tag,i,j", ELIST_SUBSET_CASES)
def test_elist_subset_parity_numerical(tag, i, j):
    """Slot-by-slot numeric parity of EList[i, j] against R's [.EList."""
    el = _build_elist_from_fixtures()
    sub = el if tag == "full" else el[i, j]

    np.testing.assert_allclose(
        np.asarray(sub.E), _load_matrix(FIXTURES / f"R_elist_{tag}_E.csv"),
        rtol=1e-6, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(sub.weights),
        _load_matrix(FIXTURES / f"R_elist_{tag}_weights.csv"),
        rtol=1e-6, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(sub.design),
        _load_matrix(FIXTURES / f"R_elist_{tag}_design.csv"),
        rtol=1e-6, atol=1e-12,
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
    return MArrayLM({
        "coefficients": pd.DataFrame(
            coef, index=gene_names, columns=["Intercept", "groupB"]),
        "stdev_unscaled": pd.DataFrame(
            stdev, index=gene_names, columns=["Intercept", "groupB"]),
        "t": pd.DataFrame(tstat, index=gene_names,
                          columns=["Intercept", "groupB"]),
        "p_value": pd.DataFrame(pval, index=gene_names,
                                columns=["Intercept", "groupB"]),
        "lods": pd.DataFrame(lods, index=gene_names,
                             columns=["Intercept", "groupB"]),
        "Amean": i_slots["Amean"].values,
        "sigma": i_slots["sigma"].values,
        "df_residual": i_slots["df_residual"].values,
        "df_total": i_slots["df_total"].values,
        "s2_post": i_slots["s2_post"].values,
        "genes": genes,
    })


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
        rtol=1e-6, atol=1e-12,
    )
    exp_i = _load_df(FIXTURES / f"R_marraylm_{tag}_i_slots.csv")
    np.testing.assert_allclose(np.asarray(sub.Amean),
                                exp_i["Amean"].values, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(np.asarray(sub.sigma),
                                exp_i["sigma"].values, rtol=1e-6, atol=1e-12)


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
    np.testing.assert_array_equal(
        np.asarray(h.E), np.asarray(el.E)[:5]
    )


# -----------------------------------------------------------------------------
# Part 2: polymorphic-input equivalence for Phase 4 functions
# -----------------------------------------------------------------------------
# If get_eawp/put_eawp have a bug (missing transpose, lost weights,
# clobbered design), the numerics diverge. These are the load-bearing
# tests for the dispatchers.

def _test_counts():
    rng = np.random.default_rng(42)
    counts = rng.poisson(20, (100, 8)).astype(float)
    design = np.column_stack([np.ones(8), np.array([0]*4 + [1]*4)])
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
    np.testing.assert_allclose(v_el.weights, v_arr["weights"],
                                rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(adata.layers["voom_E"].T, v_arr["E"],
                                rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(adata.layers["voom_weights"].T,
                                v_arr["weights"], rtol=1e-12, atol=1e-12)


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
        np.testing.assert_allclose(sorted_cols[:, j], sorted_cols[:, 0],
                                    rtol=1e-12, atol=1e-12)


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
    np.testing.assert_allclose(adata.layers["normalized"].T, out_arr,
                                rtol=1e-12, atol=1e-12)


def test_array_weights_equivalence_across_inputs():
    """array_weights must yield identical sample-level weights for
    equivalent ndarray / EList / dict / AnnData input."""
    pytest.importorskip("anndata")
    from anndata import AnnData
    from pylimma import array_weights, voom

    counts, design = _test_counts()
    v = voom(counts, design)  # dict

    aw_dict = array_weights(v)
    aw_elist = array_weights(
        EList({"E": v["E"], "weights": v["weights"], "design": design})
    )
    aw_arr = array_weights(v["E"], design=design, weights=v["weights"])

    adata = AnnData(X=v["E"].T)
    adata.layers["weights"] = v["weights"].T
    aw_adata = array_weights(adata, design=design, layer=None,
                              weights=v["weights"])

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
    dc_elist = duplicate_correlation(
        EList({"E": v["E"], "design": design}), block=block
    )
    np.testing.assert_allclose(
        dc_elist["consensus_correlation"],
        dc_arr["consensus_correlation"],
        rtol=1e-12, atol=1e-12,
    )


# -----------------------------------------------------------------------------
# Part 3: dispatcher error + write-back contracts
# -----------------------------------------------------------------------------

def test_get_eawp_rejects_unsupported_wrapper_by_name():
    """A class named like a Bioconductor wrapper should raise with a
    pointer to the scope policy, not fall through to the ndarray branch."""
    class RGList:
        pass
    with pytest.raises(TypeError, match="policy_data_class_wrappers"):
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
    obs = pd.DataFrame({"group": ["A", "A", "A", "B", "B", "B"]},
                       index=[f"sample{i}" for i in range(6)])
    var = pd.DataFrame({"symbol": [f"g{i}" for i in range(10)]},
                       index=[f"gene{i}" for i in range(10)])
    adata = AnnData(X=X, obs=obs, var=var)

    out = get_eawp(adata)
    assert out["exprs"].shape == (10, 6)
    np.testing.assert_allclose(out["exprs"], X.T)
    assert out["targets"] is not None and list(out["targets"]["group"]) == \
        list(obs["group"])
    assert out["probes"] is not None and list(out["probes"]["symbol"]) == \
        list(var["symbol"])


def test_get_eawp_dataframe_with_id_column_branch():
    """DataFrame input with a single non-numeric column should treat that
    column as gene IDs - the branch at classes.py:_parse_design-adjacent."""
    df = pd.DataFrame({
        "id": [f"g{i}" for i in range(5)],
        "s1": np.arange(5, dtype=float),
        "s2": np.arange(5, dtype=float) + 1,
        "s3": np.arange(5, dtype=float) + 2,
    })
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
    el = EList({
        "E": np.zeros((5, 3)),
        "design": np.eye(3),
        "genes": pd.DataFrame({"symbol": list("abcde")}),
        "custom_slot": "keep_me",
    })
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
    slots = {"E": np.ones((3, 4)),
             "weights": np.full((3, 4), 2.0),
             "design": np.eye(4),
             "lib_size": np.array([1, 2, 3, 4])}
    put_eawp(slots, adata, out_layer="E", weights_layer="W", uns_key="meta")

    assert "E" in adata.layers and "W" in adata.layers
    assert "meta" in adata.uns
    assert "weights" not in adata.uns["meta"]
    assert "E" not in adata.uns["meta"]
    assert "design" in adata.uns["meta"]
    assert "lib_size" in adata.uns["meta"]

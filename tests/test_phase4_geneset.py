"""
R parity tests for Phase 4 (gene-set testing and statistical utilities).

Fixtures are produced by the Phase-4 section of
``tests/fixtures/generate_all_fixtures.R``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylimma.geneset import (
    camera,
    camera_pr,
    fry,
    gene_set_test,
    ids2indices,
    inter_gene_correlation,
    mroast,
    rank_sum_test_with_correlation,
    roast,
    romer,
)
from pylimma.utils import (
    convest,
    detection_p_values,
    prop_true_null,
    tricube_moving_average,
    zscore_t,
)

from .helpers import compare_pvalues

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / name, **kwargs)


@pytest.fixture(scope="module")
def phase4_data():
    y = _load("R_phase4_y.csv", index_col=0).values
    design = _load("R_phase4_design.csv").values
    gs_long = _load("R_phase4_gene_sets.csv")
    gene_sets = {name: grp["index"].values - 1 for name, grp in gs_long.groupby("set", sort=False)}
    return {"y": y, "design": design, "gene_sets": gene_sets}


# ---------------------------------------------------------------------------
# zscore_t
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["bailey", "hill", "wallace"])
def test_zscore_t_approximations(method):
    r = _load(f"R_zscoret_{method}.csv")
    z = zscore_t(r["t"].values, r["df"].values, approx=True, method=method)
    np.testing.assert_allclose(z, r["z"].values, rtol=1e-8, atol=1e-12)


def test_zscore_t_exact_quantile():
    r = _load("R_zscoret_quantile.csv")
    z = zscore_t(r["t"].values, r["df"].values, approx=False)
    np.testing.assert_allclose(z, r["z"].values, rtol=1e-8, atol=1e-12)


def test_zscore_t_rejects_unknown_method():
    with pytest.raises(ValueError):
        zscore_t(np.array([1.0]), 10.0, approx=True, method="quantile")


# ---------------------------------------------------------------------------
# tricube_moving_average
# ---------------------------------------------------------------------------


def test_tricube_moving_average_default_span():
    r = _load("R_tricube_moving_average.csv")
    y = tricube_moving_average(r["x"].values)
    np.testing.assert_allclose(y, r["y_default"].values, rtol=1e-8, atol=1e-12)


def test_tricube_moving_average_wide_span():
    r = _load("R_tricube_moving_average.csv")
    y = tricube_moving_average(r["x"].values, span=0.8)
    np.testing.assert_allclose(y, r["y_wide"].values, rtol=1e-8, atol=1e-12)


# ---------------------------------------------------------------------------
# ids2indices
# ---------------------------------------------------------------------------


def test_ids2indices_matches_r():
    r = _load("R_ids2indices_basic.csv")
    identifiers = [f"g{i}" for i in range(1, 401)]
    gs = {
        "setA": [f"g{i}" for i in range(1, 21)],
        "setB": [f"g{i}" for i in range(15, 41)],
    }
    out = ids2indices(gs, identifiers)
    for name, idx in out.items():
        r_idx = np.sort(r[r["set"] == name]["index"].values - 1)
        np.testing.assert_array_equal(np.sort(idx), r_idx)


def test_ids2indices_non_dict_input_wraps_as_set1():
    out = ids2indices(["g1", "g3"], ["g1", "g2", "g3"])
    assert list(out.keys()) == ["Set1"]
    np.testing.assert_array_equal(out["Set1"], np.array([0, 2]))


def test_ids2indices_removes_empty():
    out = ids2indices({"hit": ["g1"], "miss": ["gX"]}, ["g1", "g2"])
    assert "miss" not in out
    assert "hit" in out


# ---------------------------------------------------------------------------
# roast
# ---------------------------------------------------------------------------


def test_roast_single_set_deterministic_match(phase4_data):
    r_pvals = _load("R_roast_pvalues.csv", index_col=0)
    r_ng = _load("R_roast_ngenes.csv")
    py = roast(
        phase4_data["y"],
        index=phase4_data["gene_sets"]["setA"],
        design=phase4_data["design"],
        contrast=1,
        nrot=999,
        rng=4,
    )
    assert int(r_ng["ngenes"].values[0]) == py["ngenes_in_set"]
    np.testing.assert_allclose(
        py["p_value"]["active_prop"].values,
        r_pvals["Active.Prop"].values,
        rtol=1e-6,
        atol=1e-12,
    )


def test_roast_pvalue_log_scale_match(phase4_data):
    r_pvals = _load("R_roast_pvalues.csv", index_col=0)
    py = roast(
        phase4_data["y"],
        index=phase4_data["gene_sets"]["setA"],
        design=phase4_data["design"],
        contrast=1,
        nrot=999,
        rng=4,
    )
    cmp = compare_pvalues(
        r_pvals["P.Value"].values,
        py["p_value"]["p_value"].values,
        max_log10_diff=0.5,
    )
    assert cmp["match"], cmp


# ---------------------------------------------------------------------------
# mroast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("set_statistic", ["mean", "floormean", "mean50", "msq"])
def test_mroast_active_proportions_match(phase4_data, set_statistic):
    r = _load(f"R_mroast_{set_statistic}.csv", index_col=0)
    py = mroast(
        phase4_data["y"],
        index=phase4_data["gene_sets"],
        design=phase4_data["design"],
        contrast=1,
        set_statistic=set_statistic,
        nrot=999,
        rng=4,
    )
    r_sorted = r.sort_index()
    py_sorted = py.sort_index()
    np.testing.assert_array_equal(py_sorted["n_genes"].values, r_sorted["NGenes"].values)
    np.testing.assert_allclose(
        py_sorted["prop_down"].values,
        r_sorted["PropDown"].values,
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        py_sorted["prop_up"].values,
        r_sorted["PropUp"].values,
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_array_equal(py_sorted["direction"].values, r_sorted["Direction"].values)


@pytest.mark.parametrize("set_statistic", ["mean", "floormean", "mean50", "msq"])
def test_mroast_pvalues_log_scale_match(phase4_data, set_statistic):
    r = _load(f"R_mroast_{set_statistic}.csv", index_col=0)
    py = mroast(
        phase4_data["y"],
        index=phase4_data["gene_sets"],
        design=phase4_data["design"],
        contrast=1,
        set_statistic=set_statistic,
        nrot=999,
        rng=4,
    )
    r_sorted = r.sort_index()
    py_sorted = py.sort_index()
    cmp = compare_pvalues(
        r_sorted["PValue"].values,
        py_sorted["p_value"].values,
        max_log10_diff=0.5,
    )
    assert cmp["match"], cmp
    cmp_mixed = compare_pvalues(
        r_sorted["PValue.Mixed"].values,
        py_sorted["p_value_mixed"].values,
        max_log10_diff=0.5,
    )
    assert cmp_mixed["match"], cmp_mixed


# ---------------------------------------------------------------------------
# fry
# ---------------------------------------------------------------------------


def test_fry_single_set_matches_r(phase4_data):
    r = _load("R_fry_single.csv", index_col=0)
    py = fry(
        phase4_data["y"],
        index=phase4_data["gene_sets"]["setA"],
        design=phase4_data["design"],
        contrast=1,
    )
    np.testing.assert_array_equal(py["n_genes"].values, r["NGenes"].values)
    np.testing.assert_array_equal(py["direction"].values, r["Direction"].values)
    np.testing.assert_allclose(py["p_value"].values, r["PValue"].values, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(
        py["p_value_mixed"].values,
        r["PValue.Mixed"].values,
        rtol=1e-6,
        atol=1e-12,
    )


def test_fry_multi_set_matches_r(phase4_data):
    r = _load("R_fry_multi.csv", index_col=0)
    py = fry(
        phase4_data["y"],
        index=phase4_data["gene_sets"],
        design=phase4_data["design"],
        contrast=1,
    )
    r_sorted = r.sort_index()
    py_sorted = py.sort_index()
    np.testing.assert_array_equal(py_sorted["n_genes"].values, r_sorted["NGenes"].values)
    np.testing.assert_array_equal(py_sorted["direction"].values, r_sorted["Direction"].values)
    np.testing.assert_allclose(
        py_sorted["p_value"].values,
        r_sorted["PValue"].values,
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        py_sorted["p_value_mixed"].values,
        r_sorted["PValue.Mixed"].values,
        rtol=1e-6,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# camera / camera_pr / inter_gene_correlation
# ---------------------------------------------------------------------------


def test_camera_default_matches_r(phase4_data):
    r = _load("R_camera_default.csv", index_col=0)
    py = camera(
        phase4_data["y"],
        index=phase4_data["gene_sets"],
        design=phase4_data["design"],
        contrast=1,
    )
    r_sorted = r.sort_index()
    py_sorted = py.sort_index()
    np.testing.assert_array_equal(py_sorted["n_genes"].values, r_sorted["NGenes"].values)
    cmp = compare_pvalues(
        r_sorted["PValue"].values,
        py_sorted["p_value"].values,
        max_log10_diff=0.5,
    )
    assert cmp["match"], cmp


def test_camera_use_ranks_matches_r(phase4_data):
    r = _load("R_camera_ranks.csv", index_col=0)
    py = camera(
        phase4_data["y"],
        index=phase4_data["gene_sets"],
        design=phase4_data["design"],
        contrast=1,
        use_ranks=True,
    )
    r_sorted = r.sort_index()
    py_sorted = py.sort_index()
    cmp = compare_pvalues(
        r_sorted["PValue"].values,
        py_sorted["p_value"].values,
        max_log10_diff=0.5,
    )
    assert cmp["match"], cmp


def test_camera_fixed_cor_matches_r(phase4_data):
    r = _load("R_camera_intergene.csv", index_col=0)
    py = camera(
        phase4_data["y"],
        index=phase4_data["gene_sets"],
        design=phase4_data["design"],
        contrast=1,
        inter_gene_cor=0.05,
    )
    r_sorted = r.sort_index()
    py_sorted = py.sort_index()
    np.testing.assert_allclose(
        py_sorted["p_value"].values,
        r_sorted["PValue"].values,
        rtol=1e-6,
        atol=1e-12,
    )


def test_inter_gene_correlation_returns_vif_and_corr(phase4_data):
    # Sanity check that interGeneCorrelation returns the R shape.
    out = inter_gene_correlation(phase4_data["y"][:30, :], phase4_data["design"])
    assert "vif" in out and "correlation" in out
    assert out["vif"] >= 0.0


def test_camera_pr_matches_r(phase4_data):
    stat_in = _load("R_camera_pr_input.csv", index_col=0)
    r = _load("R_camera_pr.csv", index_col=0)
    py = camera_pr(stat_in["statistic"].values, index=phase4_data["gene_sets"])
    r_sorted = r.sort_index()
    py_sorted = py.sort_index()
    np.testing.assert_array_equal(py_sorted["n_genes"].values, r_sorted["NGenes"].values)
    np.testing.assert_allclose(
        py_sorted["p_value"].values,
        r_sorted["PValue"].values,
        rtol=1e-8,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# romer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("set_statistic", ["mean", "floormean", "mean50"])
def test_romer_pvalues_log_scale_match(phase4_data, set_statistic):
    r = _load(f"R_romer_{set_statistic}.csv", index_col=0)
    py = romer(
        phase4_data["y"],
        index=phase4_data["gene_sets"],
        design=phase4_data["design"],
        contrast=1,
        set_statistic=set_statistic,
        nrot=999,
        rng=4,
    )
    np.testing.assert_array_equal(py["n_genes"].values, r["NGenes"].values)
    for col_py, col_r in [("up", "Up"), ("down", "Down"), ("mixed", "Mixed")]:
        cmp = compare_pvalues(r[col_r].values, py[col_py].values, max_log10_diff=0.5)
        assert cmp["match"], f"{set_statistic}/{col_py}: {cmp}"


# ---------------------------------------------------------------------------
# gene_set_test / rank_sum_test_with_correlation
# ---------------------------------------------------------------------------


def test_rank_sum_test_with_correlation_matches_r():
    r = _load("R_rank_sum_test_with_correlation.csv")
    stat_vec = _load("R_genesettest_input.csv")["statistic"].values
    out = rank_sum_test_with_correlation(np.arange(20), stat_vec, correlation=0.05, df=np.inf)
    np.testing.assert_allclose(out["less"], r["less"][0], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(out["greater"], r["greater"][0], rtol=1e-8, atol=1e-12)


def test_gene_set_test_ranks_only_matches_r():
    stat_vec = _load("R_genesettest_input.csv")["statistic"].values
    r = pd.read_csv(
        FIXTURES / "R_geneSetTest.csv",
        header=None,
        names=["alt", "ranks", "p"],
    )
    for _, row in r.iterrows():
        ranks_only = str(row["ranks"]).upper() == "TRUE"
        if not ranks_only:
            continue  # simulation path tested separately
        py = gene_set_test(
            np.arange(20),
            stat_vec,
            alternative=row["alt"],
            ranks_only=True,
            nsim=999,
            rng=4,
        )
        cmp = compare_pvalues(np.array([row["p"]]), np.array([py]), max_log10_diff=0.5)
        assert cmp["match"], f"alt={row['alt']}: py={py}, r={row['p']}"


def test_gene_set_test_simulation_matches_r_log_scale():
    # Moderate-signal fixture so the Monte-Carlo p-value is not pinned to
    # 1/(nsim+1) - the simulation path is then a real test of RNG parity
    # rather than a floor-match coincidence.
    stat_vec = _load("R_genesettest_sim_input.csv")["statistic"].values
    r = pd.read_csv(
        FIXTURES / "R_geneSetTest_sim.csv",
        header=None,
        names=["seed", "alt", "p"],
    )
    for _, row in r.iterrows():
        py = gene_set_test(
            np.arange(40),
            stat_vec,
            alternative=row["alt"],
            ranks_only=False,
            nsim=9999,
            rng=int(row["seed"]),
        )
        cmp = compare_pvalues(np.array([row["p"]]), np.array([py]), max_log10_diff=0.5)
        assert cmp["match"], f"seed={row['seed']} alt={row['alt']}: {cmp}"


# ---------------------------------------------------------------------------
# convest / prop_true_null / detection_p_values
# ---------------------------------------------------------------------------


def test_convest_matches_r():
    p = _load("R_propTrueNull_input.csv")["p"].values
    r = _load("R_convest.csv")
    np.testing.assert_allclose(convest(p), r["pi0_default"][0], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(convest(p, niter=200), r["pi0_200iter"][0], rtol=1e-6, atol=1e-12)


def test_prop_true_null_all_methods_match_r():
    p = _load("R_propTrueNull_input.csv")["p"].values
    r = pd.read_csv(FIXTURES / "R_propTrueNull.csv", header=None, names=["method", "pi0"])
    for _, row in r.iterrows():
        py = prop_true_null(p, method=row["method"])
        np.testing.assert_allclose(
            py,
            row["pi0"],
            rtol=1e-6,
            atol=1e-12,
            err_msg=f"method={row['method']}",
        )


def test_prop_true_null_rejects_unknown_method():
    with pytest.raises(ValueError):
        prop_true_null(np.array([0.1, 0.5, 0.9]), method="unknown")


def test_detection_p_values_matches_r():
    x = _load("R_detectionPValues_input.csv").values
    status = _load("R_detectionPValues_status.csv")["status"].values
    r = _load("R_detectionPValues.csv").values
    py = detection_p_values(x, status, negctrl="negative")
    np.testing.assert_allclose(py, r, rtol=1e-8, atol=1e-12)


def test_detection_p_values_raises_without_controls():
    x = np.random.default_rng(0).standard_exponential((10, 2))
    with pytest.raises(ValueError):
        detection_p_values(x, ["regular"] * 10, negctrl="negative")

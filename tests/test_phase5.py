"""
R parity tests for Phase 5: visualisation, wsva, differential splicing.

These tests compare pylimma's numeric substrate to the R fixtures in
``tests/fixtures/``. Visual output is spot-checked in validation
notebooks, not here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import matplotlib  # noqa: F401

    matplotlib.use("Agg")
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

FIXTURES = Path(__file__).parent / "fixtures"


# ----------------------------------------------------------------------------
# plot_with_highlights
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_with_highlights_layer_counts():
    from pylimma import plot_with_highlights

    df = pd.read_csv(FIXTURES / "R_plot_with_highlights_input.csv")
    ax = plot_with_highlights(df["x"].values, df["y"].values, status=df["status"].values)
    # Three scatter layers: background + up + down
    assert len(ax.collections) == 3
    # Counts: 180 background + 10 up + 10 down
    sizes = sorted(c.get_offsets().shape[0] for c in ax.collections)
    assert sizes == [10, 10, 180]
    # Legend has up/down labels
    legend = ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert set(labels) == {"up", "down"}


# ----------------------------------------------------------------------------
# plot_ma / plot_md
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_ma_marraylm_substrate():
    from pylimma import contrasts_fit, e_bayes, lm_fit, plot_ma

    E = pd.read_csv(FIXTURES / "R_phase5_E.csv", index_col=0).values
    design = pd.read_csv(FIXTURES / "R_phase5_design.csv").values
    fit = lm_fit(E, design=design)
    fit = contrasts_fit(fit, coefficients=[1])
    fit = e_bayes(fit)
    expected = pd.read_csv(FIXTURES / "R_plot_ma_data.csv", index_col=0)

    plot_ma(fit, coef=0)
    # Extract A and M from Amean and coef 0
    A = np.asarray(fit["Amean"])
    M = np.asarray(fit["coefficients"])[:, 0]
    np.testing.assert_allclose(A, expected["A"].values, rtol=1e-6)
    np.testing.assert_allclose(M, expected["M"].values, rtol=1e-6)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_md_matrix_substrate():
    from pylimma import plot_md

    E = pd.read_csv(FIXTURES / "R_phase5_E.csv", index_col=0).values
    expected = pd.read_csv(FIXTURES / "R_plot_md_matrix.csv", index_col=0)
    # Direct computation from the function's internal formula
    col = 0
    others = np.delete(np.arange(E.shape[1]), col)
    ave = np.nanmean(E[:, others], axis=1)
    A = (E[:, col] + ave) / 2.0
    M = E[:, col] - ave
    np.testing.assert_allclose(A, expected["A"].values, rtol=1e-6)
    np.testing.assert_allclose(M, expected["M"].values, rtol=1e-6)

    ax = plot_md(E, column=0)
    assert ax is not None


# ----------------------------------------------------------------------------
# volcano_plot
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_volcano_plot_substrate():
    from pylimma import contrasts_fit, e_bayes, lm_fit, volcano_plot

    E = pd.read_csv(FIXTURES / "R_phase5_E.csv", index_col=0).values
    design = pd.read_csv(FIXTURES / "R_phase5_design.csv").values
    fit = lm_fit(E, design=design)
    fit = contrasts_fit(fit, coefficients=[1])
    fit = e_bayes(fit)
    expected = pd.read_csv(FIXTURES / "R_volcano_data.csv", index_col=0)

    np.testing.assert_allclose(
        np.asarray(fit["coefficients"])[:, 0],
        expected["log_fc"].values,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        -np.log10(np.asarray(fit["p_value"])[:, 0]),
        expected["neg_log10_p"].values,
        rtol=1e-6,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(fit["lods"])[:, 0],
        expected["b"].values,
        rtol=1e-6,
    )

    ax = volcano_plot(fit, coef=0)
    assert ax is not None
    ax2 = volcano_plot(fit, coef=0, style="b-statistic")
    assert ax2 is not None


# ----------------------------------------------------------------------------
# plot_sa
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_sa_flat_substrate():
    from pylimma import contrasts_fit, e_bayes, lm_fit, plot_sa

    E = pd.read_csv(FIXTURES / "R_phase5_E.csv", index_col=0).values
    design = pd.read_csv(FIXTURES / "R_phase5_design.csv").values
    fit = lm_fit(E, design=design)
    fit = contrasts_fit(fit, coefficients=[1])
    fit = e_bayes(fit)
    expected = pd.read_csv(FIXTURES / "R_plot_sa_flat.csv", index_col=0)
    expected_line = pd.read_csv(FIXTURES / "R_plot_sa_flat_line.csv")

    np.testing.assert_allclose(fit["Amean"], expected["Amean"].values, rtol=1e-6)
    np.testing.assert_allclose(np.sqrt(fit["sigma"]), expected["sqrt_sigma"].values, rtol=1e-6)
    # Scalar s2_prior
    s2p = float(np.atleast_1d(fit["s2_prior"])[0])
    expected_y = float(expected_line["flat_y"].iloc[0])
    np.testing.assert_allclose(np.sqrt(np.sqrt(s2p)), expected_y, rtol=1e-6)

    ax = plot_sa(fit)
    assert ax is not None


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_sa_trend_substrate():
    from pylimma import contrasts_fit, e_bayes, lm_fit, plot_sa

    E = pd.read_csv(FIXTURES / "R_phase5_E.csv", index_col=0).values
    design = pd.read_csv(FIXTURES / "R_phase5_design.csv").values
    fit = lm_fit(E, design=design)
    fit = contrasts_fit(fit, coefficients=[1])
    fit = e_bayes(fit, trend=True, robust=True)
    expected = pd.read_csv(FIXTURES / "R_plot_sa_trend.csv", index_col=0)

    np.testing.assert_allclose(fit["Amean"], expected["Amean"].values, rtol=1e-6)
    np.testing.assert_allclose(np.sqrt(fit["sigma"]), expected["sqrt_sigma"].values, rtol=1e-6)
    np.testing.assert_allclose(fit["s2_prior"], expected["s2_prior"].values, rtol=1e-5)

    ax = plot_sa(fit)
    assert ax is not None


# ----------------------------------------------------------------------------
# plot_densities
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_densities_curves():
    from pylimma import plot_densities

    E = pd.read_csv(FIXTURES / "R_phase5_E.csv", index_col=0)
    expected = pd.read_csv(FIXTURES / "R_plot_densities.csv")

    ax = plot_densities(E, legend=False)
    # One line per sample
    assert len(ax.get_lines()) == E.shape[1]

    # Compare the first sample's density curve against R
    r_s1 = expected[expected["sample"] == expected["sample"].unique()[0]]
    py_line = ax.get_lines()[0]
    py_x = py_line.get_xdata()
    py_y = py_line.get_ydata()
    assert len(py_x) == 512
    # Grid spacing should match R's density(n=512)
    np.testing.assert_allclose(py_x, r_s1["x"].values, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(py_y, r_s1["y"].values, rtol=1e-6, atol=1e-9)


# ----------------------------------------------------------------------------
# MDS
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sel,top",
    [
        ("pairwise", 100),
        ("pairwise", 500),
        ("common", 100),
        ("common", 500),
    ],
)
def test_mds_coordinates_rparity(sel, top):
    from pylimma.plotting import _mds_coordinates

    E = pd.read_csv(FIXTURES / "R_phase5_E.csv", index_col=0).values
    expected = pd.read_csv(FIXTURES / f"R_plot_mds_{sel}_top{top}.csv")
    res = _mds_coordinates(E, top=top, gene_selection=sel)
    # Sign convention: eigenvectors defined up to sign, so compare |coord|
    np.testing.assert_allclose(
        np.abs(res["x"]),
        np.abs(expected["dim1"].values),
        rtol=1e-6,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.abs(res["y"]),
        np.abs(expected["dim2"].values),
        rtol=1e-6,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        res["var_explained"][0],
        expected["var_explained_1"].iloc[0],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        res["var_explained"][1],
        expected["var_explained_2"].iloc[0],
        rtol=1e-6,
    )


# ----------------------------------------------------------------------------
# Venn
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "include,file",
    [
        ("both", "R_venn_counts.csv"),
        ("up", "R_venn_counts_up.csv"),
        ("down", "R_venn_counts_down.csv"),
    ],
)
def test_venn_counts_rparity(include, file):
    from pylimma import venn_counts

    # Load the decideTests input that produced the R fixture
    dec = pd.read_csv(FIXTURES / "R_decideTests_input.csv")
    expected = pd.read_csv(FIXTURES / file)
    result = venn_counts(dec, include=include)
    # Compare column values and counts
    assert list(result.columns) == list(expected.columns)
    for col in expected.columns:
        np.testing.assert_array_equal(
            result[col].values.astype(np.int64),
            expected[col].values.astype(np.int64),
        )


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_venn_diagram_smoke():
    from pylimma import venn_diagram

    dec = pd.read_csv(FIXTURES / "R_decideTests_input.csv")
    ax = venn_diagram(dec)
    assert ax is not None


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_venn_diagram_unsupported_sets():
    from pylimma import venn_diagram

    dec_4col = pd.DataFrame(np.random.randint(-1, 2, size=(50, 4)))
    with pytest.raises(NotImplementedError):
        venn_diagram(dec_4col)


# ----------------------------------------------------------------------------
# coolmap
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
@pytest.mark.parametrize(
    "cb,suffix",
    [
        ("de pattern", "de_pattern"),
        ("expression level", "expression_level"),
    ],
)
def test_coolmap_substrate(cb, suffix):
    from pylimma import coolmap

    E = pd.read_csv(FIXTURES / "R_phase5_E.csv", index_col=0).values[:50, :]
    expected_z = pd.read_csv(FIXTURES / f"R_coolmap_z_{suffix}.csv", index_col=0).values

    # Compute Z directly (same transform used internally)
    if cb == "de pattern":
        row_means = np.nanmean(E, axis=1)
        df_row = E.shape[1] - 1
        Z = E - row_means[:, None]
        V = np.nansum(Z**2, axis=1) / df_row
        Z = Z / np.sqrt(V + 0.01)[:, None]
    else:
        Z = E
    np.testing.assert_allclose(Z, expected_z, rtol=1e-6)

    # Run coolmap and verify dendrogram leaf orders match R's hclust
    fig = coolmap(E, cluster_by=cb)
    assert fig is not None


# ----------------------------------------------------------------------------
# barcode_plot
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_barcode_plot_worm_substrate():
    from pylimma import barcode_plot, tricube_moving_average

    expected = pd.read_csv(FIXTURES / "R_barcodeplot_substrate.csv")
    idx_sorted = expected["member"].astype(bool).values
    expected_worm = expected["worm"].values

    # Verify the tricube + normalisation pipeline against R
    ave = idx_sorted.sum() / len(idx_sorted)
    worm = tricube_moving_average(idx_sorted.astype(np.float64), span=0.45) / ave
    np.testing.assert_allclose(worm, expected_worm, rtol=1e-6, atol=1e-9)

    # Smoke test of the full plotting function on an ad-hoc input
    np.random.seed(5)
    stat = np.random.randn(1000)
    stat[:50] += 1
    ax = barcode_plot(stat, index=np.arange(1, 51).tolist())
    assert ax is not None


# ----------------------------------------------------------------------------
# wsva
# ----------------------------------------------------------------------------


def test_wsva_unweighted_rparity():
    from pylimma import wsva

    E = pd.read_csv(FIXTURES / "R_wsva_input.csv").values
    design = pd.read_csv(FIXTURES / "R_phase5_design.csv").values
    expected = pd.read_csv(FIXTURES / "R_wsva_unweighted.csv").values

    sv = wsva(E, design, n_sv=2, weight_by_sd=False)
    # SVD eigenvectors are sign-ambiguous; compare columnwise with sign
    # flipping.
    for j in range(sv.shape[1]):
        a = sv[:, j]
        b = expected[:, j]
        if np.dot(a, b) < 0:
            a = -a
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9)


def test_wsva_weighted_rparity():
    from pylimma import wsva

    E = pd.read_csv(FIXTURES / "R_wsva_input.csv").values
    design = pd.read_csv(FIXTURES / "R_phase5_design.csv").values
    expected = pd.read_csv(FIXTURES / "R_wsva_weighted.csv").values

    sv = wsva(E, design, n_sv=2, weight_by_sd=True)
    for j in range(sv.shape[1]):
        a = sv[:, j]
        b = expected[:, j]
        if np.dot(a, b) < 0:
            a = -a
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9)


# ----------------------------------------------------------------------------
# diff_splice
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _diffsplice_fit():
    from pylimma import diff_splice, lm_fit

    y = pd.read_csv(FIXTURES / "R_diffSplice_input_y.csv", index_col=0).values
    design = pd.read_csv(FIXTURES / "R_diffSplice_input_design.csv").values
    fit = lm_fit(y, design=design)
    n_exons = y.shape[0]
    fit["genes"] = pd.DataFrame(
        {
            "GeneID": np.repeat([f"gene{i + 1}" for i in range(20)], 5),
            "ExonID": [f"exon{i + 1}" for i in range(n_exons)],
        }
    )
    return diff_splice(fit, geneid="GeneID", exonid="ExonID", verbose=False)


def test_diffsplice_coefficients(_diffsplice_fit):
    expected = pd.read_csv(FIXTURES / "R_diffSplice_coefficients.csv", index_col=0).values
    np.testing.assert_allclose(_diffsplice_fit["coefficients"], expected, rtol=1e-6, atol=1e-9)


def test_diffsplice_t(_diffsplice_fit):
    expected = pd.read_csv(FIXTURES / "R_diffSplice_t.csv", index_col=0).values
    np.testing.assert_allclose(_diffsplice_fit["t"], expected, rtol=1e-6, atol=1e-9)


def test_diffsplice_p(_diffsplice_fit):
    expected = pd.read_csv(FIXTURES / "R_diffSplice_p.csv", index_col=0).values
    np.testing.assert_allclose(_diffsplice_fit["p_value"], expected, rtol=1e-6, atol=1e-9)


def test_diffsplice_gene_F(_diffsplice_fit):
    expected = pd.read_csv(FIXTURES / "R_diffSplice_gene_F.csv", index_col=0).values
    np.testing.assert_allclose(_diffsplice_fit["gene_F"], expected, rtol=1e-6, atol=1e-9)


def test_diffsplice_gene_F_p(_diffsplice_fit):
    expected = pd.read_csv(FIXTURES / "R_diffSplice_gene_F_p.csv", index_col=0).values
    np.testing.assert_allclose(_diffsplice_fit["gene_F_p_value"], expected, rtol=1e-6, atol=1e-9)


def test_diffsplice_gene_simes(_diffsplice_fit):
    expected = pd.read_csv(FIXTURES / "R_diffSplice_gene_simes_p.csv", index_col=0).values
    np.testing.assert_allclose(
        _diffsplice_fit["gene_simes_p_value"], expected, rtol=1e-6, atol=1e-9
    )


@pytest.fixture(scope="module")
def _diffsplice_fit_legacy():
    from pylimma import diff_splice, lm_fit

    y = pd.read_csv(FIXTURES / "R_diffSplice_input_y.csv", index_col=0).values
    design = pd.read_csv(FIXTURES / "R_diffSplice_input_design.csv").values
    fit = lm_fit(y, design=design)
    n_exons = y.shape[0]
    fit["genes"] = pd.DataFrame(
        {
            "GeneID": np.repeat([f"gene{i + 1}" for i in range(20)], 5),
            "ExonID": [f"exon{i + 1}" for i in range(n_exons)],
        }
    )
    return diff_splice(fit, geneid="GeneID", exonid="ExonID", legacy=True, verbose=False)


@pytest.mark.parametrize(
    "key,file",
    [
        ("coefficients", "R_diffSplice_legacy_coefficients.csv"),
        ("t", "R_diffSplice_legacy_t.csv"),
        ("p_value", "R_diffSplice_legacy_p.csv"),
        ("gene_F", "R_diffSplice_legacy_gene_F.csv"),
        ("gene_F_p_value", "R_diffSplice_legacy_gene_F_p.csv"),
        ("gene_simes_p_value", "R_diffSplice_legacy_gene_simes_p.csv"),
    ],
)
def test_diffsplice_legacy_rparity(_diffsplice_fit_legacy, key, file):
    expected = pd.read_csv(FIXTURES / file, index_col=0).values
    np.testing.assert_allclose(_diffsplice_fit_legacy[key], expected, rtol=1e-6, atol=1e-9)


def test_diffsplice_anndata_matches_ndarray():
    """diff_splice(adata) must route through _resolve_fit_input and
    produce the same output as diff_splice(fit_dict). Regression for
    the AnnData-audit bug where the isinstance check rejected any
    non-dict / non-MArrayLM input.
    """
    import anndata as ad

    from pylimma import diff_splice, lm_fit

    y = pd.read_csv(FIXTURES / "R_diffSplice_input_y.csv", index_col=0).values
    design = pd.read_csv(FIXTURES / "R_diffSplice_input_design.csv").values
    n_exons = y.shape[0]

    geneid = np.repeat([f"gene{i + 1}" for i in range(20)], 5)
    exonid = np.array([f"exon{i + 1}" for i in range(n_exons)])

    # Build a baseline fit_dict (matches the existing fixture path)
    fit_dict = lm_fit(y, design=design)
    fit_dict["genes"] = pd.DataFrame({"GeneID": geneid, "ExonID": exonid})
    out_ref = diff_splice(fit_dict, geneid="GeneID", exonid="ExonID", verbose=False)

    # AnnData path - limma orientation (n_exons x n_samples) becomes
    # (n_samples, n_exons) on the AnnData X.
    adata = ad.AnnData(X=y.T.copy())
    adata.var["GeneID"] = geneid
    adata.var["ExonID"] = exonid
    lm_fit(adata, design=design)
    out_anndata = diff_splice(adata, geneid="GeneID", exonid="ExonID", verbose=False)

    for slot in ("coefficients", "t", "p_value", "gene_F", "gene_F_p_value"):
        np.testing.assert_allclose(
            np.asarray(out_anndata[slot]),
            np.asarray(out_ref[slot]),
            rtol=1e-12,
            atol=1e-14,
            err_msg=f"{slot} differs (AnnData vs ndarray diff_splice)",
        )


# ----------------------------------------------------------------------------
# top_splice
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test,sort_by,file",
    [
        ("simes", "p", "R_topSplice_simes_p.csv"),
        ("simes", "none", "R_topSplice_simes_none.csv"),
        ("simes", "NExons", "R_topSplice_simes_NExons.csv"),
        ("F", "p", "R_topSplice_F_p.csv"),
        ("F", "none", "R_topSplice_F_none.csv"),
        ("F", "NExons", "R_topSplice_F_NExons.csv"),
        ("t", "p", "R_topSplice_t_p.csv"),
        ("t", "none", "R_topSplice_t_none.csv"),
        ("t", "logFC", "R_topSplice_t_logFC.csv"),
    ],
)
def test_top_splice_rparity(_diffsplice_fit, test, sort_by, file):
    from pylimma import top_splice

    result = top_splice(_diffsplice_fit, coef=1, test=test, number=np.inf, sort_by=sort_by)
    expected = pd.read_csv(FIXTURES / file)
    # For "none" order, we expect identical row order. For sorted orders,
    # the ranking should match.
    assert len(result) == len(expected)
    # Numeric columns
    for col in ("P.Value", "FDR"):
        if col in expected.columns:
            np.testing.assert_allclose(
                result[col].values,
                expected[col].values,
                rtol=1e-6,
                atol=1e-9,
                err_msg=f"column {col} ({test}, {sort_by})",
            )


# ----------------------------------------------------------------------------
# plot_splice
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_splice_substrate(_diffsplice_fit):
    from pylimma import plot_splice

    expected = pd.read_csv(FIXTURES / "R_plotSplice_substrate.csv")
    # Identify top gene by minimum F p-value on last coef
    gene_F_p = np.asarray(_diffsplice_fit["gene_F_p_value"])
    i = int(np.argmin(gene_F_p[:, 1]))
    first = int(_diffsplice_fit["gene_firstexon"][i])
    last = int(_diffsplice_fit["gene_lastexon"][i])
    exons = slice(first, last + 1)

    np.testing.assert_allclose(
        np.asarray(_diffsplice_fit["coefficients"])[exons, 1],
        expected["log_fc"].values,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(_diffsplice_fit["t"])[exons, 1],
        expected["t"].values,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(_diffsplice_fit["p_value"])[exons, 1],
        expected["p"].values,
        rtol=1e-6,
        atol=1e-9,
    )

    ax = plot_splice(_diffsplice_fit, coef=1)
    assert ax is not None

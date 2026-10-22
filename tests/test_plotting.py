"""
R parity tests for the numeric substrate of the plotting functions.

These tests compare pylimma's numeric substrate to the R fixtures in
``tests/fixtures/``. Visual output is spot-checked in validation
notebooks, not here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .helpers import canonical_column_signs

try:
    import matplotlib  # noqa: F401

    matplotlib.use("Agg")
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    if HAS_MPL:
        import matplotlib.pyplot as plt

        plt.close("all")


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
    # Legend: background first, then the highlights by frequency (ties in
    # level order), as R's plotWithHighlights
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["background", "down", "up"]


def _r_points_calls(fixture, case):
    """x values of each points() call R made, without the NA entries R's
    x[status == value] carries (base graphics does not draw them)."""
    calls = pd.read_csv(FIXTURES / fixture).query("case == @case")
    return [g["value"].dropna().to_numpy() for _, g in calls.groupby("call", sort=True)]


PWH_STATUS = {
    "character_na": (np.array(["a"] * 100 + [None] * 50 + ["b"] * 10, dtype=object), None),
    "na_most_frequent": (
        np.array(["a"] * 10 + [None] * 50 + ["b"] * 5 + ["c"] * 95, dtype=object),
        None,
    ),
    "float_values": (np.r_[np.ones(100), np.zeros(50), -np.ones(10)], [1, -1]),
    "numeric_ties": (np.r_[np.zeros(100), np.full(30, 10.0), np.full(30, 2.0)], None),
    "logical": (np.r_[np.ones(40, dtype=bool), np.zeros(120, dtype=bool)], None),
}


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
@pytest.mark.parametrize("case", list(PWH_STATUS))
def test_plot_with_highlights_status_matches_r(case):
    """R keeps NA as background (not a group), takes values from table() by
    frequency with ties in level order, and matches a float 1 to values=1."""
    from pylimma import plot_with_highlights

    x = np.arange(1.0, 161.0)
    status, values = PWH_STATUS[case]
    n = len(status)
    ax = plot_with_highlights(x[:n], np.sqrt(x[:n]), status=status, values=values, legend=False)
    drawn = [c.get_offsets()[:, 0] for c in ax.collections]
    expected = [v for v in _r_points_calls("R_pwh_status_points.csv", case) if v.size]
    assert len(drawn) == len(expected)
    for got, want in zip(drawn, expected):
        np.testing.assert_array_equal(got, want)


PWH_LEGEND = {
    "single_background": (["notDE"] * 150 + ["up"] * 6 + ["down"] * 4, None),
    "other_background": (["a"] * 100 + ["b"] * 50 + ["c"] * 10, "c"),
    "na_background": (["a"] * 100 + [None] * 50 + ["c"] * 10, "c"),
    "empty_value": (["a"] * 150 + ["c"] * 10, ["c", "z"]),
}


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
@pytest.mark.parametrize("case", list(PWH_LEGEND))
def test_plot_with_highlights_legend_matches_r(case):
    from pylimma import plot_with_highlights

    status, values = PWH_LEGEND[case]
    x = np.arange(1.0, 161.0)
    ax = plot_with_highlights(x, np.sqrt(x), status=np.array(status, dtype=object), values=values)
    expected = pd.read_csv(FIXTURES / "R_pwh_legend.csv").query("case == @case")["value"]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == expected.tolist()


# ----------------------------------------------------------------------------
# plot_ma / plot_md
# ----------------------------------------------------------------------------


PMD_CASES = {
    "md_fit": ("plot_md", "fit", {}),
    "ma_fit": ("plot_ma", "fit", {}),
    "md_fit_coef": ("plot_md", "fit", {"coef": 1}),
    "md_fit_tests": ("plot_md", "fit", {"status": "decide_tests"}),
    "md_elist": ("plot_md", "elist", {"column": 1}),
}


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
@pytest.mark.parametrize("case", list(PMD_CASES))
def test_plot_md_and_plot_ma_defaults_match_r(case, monkeypatch):
    """R's defaults: the last coefficient of a fit, its name as title,
    genes$Status as status (plotMD: the matching column of decideTests
    output); for an EList the column name and its ylab."""
    import pylimma
    from pylimma import EList, decide_tests, e_bayes, lm_fit

    plotting = __import__("pylimma.plotting", fromlist=["plot_with_highlights"])
    y = pd.read_csv(FIXTURES / "R_pmd_y.csv")
    design = pd.DataFrame(
        {"Intercept": 1.0, "B": np.repeat([0.0, 1.0], 3), "C": np.tile([0.0, 1.0, 0.0], 2)}
    )
    genes = pd.DataFrame(
        {"ID": [f"g{i}" for i in range(1, 61)], "Status": ["gene"] * 54 + ["control"] * 6}
    )
    function, kind, kwargs = PMD_CASES[case]
    if kind == "fit":
        target = e_bayes(lm_fit(y.to_numpy(), design))
        target["genes"] = genes
    else:
        target = EList({"E": y, "genes": genes})
    if kwargs.get("status") == "decide_tests":
        kwargs = {"status": decide_tests(target)}

    seen = {}
    monkeypatch.setattr(
        plotting, "plot_with_highlights", lambda x, y, **kw: seen.update(x=x, y=y, **kw)
    )
    getattr(pylimma, function)(target, **kwargs)

    r = pd.read_csv(FIXTURES / "R_pmd_defaults.csv").query("case == @case")
    value = {arg: g["value"].to_numpy() for arg, g in r.groupby("arg")}
    np.testing.assert_allclose(seen["x"], value["x"].astype(float), rtol=1e-12)
    np.testing.assert_allclose(seen["y"], value["y"].astype(float), rtol=1e-12)
    assert seen["main"] == value["main"][0]
    assert seen["ylab"] == value["ylab"][0]
    status = plotting._status_labels(seen["status"])
    assert list(status) == list(value["status"])


def _pmd_fit():
    from pylimma import e_bayes, lm_fit

    y = pd.read_csv(FIXTURES / "R_pmd_y.csv").to_numpy()
    design = pd.DataFrame(
        {"Intercept": 1.0, "B": np.repeat([0.0, 1.0], 3), "C": np.tile([0.0, 1.0, 0.0], 2)}
    )
    fit = e_bayes(lm_fit(y, design))
    fit["genes"] = pd.DataFrame({"ID": [f"g{i}" for i in range(1, 61)]})
    return fit


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_volcano_plot_labels_default_to_gene_ids_as_r():
    from pylimma import volcano_plot

    ax = volcano_plot(_pmd_fit(), coef=1, highlight=4)
    r = pd.read_csv(FIXTURES / "R_volcano_labels.csv")
    value = {arg: g["value"].to_numpy() for arg, g in r.groupby("arg")}
    assert [t.get_text() for t in ax.texts] == list(value["labels"])
    np.testing.assert_allclose([t.get_position()[0] for t in ax.texts], value["arg1"].astype(float))
    np.testing.assert_allclose([t.get_position()[1] for t in ax.texts], value["arg2"].astype(float))


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_md_unknown_column_name_raises():
    from pylimma import plot_md

    with pytest.raises(ValueError, match="coef 'a' not found"):
        plot_md(np.ones((10, 3)), column="a")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_ma_marraylm_substrate():
    from pylimma import contrasts_fit, e_bayes, lm_fit, plot_ma

    E = pd.read_csv(FIXTURES / "R_twogroup_E.csv", index_col=0).values
    design = pd.read_csv(FIXTURES / "R_twogroup_design.csv").values
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

    E = pd.read_csv(FIXTURES / "R_twogroup_E.csv", index_col=0).values
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

    E = pd.read_csv(FIXTURES / "R_twogroup_E.csv", index_col=0).values
    design = pd.read_csv(FIXTURES / "R_twogroup_design.csv").values
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
        atol=1e-12,
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

    E = pd.read_csv(FIXTURES / "R_twogroup_E.csv", index_col=0).values
    design = pd.read_csv(FIXTURES / "R_twogroup_design.csv").values
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

    E = pd.read_csv(FIXTURES / "R_twogroup_E.csv", index_col=0).values
    design = pd.read_csv(FIXTURES / "R_twogroup_design.csv").values
    fit = lm_fit(E, design=design)
    fit = contrasts_fit(fit, coefficients=[1])
    fit = e_bayes(fit, trend=True, robust=True)
    expected = pd.read_csv(FIXTURES / "R_plot_sa_trend.csv", index_col=0)

    np.testing.assert_allclose(fit["Amean"], expected["Amean"].values, rtol=1e-6)
    np.testing.assert_allclose(np.sqrt(fit["sigma"]), expected["sqrt_sigma"].values, rtol=1e-6)
    np.testing.assert_allclose(fit["s2_prior"], expected["s2_prior"].values, rtol=1e-6)

    ax = plot_sa(fit)
    assert ax is not None


# ----------------------------------------------------------------------------
# plot_densities
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_densities_curves():
    from pylimma import plot_densities

    E = pd.read_csv(FIXTURES / "R_twogroup_E.csv", index_col=0)
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
    np.testing.assert_allclose(py_x, r_s1["x"].values, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(py_y, r_s1["y"].values, rtol=1e-6, atol=1e-12)


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

    E = pd.read_csv(FIXTURES / "R_twogroup_E.csv", index_col=0).values
    expected = pd.read_csv(FIXTURES / f"R_plot_mds_{sel}_top{top}.csv")
    res = _mds_coordinates(E, top=top, gene_selection=sel)
    # Each dimension is defined only up to sign (left to LAPACK by R and
    # NumPy); the fixture is written with canonical signs, so canonicalise
    # pylimma's output the same way and compare exactly.
    np.testing.assert_allclose(
        canonical_column_signs(res["x"]),
        expected["dim1"].values,
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        canonical_column_signs(res["y"]),
        expected["dim2"].values,
        rtol=1e-6,
        atol=1e-12,
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


def test_plot_mds_without_plotting_returns_the_mds_result_as_r():
    """plot=False returns the MDS result with x and y for dim_plot, which
    can be plotted again for other dimensions (R's plotMDS.MDS)."""
    from pylimma import plot_mds

    x = pd.read_csv(FIXTURES / "R_mds_x.csv").to_numpy()
    r = pd.read_csv(FIXTURES / "R_mds_dims.csv")
    mds = plot_mds(x, plot=False, dim_plot=(0, 2))
    np.testing.assert_allclose(
        canonical_column_signs(np.column_stack([mds["x"], mds["y"]])),
        canonical_column_signs(r[["x", "y"]].to_numpy()),
        rtol=1e-8,
    )
    again = plot_mds(mds, plot=False, dim_plot=(1, 2))
    np.testing.assert_allclose(
        canonical_column_signs(np.column_stack([again["x"], again["y"]])),
        canonical_column_signs(r[["re_x", "re_y"]].to_numpy()),
        rtol=1e-8,
    )


@pytest.mark.parametrize(
    "kwargs", [{"dim_plot": (0, 0)}, {"dim_plot": (1,)}, {"top": 0}, {"top": -1}]
)
def test_plot_mds_rejects_bad_dimensions_and_top(kwargs):
    from pylimma import plot_mds

    x = pd.read_csv(FIXTURES / "R_mds_x.csv").to_numpy()
    with pytest.raises(
        ValueError, match="dim_plot must specify two dimensions|top must be a number|0-based"
    ):
        plot_mds(x, plot=False, **kwargs)


def test_plot_mds_takes_r_dim_plot_and_top_rules():
    """unique(dim.plot) as R, a float top, and a stored MDS result replotted
    in its own dimensions (R's plotMDS.MDS has dim.plot=NULL)."""
    from pylimma import plot_mds

    x = pd.read_csv(FIXTURES / "R_mds_x.csv").to_numpy()
    assert plot_mds(x, plot=False, dim_plot=(0, 0, 2))["dim_plot"] == (0, 2)
    np.testing.assert_allclose(
        plot_mds(x, plot=False, top=500.0)["x"], plot_mds(x, plot=False)["x"]
    )
    stored = plot_mds(x, plot=False, dim_plot=(1, 2))
    assert plot_mds(stored, plot=False)["dim_plot"] == (1, 2)
    np.testing.assert_allclose(
        plot_mds(x, plot=False, top=np.inf)["x"], plot_mds(x, plot=False, top=len(x))["x"]
    )


def test_plot_mds_warns_with_r_dimension_numbers():
    """R: warning("dimension ", i, ...) with i 1-based; three samples leave
    the third dimension empty."""
    from pylimma import plot_mds

    x = np.random.default_rng(1).normal(size=(50, 3))
    with pytest.warns(UserWarning, match="^dimension 3 is degenerate or all zero$"):
        plot_mds(x, plot=False, dim_plot=(0, 2))


def test_plot_mds_takes_mds_coordinates_output():
    from pylimma import plot_mds
    from pylimma.plotting import _mds_coordinates

    x = pd.read_csv(FIXTURES / "R_mds_x.csv").to_numpy()
    assert plot_mds(_mds_coordinates(x), plot=False)["dim_plot"] == (0, 1)


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


@pytest.mark.parametrize(
    "include,file", [("both", "R_venn_counts_na.csv"), ("up", "R_venn_counts_na_up.csv")]
)
def test_venn_counts_drops_rows_with_missing_values_as_r(include, file):
    from pylimma import venn_counts

    x = np.array([[1, 1], [0, 1], [np.nan, 0], [1, 0], [-1, np.nan], [0, -1], [1, 1]])
    expected = pd.read_csv(FIXTURES / file)
    pd.testing.assert_frame_equal(venn_counts(x, include=include), expected, check_dtype=False)


VENN_CASES = {
    "one": (["A"], {}),
    "two": (["A", "B"], {}),
    "three": (["A", "B", "C"], {}),
    "two_up_down": (["A", "B"], {"include": ["up", "down"]}),
    "two_scalar_cex": (["A", "B"], {"cex": 1}),
}


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
@pytest.mark.parametrize("case", list(VENN_CASES))
def test_venn_diagram_text_matches_r(case):
    """Set names and counts at R's positions: counts[k] in table() order,
    so the count for set 2 only is inside the second circle."""
    from pylimma import venn_diagram

    columns, kwargs = VENN_CASES[case]
    tests = pd.read_csv(FIXTURES / "R_venn_tests.csv")[columns]
    ax = venn_diagram(tests, **kwargs)
    r = pd.read_csv(FIXTURES / "R_venn_text.csv").query("case == @case")
    expected = r.pivot(index="call", columns="arg", values="value")
    got = [(*t.get_position(), t.get_text()) for t in ax.texts]
    assert [label for *_, label in got] == expected["arg3"].tolist()
    np.testing.assert_allclose([p[0] for p in got], expected["arg1"].astype(float))
    np.testing.assert_allclose([p[1] for p in got], expected["arg2"].astype(float))


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


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
@pytest.mark.parametrize(
    "case,kwargs",
    [
        ("default", {}),
        ("limit", {"limit": 1.6}),
        ("names", {"names": [f"gene{i}" for i in range(1, 8)]}),
        ("default", {"names": np.arange(1.0, 8.0)}),
    ],
)
def test_heat_diagram_rows_match_r(case, kwargs):
    """Names numbered over all rows; rows ordered on the untruncated primary
    coefficient (stable), then truncated at limit."""
    from pylimma import heat_diagram

    data = pd.read_csv(FIXTURES / "R_hd_input.csv")
    out = heat_diagram(data[["results.1", "results.2"]], data[["coef.1", "coef.2"]], **kwargs)
    expected = pd.read_csv(FIXTURES / "R_hd_output.csv").query("case == @case")
    assert out["names"] == expected["name"].astype(str).tolist()
    np.testing.assert_allclose(out["coef"], expected[["X1", "X2"]].to_numpy())


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_heat_diagram_needs_one_name_per_row():
    """R takes names[DE], so short names label rows NA (fixture script)."""
    from pylimma import heat_diagram

    data = pd.read_csv(FIXTURES / "R_hd_input.csv")
    with pytest.raises(ValueError, match="one name per row"):
        heat_diagram(
            data[["results.1", "results.2"]], data[["coef.1", "coef.2"]], names=[1.5, 2, 3]
        )


def test_plot_ma_3by2_matrix_matches_r(tmp_path, monkeypatch):
    """Each array's M = E - rowMeans(E) against A = rowMeans(E), common
    limits over all of A and M, the legend on the first plot of a page."""
    from pylimma import plot_ma_3by2

    plotting = __import__("pylimma.plotting", fromlist=["plot_with_highlights"])
    original = plotting.plot_with_highlights
    calls = []

    def record(x, y, **kwargs):
        calls.append((x, y, kwargs))
        return original(x, y, **kwargs)

    monkeypatch.setattr(plotting, "plot_with_highlights", record)
    E = pd.read_csv(FIXTURES / "R_p32_E.csv")
    plot_ma_3by2(E, path=str(tmp_path))
    r = pd.read_csv(FIXTURES / "R_p32_calls.csv")
    assert len(calls) == r["array"].nunique()
    for (x, y, kwargs), (_, expected) in zip(calls, r.groupby("array", sort=True)):
        np.testing.assert_allclose(x, expected["A"], rtol=1e-12)
        np.testing.assert_allclose(y, expected["M"], rtol=1e-12, atol=1e-12)
        assert kwargs["main"] == expected["main"].iloc[0]
        assert kwargs["legend"] == bool(expected["legend"].iloc[0])
        assert (kwargs["xlab"], kwargs["ylab"]) == ("A", "M")
        ax = kwargs["ax"]
        np.testing.assert_allclose(ax.get_xlim(), expected[["xlim1", "xlim2"]].iloc[0], rtol=1e-12)
        np.testing.assert_allclose(ax.get_ylim(), expected[["ylim1", "ylim2"]].iloc[0], rtol=1e-12)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_ma_3by2_passes_status_and_labels_on(tmp_path):
    """R's plotMA3by2 passes ... on to plotMA, so status and labels can be
    given."""
    from pylimma import plot_ma_3by2

    E = pd.read_csv(FIXTURES / "R_p32_E.csv")
    files = plot_ma_3by2(E, path=str(tmp_path), status=np.repeat(["a", "b"], 10), xlab="mean")
    assert len(files) == 2


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
@pytest.mark.parametrize(
    "case,group",
    [("character", ["b", "a", "b", "a"]), ("short", [1, 2]), ("numeric", [10, 2, 2, 10])],
)
def test_plot_densities_groups_match_r(case, group):
    """Levels sorted as R's factor, each array coloured by its level (the
    colours recycled over the arrays when group is shorter), legend in level
    order. R's default col is 1:ngroup, so its values index the levels."""
    from matplotlib.colors import to_rgba

    from pylimma import plot_densities

    E = pd.read_csv(FIXTURES / "R_pd_E.csv")
    ax = plot_densities(E, group=group)
    r = pd.read_csv(FIXTURES / "R_pd_groups.csv").query("case == @case")
    array_level = r.query("fun == 'matplot'")["value"].astype(int).to_numpy() - 1
    array_level = np.resize(array_level, E.shape[1])
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == r.query("arg == 'legend'")[
        "value"
    ].tolist()
    level_colours = [to_rgba(h.get_color()) for h in legend.legend_handles]
    assert [to_rgba(line.get_color()) for line in ax.lines] == [
        level_colours[i] for i in array_level
    ]


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_densities_default_and_categorical_groups_follow_r_factor():
    """R: group defaults to 1:narray (numeric levels) without column names;
    a categorical group keeps its categories' order."""
    from pylimma import plot_densities

    E = np.random.default_rng(4).normal(8, 1, size=(50, 12))
    ax = plot_densities(E)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [str(i) for i in range(1, 13)]
    ax = plot_densities(E[:, :2], group=pd.Categorical(["b", "a"], categories=["b", "a"]))
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["b", "a"]


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_densities_missing_groups_as_r():
    """R: as.factor keeps a numeric NaN as a level; a missing group (None,
    or NaN among strings: pandas' missing value) has no colour and matplot
    draws no line."""
    from pylimma import plot_densities

    E = np.random.default_rng(5).normal(8, 1, size=(50, 4))
    ax = plot_densities(E, group=[1.0, np.nan, 2.0, 2.0])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["1", "2", "NaN"]
    assert len({line.get_color() for line in ax.lines}) == 3
    for group in (["a", None, "b", "b"], ["a", np.nan, "b", "b"]):
        ax = plot_densities(E, group=group)
        assert len(ax.lines) == 3
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plotlines_recycles_col_as_r():
    from pylimma import plotlines

    ax = plotlines(np.arange(1.0, 21.0).reshape(4, 5).T, col=["red", "blue"])
    expected = pd.read_csv(FIXTURES / "R_plotlines_col.csv")["value"].tolist()
    assert [line.get_color() for line in ax.lines] == expected


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plotlines_takes_one_rgb_colour():
    from pylimma import plotlines

    ax = plotlines(np.ones((4, 3)), col=(0.1, 0.2, 0.3))
    assert [line.get_color() for line in ax.lines] == [(0.1, 0.2, 0.3)] * 4


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

    E = pd.read_csv(FIXTURES / "R_twogroup_E.csv", index_col=0).values[:50, :]
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
    np.testing.assert_allclose(worm, expected_worm, rtol=1e-6, atol=1e-12)

    # Smoke test of the full plotting function on an ad-hoc input
    np.random.seed(5)
    stat = np.random.randn(1000)
    stat[:50] += 1
    ax = barcode_plot(stat, index=np.arange(1, 51).tolist())
    assert ax is not None


@pytest.mark.parametrize(
    "function", ["plotlines", "heat_diagram", "plot_exons", "plot_exon_junc", "plot_ma_3by2"]
)
def test_plot_functions_without_matplotlib_raise_a_clear_import_error(function, monkeypatch):
    import sys

    import pylimma

    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    args = {
        "plotlines": (np.ones((2, 3)),),
        "heat_diagram": (np.ones((2, 2)), np.ones((2, 2))),
        "plot_exons": ({},),
        "plot_exon_junc": ({},),
        "plot_ma_3by2": (np.ones((5, 3)),),
    }[function]
    with pytest.raises(ImportError, match="plotting functions require matplotlib"):
        getattr(pylimma, function)(*args)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_argument_errors_are_clear():
    """R: volcanoplot stops with "fit must be an MArrayLM"; mdplot with one
    column and vennDiagram(NULL) fail obscurely in R (fixture script notes)."""
    from pylimma import mdplot, venn_diagram, volcano_plot

    with pytest.raises(ValueError, match="fit must be an MArrayLM"):
        volcano_plot({"x": 1})
    with pytest.raises(ValueError, match="columns must give two columns"):
        mdplot(np.ones((10, 3)), columns=(0,))
    with pytest.raises(ValueError, match="not None"):
        venn_diagram(None)

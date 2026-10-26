"""
Interface and error-handling tests for goana / kegga / goana_trend that the
value fixtures in test_r_parity.py do not cover. R's behaviour for each case
is recorded in ``tests/fixtures/generate_all_fixtures.R``.
"""

from pathlib import Path

import pandas as pd
import pytest

import pylimma

FIXTURES = Path(__file__).parent / "fixtures"


def _kegga_inputs():
    gp = pd.read_csv(FIXTURES / "R_kegga_genepathway.csv")
    pn = pd.read_csv(FIXTURES / "R_kegga_pathnames.csv")
    universe = pd.read_csv(FIXTURES / "R_goana_universe.csv")["gene_id"].tolist()
    return gp, pn, universe


@pytest.mark.parametrize("fn_name", ["goana", "kegga"])
def test_single_gene_id_is_one_de_set(fn_name):
    """R: a non-list de becomes list(DE = de), so one gene ID is one set."""
    gp, pn, universe = _kegga_inputs()
    kwargs = {"pathway_names": pn} if fn_name == "kegga" else {}
    fn = getattr(pylimma, fn_name)
    pd.testing.assert_frame_equal(
        fn("gene001", gp, universe=universe, **kwargs), fn(["gene001"], gp, universe=universe, **kwargs)
    )


@pytest.mark.parametrize("fn_name", ["goana", "kegga"])
def test_data_frame_de_raises(fn_name):
    """R stops for a data.frame de (it would otherwise be read as a list of
    columns); pylimma read the column labels as gene IDs."""
    gp, _, universe = _kegga_inputs()
    de = pd.DataFrame({"x": ["gene001", "gene002"]})
    with pytest.raises(ValueError, match="It should not be a data.frame"):
        getattr(pylimma, fn_name)(de, gp, universe=universe)


@pytest.mark.parametrize("de", [None, {"a": None}])
def test_missing_de_set_raises(de):
    """R: is.vector(NULL) is FALSE, "components of de should be vectors"."""
    gp, _, universe = _kegga_inputs()
    with pytest.raises(ValueError, match="^components of de should be vectors$"):
        pylimma.kegga(de, gp, universe=universe)


@pytest.mark.parametrize("fn_name", ["goana", "kegga"])
def test_list_of_sets_and_dict_views(fn_name):
    """R: an unnamed list of sets is named DE1, DE2, ...; any iterable of IDs
    (a dict view included) is one set."""
    gp, _, universe = _kegga_inputs()
    fn = getattr(pylimma, fn_name)
    expected = fn({"DE1": ["gene001", "gene002"], "DE2": ["gene003"]}, gp, universe=universe)
    pd.testing.assert_frame_equal(fn([["gene001", "gene002"], ["gene003"]], gp, universe=universe), expected)
    up = dict.fromkeys(["gene001", "gene002"], 1.0)
    pd.testing.assert_frame_equal(
        fn({"DE1": up.keys(), "DE2": {"gene003"}}, gp, universe=universe), expected
    )


@pytest.mark.parametrize(
    "fn_name, message", [("kegga", "^No genes in universe$"), ("goana", "^No annotated genes found in universe$")]
)
def test_empty_universe_raises(fn_name, message):
    """R: kegga stops "No genes in universe"; goana's universe is then
    empty after keeping annotated genes."""
    gp, _, _ = _kegga_inputs()
    with pytest.raises(ValueError, match=message):
        getattr(pylimma, fn_name)(["gene001"], gp, universe=[])


def test_kegga_fit_passes_restrict_universe():
    """kegga.MArrayLM passes restrict.universe on to kegga.default."""
    import numpy as np

    gp, pn, _ = _kegga_inputs()
    genes = [f"gene{i:03d}" for i in range(1, 121)]
    rng = np.random.default_rng(1)
    y = rng.normal(size=(120, 6))
    y[:15, 3:] += 3
    fit = pylimma.e_bayes(pylimma.lm_fit(pd.DataFrame(y, index=genes), np.column_stack([np.ones(6), np.repeat([0, 1], 3)])))
    tab = pylimma.top_table(fit, coef=1, number=np.inf, sort_by="none")
    up = [g for g, q, lfc in zip(genes, tab["adj_p_value"], tab["log_fc"]) if q < 0.05 and lfc > 0]
    down = [g for g, q, lfc in zip(genes, tab["adj_p_value"], tab["log_fc"]) if q < 0.05 and lfc < 0]
    pd.testing.assert_frame_equal(
        pylimma.kegga(fit, gp, pathway_names=pn, restrict_universe=True),
        pylimma.kegga({"Up": up, "Down": down}, gp, pathway_names=pn, universe=genes, restrict_universe=True),
    )


@pytest.mark.parametrize("fn_name", ["goana", "kegga"])
def test_covariate_and_prior_ignored_without_universe(fn_name):
    """R: "Ignoring covariate and null.prob because universe not set", then
    the plain hypergeometric test."""
    gp, _, _ = _kegga_inputs()
    fn = getattr(pylimma, fn_name)
    with pytest.warns(UserWarning, match="universe not set"):
        ignored = fn(["gene001", "gene002"], gp, covariate=[1.0, 2.0, 3.0], prior_prob=[0.1, 0.2, 0.3])
    pd.testing.assert_frame_equal(ignored, fn(["gene001", "gene002"], gp))


@pytest.mark.parametrize("fn_name", ["goana", "kegga"])
def test_covariate_without_de_genes_is_dropped(fn_name):
    """R: with no DE gene in the universe, goanaTrend is not run ("No DE
    genes") and the plain test is used."""
    gp, _, universe = _kegga_inputs()
    fn = getattr(pylimma, fn_name)
    covariate = [i / 200 for i in range(1, 201)]
    with pytest.warns(UserWarning, match="No DE genes"):
        dropped = fn("not_a_gene", gp, universe=universe, covariate=covariate)
    pd.testing.assert_frame_equal(dropped, fn("not_a_gene", gp, universe=universe))


@pytest.mark.parametrize("fn_name", ["goana", "kegga"])
@pytest.mark.parametrize("arg", ["prior_prob", "covariate"])
def test_null_prob_needs_wallenius(fn_name, arg):
    """R switches to Wallenius' noncentral hypergeometric test (BiasedUrn),
    which is not ported; pylimma ignored prior_prob."""
    gp, _, universe = _kegga_inputs()
    with pytest.raises(NotImplementedError, match="BiasedUrn"):
        getattr(pylimma, fn_name)(["gene001"], gp, universe=universe, **{arg: [i / 200 for i in range(1, 201)]})


def _two_group_fit():
    import numpy as np

    genes = [f"gene{i:03d}" for i in range(1, 121)]
    y = np.random.default_rng(1).normal(size=(120, 6))
    y[:15, 3:] += 3
    y[15:25, 3:] -= 3
    design = pd.DataFrame({"Intercept": 1.0, "B": np.repeat([0.0, 1.0], 3)})
    return pylimma.e_bayes(pylimma.lm_fit(pd.DataFrame(y, index=genes), design)), genes


@pytest.mark.parametrize("fn_name", ["goana", "kegga"])
def test_coef_by_name(fn_name):
    """R takes coef as a column number or name (length(coef) == 1); a name
    always raised "Only one coef can be specified."."""
    gp, _, _ = _kegga_inputs()
    fit, _ = _two_group_fit()
    fn = getattr(pylimma, fn_name)
    by_number = fn(fit, gp, coef=1)
    pd.testing.assert_frame_equal(fn(fit, gp, coef="B"), by_number)
    contrast = pd.DataFrame({"B_vs_A": [0.0, 1.0]}, index=["Intercept", "B"])
    contrast_fit = pylimma.e_bayes(pylimma.contrasts_fit(fit, contrasts=contrast))
    pd.testing.assert_frame_equal(fn(contrast_fit, gp, coef="B_vs_A"), by_number)


@pytest.mark.parametrize("fn_name", ["goana", "kegga"])
@pytest.mark.parametrize("arg", ["universe", "covariate"])
def test_fit_defines_its_own_universe_and_covariate(fn_name, arg):
    """R stops: "kegga.MArrayLM defines its own universe"; a covariate
    collides with the one the fit method passes on (R: "formal argument
    matched by multiple actual arguments"). pylimma ignored universe."""
    gp, _, universe = _kegga_inputs()
    fit, genes = _two_group_fit()
    value = genes if arg == "universe" else [1.0] * len(genes)
    with pytest.raises(ValueError, match=f"defines its own {arg}"):
        getattr(pylimma, fn_name)(fit, gp, **{arg: value})


@pytest.mark.parametrize("fn_name", ["goana", "kegga"])
def test_numpy_false_trend_is_false(fn_name):
    """R: is.logical(trend); a numpy boolean is a logical too."""
    import numpy as np

    gp, _, universe = _kegga_inputs()
    fit, _ = _two_group_fit()
    fn = getattr(pylimma, fn_name)
    pd.testing.assert_frame_equal(fn(fit, gp, trend=np.False_), fn(fit, gp, trend=False))
    pd.testing.assert_frame_equal(
        fn(["gene001"], gp, universe=universe, trend=np.False_), fn(["gene001"], gp, universe=universe)
    )


def test_pathway_names_first_row_across_id_types():
    """IDs are compared as text (R: match on character IDs), so 10 and "10"
    are the same pathway and the first row names it."""
    gp = pd.DataFrame({"gene": ["g1", "g2", "g3"], "path": ["10", "10", "20"]})
    names = pd.DataFrame({"id": [10, "10", 20], "name": ["first", "second", "other"]}, dtype=object)
    tab = pylimma.kegga(["g1"], gp, pathway_names=names, universe=["g1", "g2", "g3"])
    assert tab.loc["10", "pathway"] == "first"


@pytest.mark.parametrize("gene_path_ids", [[10.0, 20.0], [10, 20]])
def test_pathway_names_numeric_ids(gene_path_ids):
    """Float pathway IDs (as read_csv gives for a column with a gap) name
    their pathways: R compares as.character(10) == "10" (stopifnot in
    generate_all_fixtures.R)."""
    genes = [f"g{i}" for i in range(1, 121)]
    gp = pd.DataFrame({"gene": genes, "path": [gene_path_ids[0] if i < 30 else gene_path_ids[1] for i in range(120)]})
    names = pd.DataFrame({"id": [10.0, 20.0], "name": ["ten", "twenty"]})
    tab = pylimma.kegga(genes[:10], gp, pathway_names=names, universe=genes)
    assert list(tab.loc[["10", "20"], "pathway"]) == ["ten", "twenty"]


def _goana_table():
    gp = pd.read_csv(FIXTURES / "R_goana_genepathway.csv")
    universe = pd.read_csv(FIXTURES / "R_goana_universe.csv")["gene_id"].tolist()
    return pylimma.goana({"Up": universe[:10], "Down": universe[10:20]}, gp, universe=universe)


def test_top_go_ontology_string_and_abbreviation():
    """R: match.arg(unique(ontology), several.ok = TRUE), so one string and
    an unambiguous abbreviation both work; it iterated the characters."""
    results = _goana_table()
    bp = pylimma.top_go(results, ontology=["BP", "BP"])
    assert set(bp["ontology"]) == {"BP"}
    pd.testing.assert_frame_equal(pylimma.top_go(results, ontology="BP"), bp)
    pd.testing.assert_frame_equal(pylimma.top_go(results, ontology="B"), bp)


@pytest.mark.parametrize("ontology", [["BP", "XX"], "XX", []])
def test_top_go_invalid_ontology_raises(ontology):
    """R drops "XX" when another element matches; pylimma raises for any
    element that matches nothing (known_differences.rst)."""
    with pytest.raises(ValueError, match="'arg'"):
        pylimma.top_go(_goana_table(), ontology=ontology)


@pytest.mark.parametrize("fn_name", ["top_go", "top_kegg"])
def test_top_number_inf_and_nan(fn_name):
    """R caps number at the row count (Inf gives every row) and fails on
    NaN with "missing value where TRUE/FALSE needed"; as top_table, pylimma
    raises a clear error for NaN."""
    import numpy as np

    gp, pn, universe = _kegga_inputs()
    results = _goana_table() if fn_name == "top_go" else pylimma.kegga(universe[:10], gp, pathway_names=pn, universe=universe)
    fn = getattr(pylimma, fn_name)
    pd.testing.assert_frame_equal(fn(results, number=np.inf), fn(results, number=len(results)))
    with pytest.raises(ValueError, match="^number must not be NaN$"):
        fn(results, number=np.nan)


@pytest.mark.parametrize("fn_name", ["goana", "kegga"])
@pytest.mark.parametrize("names", [("A", "a"), ("Up", "N")])
def test_set_names_that_clash_in_lower_case_raise(fn_name, names):
    """Output columns are lower-case set names; R keeps "A" and "a" as two
    sets, where pylimma silently kept only one (known_differences.rst)."""
    gp, _, universe = _kegga_inputs()
    de = {names[0]: universe[:10], names[1]: universe[:2]}
    with pytest.raises(ValueError, match="same output column"):
        getattr(pylimma, fn_name)(de, gp, universe=universe)


@pytest.mark.parametrize(
    "index_de, message",
    [([-1], "between 0 and 9"), ([10], "between 0 and 9"), ([1.5], "whole numbers"), (["a"], "whole numbers")],
)
def test_goana_trend_invalid_positions_raise(index_de, message):
    """R excludes a negative position, truncates a fraction and extends
    isDE past the end (generate_all_fixtures.R); pylimma raises
    (known_differences.rst). It wrapped negatives and raised IndexError."""
    import numpy as np

    with pytest.raises(ValueError, match=message):
        pylimma.goana_trend(index_de, np.arange(1, 11) / 10)


def test_goana_trend_whole_number_floats_are_positions():
    import numpy as np

    covariate = np.arange(1, 11) / 10
    np.testing.assert_array_equal(
        pylimma.goana_trend([1.0, 4.0], covariate), pylimma.goana_trend([1, 4], covariate)
    )


def test_goana_trend_plot_draws_r_trend():
    """R's goanaTrend(plot=TRUE) draws p.o against covariate rank 1..n
    (lwd 2) with ylim c(0, ceiling(max(p) * n) / n), and a tick of height
    ylim[2] / 9 at the rank of each DE gene (lwd 2 below 100 DE genes)."""
    import matplotlib.pyplot as plt
    import numpy as np

    covariate = np.random.default_rng(1).permutation(40) / 40
    index_de = np.arange(0, 40, 3)
    plt.close("all")
    p = pylimma.goana_trend(index_de, covariate, plot=True, xlab="x lab", ylab="y lab", main="title")
    ax = plt.gcf().axes[0]
    order = np.argsort(covariate, kind="stable")
    line = ax.lines[0]
    np.testing.assert_array_equal(line.get_xdata(), np.arange(1, 41))
    np.testing.assert_allclose(line.get_ydata(), p[order])
    assert line.get_linewidth() == 2
    top = np.ceil(p.max() * 40) / 40
    assert ax.get_ylim() == (0, top)
    ticks = ax.collections[0].get_segments()
    de_ranks = np.flatnonzero(np.isin(order, index_de)) + 1
    np.testing.assert_allclose([seg[0] for seg in ticks], np.column_stack([de_ranks, np.zeros(14)]))
    np.testing.assert_allclose([seg[1] for seg in ticks], np.column_stack([de_ranks, np.full(14, top / 9)]))
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == ("x lab", "y lab", "title")
    plt.close("all")


@pytest.mark.parametrize("fn_name", ["goana", "kegga"])
def test_gene_ids_compare_as_r_character(fn_name):
    """R compares as.character() IDs, so 1.0 and 1 are the same gene
    (generate_all_fixtures.R); str() made them "1.0" and "1"."""
    fn = getattr(pylimma, fn_name)
    gp = pd.DataFrame({"gene": [1, 2, 3, 4, 5], "path": ["p1", "p1", "p1", "p2", "p2"]})
    by_number = fn([1.0, 2.0], gp, universe=[1.0, 2.0, 3.0, 4.0, 5.0])
    by_text = fn(["1", "2"], gp.astype({"gene": str}), universe=["1", "2", "3", "4", "5"])
    pd.testing.assert_frame_equal(by_number, by_text)
    assert by_number.loc["p1", "de"] == 2


@pytest.mark.parametrize("fn_name", ["goana", "kegga"])
def test_missing_ids_are_dropped(fn_name):
    """R drops is.na() IDs from the universe and gene.pathway; a pandas NA
    became the gene "<NA>"."""
    fn = getattr(pylimma, fn_name)
    gp, _, universe = _kegga_inputs()
    gp_with_na = pd.concat([gp, pd.DataFrame({"gene_id": [pd.NA], "path_id": ["path:hsa00010"]})], ignore_index=True)
    with_na = fn(["gene001"], gp_with_na, universe=pd.array(universe + [None], dtype="string"))
    pd.testing.assert_frame_equal(with_na, fn(["gene001"], gp, universe=universe))


def test_fit_missing_gene_ids_are_dropped():
    """R: genes with an NA ID are dropped before selecting DE genes."""
    gp, _, _ = _kegga_inputs()
    fit, genes = _two_group_fit()
    with_na = pd.array(genes[:-1] + [None], dtype="string")
    expected = pylimma.kegga(fit[:-1, :], gp, geneid=genes[:-1])
    pd.testing.assert_frame_equal(pylimma.kegga(fit, gp, geneid=with_na), expected)

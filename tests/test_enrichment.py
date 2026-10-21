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

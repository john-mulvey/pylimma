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

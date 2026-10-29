"""
R parity tests for wsva (weighted surrogate variable analysis) and
remove_batch_effect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# ----------------------------------------------------------------------------
# wsva
# ----------------------------------------------------------------------------


def test_wsva_unweighted_rparity():
    from pylimma import wsva

    E = pd.read_csv(FIXTURES / "R_wsva_input.csv").values
    design = pd.read_csv(FIXTURES / "R_twogroup_design.csv").values
    expected = pd.read_csv(FIXTURES / "R_wsva_unweighted.csv").values

    sv = wsva(E, design, n_sv=2, weight_by_sd=False)
    # SVD eigenvectors are sign-ambiguous; compare columnwise with sign
    # flipping.
    for j in range(sv.shape[1]):
        a = sv[:, j]
        b = expected[:, j]
        if np.dot(a, b) < 0:
            a = -a
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-12)


def test_wsva_design_checks_follow_r():
    """R's .lmEffects stops for a missing design, also for an EList with a
    design (R's as.matrix(y) drops it), and one with the wrong number of
    rows (fixture script)."""
    from pylimma import EList, wsva

    y = np.random.default_rng(5).normal(size=(20, 6))
    design = np.column_stack([np.ones(6), np.repeat([0.0, 1.0], 3)])
    for data in (y, EList({"E": y, "design": design})):
        with pytest.raises(ValueError, match="design matrix not specified"):
            wsva(data, None)
    with pytest.raises(ValueError, match="row dimension of design matrix must match"):
        wsva(y, np.column_stack([np.ones(5), np.arange(5.0)]))


def test_wsva_weighted_rparity():
    from pylimma import wsva

    E = pd.read_csv(FIXTURES / "R_wsva_input.csv").values
    design = pd.read_csv(FIXTURES / "R_twogroup_design.csv").values
    expected = pd.read_csv(FIXTURES / "R_wsva_weighted.csv").values

    sv = wsva(E, design, n_sv=2, weight_by_sd=True)
    for j in range(sv.shape[1]):
        a = sv[:, j]
        b = expected[:, j]
        if np.dot(a, b) < 0:
            a = -a
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize(
    "name, kwargs",
    [("batch", {"batch": ["a", "a", "a"]}), ("batch2", {"batch": [1, 2, 1], "batch2": ["x", "x", "x"]})],
)
def test_single_level_batch_raises(name, kwargs):
    """R's contr.sum stops on a one-level factor (generate_all_fixtures.R)."""
    from pylimma import remove_batch_effect

    y = np.random.default_rng(1).normal(size=(5, 3))
    with pytest.raises(
        ValueError, match=f"^{name} has a single level: not enough degrees of freedom to define contrasts$"
    ):
        remove_batch_effect(y, **kwargs)


@pytest.mark.parametrize("weight_by_sd", [False, True])
def test_wsva_rank_deficient_design_raises(weight_by_sd):
    """R's .lmEffects stops: "design must be full column rank"
    (generate_all_fixtures.R)."""
    from pylimma import wsva

    y = np.random.default_rng(20261110).normal(size=(10, 8))
    g1 = np.tile([0.0, 1.0], 4)
    g2 = np.tile([0.0, 0.0, 1.0, 1.0], 2)
    design = np.column_stack([np.ones(8), g1, g2, g1 + g2])
    with pytest.raises(ValueError, match="design must be full column rank"):
        wsva(y, design, weight_by_sd=weight_by_sd)


@pytest.mark.parametrize(
    "batch",
    [
        ["a", None, "a"],
        ["a", None, "b"],
        ["a", np.nan, "b"],
        pd.Categorical(["a", np.nan, "b"]),
        pd.array(["a", pd.NA, "b"]),
    ],
)
def test_missing_batch_label_raises(batch):
    """R's as.factor leaves NA out of the levels and R then stops
    (generate_all_fixtures.R); pylimma raises a clear error. NaN among
    strings is pandas' missing value (R's NA), not a level."""
    from pylimma import remove_batch_effect

    y = pd.read_csv(FIXTURES / "R_rbm_y.csv").to_numpy()
    with pytest.raises(ValueError, match="batch contains missing labels"):
        remove_batch_effect(y, batch=batch)


def test_numeric_nan_batch_is_a_level():
    """As in R, a numeric NaN is a level of its own ("NaN")."""
    from pylimma import remove_batch_effect

    y = pd.read_csv(FIXTURES / "R_rbm_y.csv").to_numpy()
    with pytest.warns(UserWarning, match="design matrix of interest not specified"):
        corrected = remove_batch_effect(y, batch=[1.0, np.nan, 1.0])
    np.testing.assert_allclose(
        corrected, pd.read_csv(FIXTURES / "R_rbm_nan_level.csv").to_numpy(), rtol=1e-6, atol=1e-12
    )


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_wsva_missing_or_infinite_values_raise(value):
    """R's .lmEffects: "All y values must be finite and non-NA" for NA; R
    stops on Inf in qr() (fixture script)."""
    from pylimma import wsva

    y = np.random.default_rng(1).normal(size=(10, 8))
    y[1, 2] = value
    with pytest.raises(ValueError, match="^All y values must be finite and non-NA$"):
        wsva(y, np.column_stack([np.ones(8), np.tile([0.0, 1.0], 4)]))


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("batch", {"batch": list("abababa")}),
        ("batch2", {"batch": list("abababab"), "batch2": list("xyxyxyx")}),
        ("covariates", {"covariates": np.arange(7.0)}),
    ],
)
def test_wrong_length_batch_raises(name, kwargs):
    """R stops in cbind ("number of rows of matrices must match")."""
    from pylimma import remove_batch_effect

    y = np.random.default_rng(1).normal(size=(10, 8))
    with pytest.raises(ValueError, match=f"^{name} has 7 rows/values, but there are 8 samples$"):
        remove_batch_effect(y, **kwargs)

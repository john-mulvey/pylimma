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

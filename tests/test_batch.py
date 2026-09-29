"""
R parity tests for wsva (weighted surrogate variable analysis).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9)


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
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9)

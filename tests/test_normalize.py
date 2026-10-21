"""Error and interface cases for pylimma.normalize (R behaviour for the same
cases is recorded in tests/fixtures/generate_all_fixtures.R)."""

import numpy as np
import pytest

from pylimma import normalize_between_arrays, normalize_vsn


@pytest.fixture
def intensities():
    return np.exp(np.random.default_rng(0).normal(6, 1, size=(100, 4)))


class TestNormalizeVSNInput:
    def test_too_few_rows_raise(self, intensities):
        for n_rows in (30, 0):
            with pytest.raises(ValueError, match=f"number of rows in the data matrix, {n_rows},"):
                normalize_vsn(intensities[:n_rows])

    def test_min_data_points_per_stratum_lowers_the_limit(self, intensities):
        assert normalize_vsn(intensities[:30], min_data_points_per_stratum=20).shape == (30, 4)

    def test_lts_quantile_outside_half_to_one_raises(self, intensities):
        for lts_quantile in (0.3, 0.0, 1.5, -0.1, np.nan):
            with pytest.raises(ValueError, match="lts_quantile must be a number between 0.5 and 1"):
                normalize_vsn(intensities, lts_quantile=lts_quantile)
        assert normalize_vsn(intensities, lts_quantile=0.5).shape == (100, 4)

    def test_optimiser_settings_out_of_range_raise(self, intensities):
        cases = [
            ("factr", -1.0, "factr must be a number of 0 or more"),
            ("pgtol", np.nan, "pgtol must be a number of 0 or more"),
            ("maxit", 0, "maxit must be an integer between 1"),
            ("cvg_niter", 0, "cvg_niter must be an integer between 1"),
            ("cvg_niter", 2.5, "cvg_niter must be an integer between 1"),
        ]
        for name, value, message in cases:
            with pytest.raises(ValueError, match=message):
                normalize_vsn(intensities, **{name: value})

    def test_likelihood_that_is_not_finite_raises(self, intensities):
        # R: "L-BFGS-B needs finite values of 'fn'"
        infinite = intensities.copy()
        infinite[0, 0] = np.inf
        for x in (infinite, -infinite, np.zeros((100, 4)), np.full((100, 4), 5.0)):
            with pytest.raises(ValueError, match="needs finite values"):
                normalize_vsn(x)

    def test_all_missing_matrix_raises(self):
        # R fails with "Elements of argument 'Sstrat' must be in ascending order"
        with pytest.warns(UserWarning, match="100 rows were removed"):
            with pytest.raises(ValueError, match="no row of x has a non-missing value"):
                normalize_vsn(np.full((100, 4), np.nan))


def test_normalize_between_arrays_vsn_points_to_normalize_vsn(intensities):
    with pytest.raises(ValueError, match="Please use normalize_vsn instead"):
        normalize_between_arrays(intensities, method="vsn")

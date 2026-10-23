"""Error and interface cases for pylimma.normalize (R behaviour for the same
cases is recorded in tests/fixtures/generate_all_fixtures.R)."""

import numpy as np
import pytest

from pylimma import normalize_between_arrays, normalize_cyclic_loess, normalize_vsn


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


class TestNormalizeCyclicLoessInput:
    def test_infinite_value_warning_points_at_the_caller(self, intensities):
        infinite = intensities.copy()
        infinite[0, 0] = np.inf
        with pytest.warns(UserWarning, match="infinite values in 1 row") as record:
            normalize_cyclic_loess(infinite)
        assert record[0].filename == __file__

    def test_iterations_below_one_raise(self, intensities):
        # R: for (k in 1:iterations) runs two passes for 0 (fixture script)
        for iterations in (0, -1, 0.5, np.nan, np.inf):
            with pytest.raises(ValueError, match="iterations must be a finite number of 1 or more"):
                normalize_cyclic_loess(intensities, iterations=iterations)
        # as R's 1:iterations, a fraction runs its whole part
        np.testing.assert_array_equal(
            normalize_cyclic_loess(intensities, iterations=2.5),
            normalize_cyclic_loess(intensities, iterations=2),
        )

    def test_too_few_columns_raise(self, intensities):
        # R fails with "subscript out of bounds"; "fast" works on one column
        for method in ("pairs", "affy"):
            with pytest.raises(ValueError, match="needs at least 2 column"):
                normalize_cyclic_loess(intensities[:, :1], method=method)
        with pytest.raises(ValueError, match="needs at least 1 column"):
            normalize_cyclic_loess(intensities[:, :0])
        assert normalize_cyclic_loess(intensities[:, :1]).shape == (100, 1)

    def test_weights_not_one_per_row_raise(self, intensities):
        # R: "y and weights have different lengths" from loessFit, which
        # accepts any weights with one value per row, such as an n x 1 matrix
        for weights in (np.ones((100, 4)), np.ones(99)):
            with pytest.raises(ValueError, match="y and weights have different lengths"):
                normalize_cyclic_loess(intensities, weights=weights)
        weights = np.random.default_rng(1).uniform(size=100)
        np.testing.assert_array_equal(
            normalize_cyclic_loess(intensities, weights=weights[:, None]),
            normalize_cyclic_loess(intensities, weights=weights),
        )


def test_normalize_between_arrays_vsn_points_to_normalize_vsn(intensities):
    with pytest.raises(ValueError, match="Please use normalize_vsn instead"):
        normalize_between_arrays(intensities, method="vsn")

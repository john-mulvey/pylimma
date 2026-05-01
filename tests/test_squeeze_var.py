"""Tests for pylimma squeeze_var module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylimma.squeeze_var import _squeeze_var_core, fit_f_dist, fit_f_dist_unequal_df1, squeeze_var

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestFitFDist:
    """Tests for fit_f_dist function."""

    def test_r_parity(self):
        """Test fit_f_dist matches R limma's fitFDist."""
        var_input = pd.read_csv(FIXTURES_DIR / "squeeze_var_input.csv")
        ref = pd.read_csv(FIXTURES_DIR / "fit_f_dist_output.csv")

        sample_var = var_input["sample_var"].values
        result = fit_f_dist(sample_var, df1=5)

        np.testing.assert_allclose(result["scale"], ref["fit_scale"].iloc[0], rtol=1e-5)
        np.testing.assert_allclose(result["df2"], ref["fit_df2"].iloc[0], rtol=1e-5)

    def test_empty_input(self):
        """Test that empty input returns NaN."""
        result = fit_f_dist(np.array([]), df1=5)
        assert np.isnan(result["scale"])
        assert np.isnan(result["df2"])

    def test_single_value(self):
        """Test that single value returns that value with df2=0."""
        result = fit_f_dist(np.array([0.5]), df1=5)
        assert result["scale"] == 0.5
        assert result["df2"] == 0.0

    def test_constant_variance(self):
        """Test that constant variances give infinite df2."""
        var = np.full(100, 0.5)
        result = fit_f_dist(var, df1=5)
        # When variance of log(var) is ~0, df2 should be very large or infinite
        assert result["df2"] > 100 or np.isinf(result["df2"])

    def test_zero_variance_warning(self):
        """Test that zero variances produce a warning."""
        var = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        with pytest.warns(UserWarning, match="Zero sample variances"):
            fit_f_dist(var, df1=5)


class TestSqueezeVarCore:
    """Tests for _squeeze_var_core function."""

    def test_shrinkage_formula(self):
        """Test the basic shrinkage formula."""
        var = np.array([0.1, 0.5, 1.0, 2.0])
        df = 5.0
        var_prior = 0.5
        df_prior = 4.0

        result = _squeeze_var_core(var, df, var_prior, df_prior)
        expected = (df * var + df_prior * var_prior) / (df + df_prior)

        np.testing.assert_allclose(result, expected)

    def test_infinite_df_prior(self):
        """Test that infinite df_prior returns var_prior."""
        var = np.array([0.1, 0.5, 1.0, 2.0])
        df = 5.0
        var_prior = 0.5
        df_prior = np.inf

        result = _squeeze_var_core(var, df, var_prior, df_prior)
        expected = np.full_like(var, var_prior)

        np.testing.assert_allclose(result, expected)

    def test_zero_df_prior(self):
        """Test that zero df_prior returns original variance."""
        var = np.array([0.1, 0.5, 1.0, 2.0])
        df = 5.0
        var_prior = 0.5
        df_prior = 0.0

        result = _squeeze_var_core(var, df, var_prior, df_prior)

        np.testing.assert_allclose(result, var)


class TestSqueezeVar:
    """Tests for squeeze_var function."""

    def test_r_parity(self):
        """Test squeeze_var matches R limma's squeezeVar."""
        var_input = pd.read_csv(FIXTURES_DIR / "squeeze_var_input.csv")
        ref = pd.read_csv(FIXTURES_DIR / "squeeze_var_output.csv")

        sample_var = var_input["sample_var"].values
        result = squeeze_var(sample_var, df=5)

        np.testing.assert_allclose(result["var_prior"], ref["var_prior"].iloc[0], rtol=1e-5)
        np.testing.assert_allclose(result["df_prior"], ref["df_prior"].iloc[0], rtol=1e-5)
        np.testing.assert_allclose(result["var_post"], ref["var_post"].values, rtol=1e-5)

    def test_empty_input(self):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="var is empty"):
            squeeze_var(np.array([]), df=5)

    def test_small_input(self):
        """Test that small input returns original variances."""
        var = np.array([0.5, 0.6])
        result = squeeze_var(var, df=5)

        np.testing.assert_array_equal(result["var_post"], var)
        assert result["df_prior"] == 0.0

    def test_shrinkage_towards_prior(self):
        """Test that posterior variances are between sample and prior."""
        np.random.seed(123)
        var = np.random.exponential(scale=0.5, size=50)
        df = 5

        result = squeeze_var(var, df=df)

        # Posterior should be between min and max of sample variance and prior
        lower = np.minimum(var, result["var_prior"])
        upper = np.maximum(var, result["var_prior"])

        assert np.all(result["var_post"] >= lower - 1e-10)
        assert np.all(result["var_post"] <= upper + 1e-10)


class TestCovariateSupport:
    """Tests for covariate (trend) support in fit_f_dist and squeeze_var."""

    def test_fit_f_dist_with_covariate_returns_array(self):
        """Test that fit_f_dist with covariate returns array of scales."""
        np.random.seed(42)
        n = 100
        covariate = np.linspace(5, 15, n)
        # Variance increases with mean (common in RNA-seq)
        var = 0.1 + 0.05 * covariate + np.random.exponential(0.1, n)

        result = fit_f_dist(var, df1=5, covariate=covariate)

        # Scale should be an array, not scalar
        assert isinstance(result["scale"], np.ndarray)
        assert len(result["scale"]) == n
        # df2 should still be scalar
        assert np.isscalar(result["df2"]) or result["df2"].ndim == 0

    def test_fit_f_dist_covariate_captures_trend(self):
        """Test that covariate fit captures the variance-mean trend."""
        np.random.seed(42)
        n = 200
        covariate = np.linspace(5, 15, n)
        # Strong linear trend in variance
        true_scale = 0.2 + 0.1 * covariate
        var = true_scale * np.random.exponential(1.0, n)

        result = fit_f_dist(var, df1=5, covariate=covariate)

        # Estimated scale should be correlated with true trend
        correlation = np.corrcoef(result["scale"], true_scale)[0, 1]
        assert correlation > 0.8

    def test_fit_f_dist_covariate_nan_raises(self):
        """Test that NaN in covariate raises error."""
        var = np.array([0.1, 0.2, 0.3, 0.4])
        covariate = np.array([1.0, np.nan, 3.0, 4.0])

        with pytest.raises(ValueError, match="NA covariate"):
            fit_f_dist(var, df1=5, covariate=covariate)

    def test_fit_f_dist_covariate_length_mismatch(self):
        """Test that mismatched lengths raise error."""
        var = np.array([0.1, 0.2, 0.3, 0.4])
        covariate = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="same length"):
            fit_f_dist(var, df1=5, covariate=covariate)

    def test_squeeze_var_with_covariate(self):
        """Test squeeze_var with covariate support."""
        np.random.seed(42)
        n = 100
        covariate = np.linspace(5, 15, n)
        var = 0.1 + 0.05 * covariate + np.random.exponential(0.1, n)

        result = squeeze_var(var, df=5, covariate=covariate)

        # var_prior should be array when covariate provided
        assert isinstance(result["var_prior"], np.ndarray)
        assert len(result["var_prior"]) == n
        # var_post should be same length
        assert len(result["var_post"]) == n
        # Shrinkage should still occur
        assert not np.array_equal(result["var_post"], var)

    def test_squeeze_var_covariate_preserves_trend(self):
        """Test that covariate-based shrinkage preserves the variance trend."""
        np.random.seed(42)
        n = 200
        covariate = np.linspace(5, 15, n)
        true_scale = 0.2 + 0.1 * covariate
        var = true_scale * np.random.exponential(1.0, n)

        result = squeeze_var(var, df=5, covariate=covariate)

        # Posterior variance should still show the trend
        # Compare high vs low covariate groups
        low_mean = np.mean(result["var_post"][covariate < 8])
        high_mean = np.mean(result["var_post"][covariate > 12])
        assert high_mean > low_mean

    def test_few_unique_covariate_values(self):
        """Test handling of covariate with few unique values."""
        np.random.seed(42)
        var = np.random.exponential(0.5, 20)
        # Only 2 unique values - should fall back gracefully
        covariate = np.array([1.0] * 10 + [2.0] * 10)

        # Should not raise, may fall back to no covariate
        result = fit_f_dist(var, df1=5, covariate=covariate)
        assert not np.isnan(result["df2"])


class TestFitFDistUnequalDF1:
    """Tests for fit_f_dist_unequal_df1 function."""

    def test_constant_df1_similar_to_fit_f_dist(self):
        """Test that constant df1 gives similar results to fit_f_dist."""
        np.random.seed(42)
        n = 100
        df1_val = 5.0
        df1 = np.full(n, df1_val)
        var = np.random.exponential(0.5, n)

        result_unequal = fit_f_dist_unequal_df1(var, df1=df1)
        result_standard = fit_f_dist(var, df1=df1_val)

        # Should be reasonably close (not identical due to different methods)
        np.testing.assert_allclose(result_unequal["scale"], result_standard["scale"], rtol=0.3)
        # df2 can differ more due to method differences
        assert result_unequal["df2"] > 0
        assert result_standard["df2"] > 0

    def test_handles_small_df1(self):
        """Test handling of small df1 values."""
        np.random.seed(42)
        n = 50
        # Include some very small df1 values
        df1 = np.concatenate([np.full(10, 0.005), np.random.uniform(2, 10, 40)])
        var = np.random.exponential(0.5, n)

        # Should not raise
        result = fit_f_dist_unequal_df1(var, df1=df1)
        assert not np.isnan(result["df2"])

    def test_handles_na_in_x(self):
        """Test handling of NaN in x."""
        np.random.seed(42)
        n = 50
        df1 = np.random.uniform(2, 10, n)
        var = np.random.exponential(0.5, n)
        var[5] = np.nan
        var[10] = np.nan

        result = fit_f_dist_unequal_df1(var, df1=df1)
        assert not np.isnan(result["df2"])

    def test_with_covariate(self):
        """Test with covariate for trend fitting."""
        np.random.seed(42)
        n = 100
        df1 = np.random.uniform(2, 10, n)
        covariate = np.linspace(5, 15, n)
        # Variance trend with mean
        var = (0.1 + 0.05 * covariate) * np.random.exponential(1.0, n)

        result = fit_f_dist_unequal_df1(var, df1=df1, covariate=covariate)

        # Scale should be array when covariate provided
        assert isinstance(result["scale"], np.ndarray)
        assert len(result["scale"]) == n

    def test_robust_mode(self):
        """Test robust mode with outlier detection."""
        np.random.seed(42)
        n = 100
        df1 = np.random.uniform(2, 10, n)
        var = np.random.exponential(0.5, n)
        # Add some outliers
        var[:5] = var[:5] * 20

        result = fit_f_dist_unequal_df1(var, df1=df1, robust=True)

        assert "scale" in result
        assert "df2" in result
        # May or may not have df2_shrunk depending on outlier detection
        if "df2_shrunk" in result:
            assert len(result["df2_shrunk"]) == n

    def test_with_prior_weights(self):
        """Test with prior weights."""
        np.random.seed(42)
        n = 50
        df1 = np.random.uniform(2, 10, n)
        var = np.random.exponential(0.5, n)
        # Downweight first 10 observations
        prior_weights = np.ones(n)
        prior_weights[:10] = 0.1

        result = fit_f_dist_unequal_df1(var, df1=df1, prior_weights=prior_weights)
        assert not np.isnan(result["df2"])

    def test_length_mismatch_raises(self):
        """Test that mismatched lengths raise error."""
        var = np.array([0.1, 0.2, 0.3, 0.4])
        df1 = np.array([5.0, 6.0, 7.0])

        with pytest.raises(ValueError, match="different lengths"):
            fit_f_dist_unequal_df1(var, df1=df1)

    def test_na_df1_raises(self):
        """Test that NaN in df1 raises error."""
        var = np.array([0.1, 0.2, 0.3, 0.4])
        df1 = np.array([5.0, np.nan, 7.0, 8.0])

        with pytest.raises(ValueError, match="NA df1"):
            fit_f_dist_unequal_df1(var, df1=df1)

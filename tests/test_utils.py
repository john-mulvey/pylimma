"""Tests for pylimma utility functions."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from pylimma.utils import (
    trigamma_inverse, logmdigamma, p_adjust,
    qqt, qqf, choose_lowess_span, loess_fit
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestTrigammaInverse:
    """Tests for trigamma_inverse function."""

    def test_r_parity(self):
        """Test trigamma_inverse matches R limma's trigammaInverse."""
        ref = pd.read_csv(FIXTURES_DIR / "trigamma_inverse.csv")
        x = ref["x"].values
        expected = ref["trigamma_inverse"].values

        result = trigamma_inverse(x)

        np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-10)

    def test_scalar_input(self):
        """Test that scalar input returns scalar output."""
        result = trigamma_inverse(1.0)
        assert isinstance(result, float)
        assert np.isclose(result, 1.42625512, rtol=1e-6)

    def test_negative_input_warns(self):
        """Test that negative input produces NaN with warning."""
        with pytest.warns(RuntimeWarning, match="NaNs produced"):
            result = trigamma_inverse(-1.0)
        assert np.isnan(result)

    def test_nan_propagation(self):
        """Test that NaN input propagates."""
        result = trigamma_inverse(np.array([1.0, np.nan, 2.0]))
        assert np.isnan(result[1])
        assert not np.isnan(result[0])
        assert not np.isnan(result[2])

    def test_inverse_property(self):
        """Test that trigamma(trigamma_inverse(x)) ~ x."""
        from scipy.special import polygamma

        x = np.array([0.5, 1.0, 2.0, 5.0])
        y = trigamma_inverse(x)
        x_recovered = polygamma(1, y)

        np.testing.assert_allclose(x_recovered, x, rtol=1e-7)


class TestLogmdigamma:
    """Tests for logmdigamma function."""

    def test_definition(self):
        """Test logmdigamma equals log(x) - digamma(x)."""
        from scipy.special import digamma

        x = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        expected = np.log(x) - digamma(x)
        result = logmdigamma(x)

        np.testing.assert_allclose(result, expected, rtol=1e-14)

class TestPAdjust:
    """Tests for p_adjust function."""

    def test_bh_correction(self):
        """Test Benjamini-Hochberg correction."""
        p = np.array([0.01, 0.04, 0.03, 0.20])
        adj = p_adjust(p, method="BH")

        # Adjusted p-values should be >= original
        assert np.all(adj >= p)
        # Should maintain order for these values
        assert adj[0] <= adj[2] <= adj[1] <= adj[3]

    def test_no_correction(self):
        """Test that method='none' returns original p-values."""
        p = np.array([0.01, 0.04, 0.03, 0.20])
        adj = p_adjust(p, method="none")
        np.testing.assert_array_equal(adj, p)

    def test_nan_handling(self):
        """Test that NaN p-values are preserved."""
        p = np.array([0.01, np.nan, 0.03])
        adj = p_adjust(p, method="BH")
        assert np.isnan(adj[1])
        assert not np.isnan(adj[0])
        assert not np.isnan(adj[2])


class TestQQT:
    """Tests for qqt function."""

    def test_basic_output(self):
        """Test qqt returns expected structure."""
        np.random.seed(42)
        y = np.random.standard_t(df=10, size=50)

        result = qqt(y, df=10, plot_it=False)

        assert "x" in result
        assert "y" in result
        assert len(result["x"]) == 50
        assert len(result["y"]) == 50

    def test_normal_distribution(self):
        """Test qqt with df=inf (normal distribution)."""
        np.random.seed(42)
        y = np.random.randn(100)

        result = qqt(y, df=np.inf, plot_it=False)

        assert len(result["x"]) == 100
        # Theoretical quantiles should be from normal distribution
        from scipy import stats
        expected_x = stats.norm.ppf((np.arange(1, 101) - 0.5) / 100)
        # Check that x values span similar range
        assert np.isclose(np.min(result["x"]), np.min(expected_x), rtol=0.1)
        assert np.isclose(np.max(result["x"]), np.max(expected_x), rtol=0.1)

    def test_nan_handling(self):
        """Test that NaN values are removed."""
        y = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        result = qqt(y, df=10, plot_it=False)
        assert len(result["x"]) == 4
        assert len(result["y"]) == 4

    def test_empty_input_raises(self):
        """Test that empty input raises error."""
        with pytest.raises(ValueError, match="empty"):
            qqt(np.array([]), plot_it=False)


class TestQQF:
    """Tests for qqf function."""

    def test_basic_output(self):
        """Test qqf returns expected structure."""
        np.random.seed(42)
        from scipy import stats
        y = stats.f.rvs(dfn=5, dfd=20, size=50)

        result = qqf(y, df1=5, df2=20, plot_it=False)

        assert "x" in result
        assert "y" in result
        assert len(result["x"]) == 50
        assert len(result["y"]) == 50

    def test_nan_handling(self):
        """Test that NaN values are removed."""
        y = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        result = qqf(y, df1=5, df2=20, plot_it=False)
        assert len(result["x"]) == 4


class TestChooseLowessSpan:
    """Tests for choose_lowess_span function."""

    def test_small_n(self):
        """Test span is 1 for very small n."""
        span = choose_lowess_span(n=10, small_n=50)
        assert span == 1.0

    def test_large_n(self):
        """Test span approaches min_span for large n."""
        span = choose_lowess_span(n=10000, small_n=50, min_span=0.3)
        # With power=1/3, (50/10000)^(1/3) = 0.171, so span = 0.3 + 0.7*0.171 = 0.42
        expected = 0.3 + (1 - 0.3) * (50 / 10000) ** (1/3)
        assert np.isclose(span, expected, rtol=1e-10)

    def test_default_values(self):
        """Test default parameter values."""
        span = choose_lowess_span(n=1000)
        assert 0 < span <= 1

    def test_monotonic_in_n(self):
        """Test that span decreases as n increases."""
        spans = [choose_lowess_span(n=n) for n in [50, 100, 500, 1000, 5000]]
        for i in range(len(spans) - 1):
            assert spans[i] >= spans[i + 1]


class TestLoessFit:
    """Tests for loess_fit function."""

    def test_basic_fit(self):
        """Test basic LOWESS fit."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.randn(100) * 0.2

        result = loess_fit(y, x, span=0.3)

        assert "fitted" in result
        assert "residuals" in result
        assert len(result["fitted"]) == 100
        assert len(result["residuals"]) == 100
        # Residuals should be y - fitted
        np.testing.assert_allclose(
            result["residuals"],
            y - result["fitted"],
            rtol=1e-10
        )

    def test_weighted_r_parity(self):
        """Test weighted LOWESS matches R limma's weightedLowess."""
        # Load fixture generated from R
        ref = pd.read_csv(FIXTURES_DIR / "weighted_lowess.csv")

        result = loess_fit(
            ref["y"].values,
            ref["x"].values,
            weights=ref["weights"].values,
            span=0.3,
            iterations=4
        )

        # Should match R within numerical precision
        np.testing.assert_allclose(
            result["fitted"], ref["fitted"].values, rtol=1e-6
        )

    def test_nan_handling(self):
        """Test that NaN values are handled."""
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([1.0, np.nan, 3.0, 4.0, 5.0])

        result = loess_fit(y, x, span=0.5)

        assert np.isnan(result["fitted"][1])
        assert np.isnan(result["residuals"][1])
        assert not np.isnan(result["fitted"][0])

    def test_different_lengths_raises(self):
        """Test that different length inputs raise error."""
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3, 4])

        with pytest.raises(ValueError, match="different lengths"):
            loess_fit(y, x)

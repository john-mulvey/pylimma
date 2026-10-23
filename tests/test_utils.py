"""Tests for pylimma utility functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylimma.utils import (
    choose_lowess_span,
    fit_gamma_intercept,
    loess_fit,
    logmdigamma,
    p_adjust,
    qqf,
    qqt,
    trigamma_inverse,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestTrigammaInverse:
    """Tests for trigamma_inverse function."""

    def test_scalar_input(self):
        """Test that scalar input returns scalar output."""
        result = trigamma_inverse(1.0)
        assert isinstance(result, float)
        assert np.isclose(result, 1.42625512, rtol=1e-6, atol=1e-12)

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
        assert np.isclose(np.min(result["x"]), np.min(expected_x), rtol=1e-6, atol=1e-12)
        assert np.isclose(np.max(result["x"]), np.max(expected_x), rtol=1e-6, atol=1e-12)

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

    def test_draws_what_r_draws(self):
        """R's plot(x, y, main, xlab, ylab, ylim): the points only (no
        reference line), with R's title also for df = Inf."""
        plt = pytest.importorskip("matplotlib.pyplot")
        y = np.array([1.5, -0.3, 0.8, 2.2, -1.1])
        for fun, kwargs, title in (
            (qqt, {}, "Student's t Q-Q Plot"),
            (qqf, {"df1": 3, "df2": 10}, "F Distribution Q-Q Plot"),
        ):
            out = fun(y, **kwargs)
            ax = plt.gca()
            assert len(ax.lines) == 1
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), out["x"])
            assert ax.get_title() == title
            assert ax.get_xlabel() == "Theoretical Quantiles"
            assert ax.get_ylabel() == "Sample Quantiles"
            plt.close("all")
        qqt(y, main="m", xlab="a", ylab="b", ylim=(-3, 3))
        ax = plt.gca()
        assert (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == ("m", "a", "b")
        assert ax.get_ylim() == (-3, 3)
        plt.close("all")


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


class TestFitGammaInterceptValidation:
    """Inputs where R fails or behaves accidentally get a clear error
    (docs/validation/known_differences.rst)."""

    @pytest.mark.parametrize(
        "y, offset",
        [([1.0, np.nan, 2.0], [0.5, 1.0, 1.5]), ([1.0, 2.0, 3.0], [0.5, np.nan, 1.5])],
    )
    def test_missing_values_raise(self, y, offset):
        # R stops with "missing value where TRUE/FALSE needed"
        with pytest.raises(ValueError, match="y and offset must not contain missing values"):
            fit_gamma_intercept(y, offset=offset)

    def test_empty_y_gives_nan(self):
        # R: mean(numeric(0)) - offset through the constant-offset shortcut
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert np.isnan(fit_gamma_intercept([], offset=1.0))

    def test_undefined_newton_step_raises(self):
        # 0 / 0 where the offset and y are both zero; R stops with
        # "missing value where TRUE/FALSE needed"
        with pytest.raises(ValueError, match="Newton step is undefined"):
            fit_gamma_intercept([0.0, 1.0], offset=[0.0, 1.0])

    def test_offset_length_mismatch_raises(self):
        # R silently recycles offset
        with pytest.raises(ValueError, match="offset must be a scalar or have the same length as y"):
            fit_gamma_intercept([1.0, 2.0, 3.0, 4.0], offset=[0.5, 1.0])


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
        expected = 0.3 + (1 - 0.3) * (50 / 10000) ** (1 / 3)
        assert np.isclose(span, expected, rtol=1e-10, atol=1e-12)

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
        np.testing.assert_allclose(result["residuals"], y - result["fitted"], rtol=1e-10)

    def test_weighted_r_parity(self):
        """Test weighted LOWESS matches R limma's weightedLowess."""
        # Load fixture generated from R
        ref = pd.read_csv(FIXTURES_DIR / "weighted_lowess.csv")

        result = loess_fit(
            ref["y"].values, ref["x"].values, weights=ref["weights"].values, span=0.3, iterations=4
        )

        # Should match R within numerical precision
        np.testing.assert_allclose(result["fitted"], ref["fitted"].values, rtol=1e-6)

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


class TestLowessInput:
    """R's lowess() refuses f <= 0, iter < 0 and delta < 0, so loess_fit with
    iterations < 1 and voom with span = 0 fail in R too (fixture script)."""

    def test_lowess_arguments_out_of_range_raise(self):
        from pylimma.utils import _r_lowess

        x = np.arange(10.0)
        for kwargs, message in (
            ({"f": 0}, "'f' must be finite and > 0"),
            ({"f": np.nan}, "'f' must be finite and > 0"),
            ({"iter": -1}, "'iter' must be finite and >= 0"),
            ({"iter": 1e10}, "'iter' must be finite and >= 0"),
            ({"delta": -1.0}, "'delta' must be finite and > 0"),
        ):
            with pytest.raises(ValueError, match=message):
                _r_lowess(x, x, **kwargs)
        # as R's asInteger(), iter is truncated before the check
        truncated = _r_lowess(x, x**2, iter=-0.5)[1]
        np.testing.assert_array_equal(truncated, _r_lowess(x, x**2, iter=0)[1])
        # f * n overflows to inf: all points, as for any f > 1 (R likewise)
        np.testing.assert_array_equal(_r_lowess(x, x, f=1e308)[1], _r_lowess(x, x, f=1)[1])

    def test_approx_needs_two_x_values(self):
        from pylimma.utils import _r_approx_rule2

        with pytest.raises(ValueError, match="need at least two non-NA values to interpolate"):
            _r_approx_rule2(np.array([1.0, 1.0, np.nan]), np.array([1.0, 2.0, 3.0]), np.array([2.0]))

    def test_loess_fit_iterations_below_one_raise(self):
        x = np.arange(10.0)
        for iterations in (0, -1):
            with pytest.raises(ValueError, match="'iter' must be finite and >= 0"):
                loess_fit(x, x, iterations=iterations)
        # iter = iterations - 1 = -0.5 runs as 0 in R
        np.testing.assert_array_equal(
            loess_fit(x**2, x, iterations=0.5)["fitted"], loess_fit(x**2, x, iterations=1)["fitted"]
        )

    def test_voom_span_zero_raises(self):
        from pylimma import voom

        counts = np.random.default_rng(0).poisson(50, size=(30, 6))
        design = np.column_stack([np.ones(6), np.tile([0, 1], 3)])
        with pytest.raises(ValueError, match="'f' must be finite and > 0"):
            voom(counts, design, span=0, adaptive_span=False)


@pytest.mark.parametrize(
    "case,kwargs",
    [
        ("n", {"var": [1.2, 0.7, 2.5], "n": [3, 4, 5]}),
        ("df_multiplier", {"var": [1.2, 0.7], "df": [4, 9], "multiplier": [0.5, 2]}),
        ("zero_var", {"var": [0.0, 0.0, 0.0], "n": [3, 4, 5]}),
        ("na_var", {"var": [1.0, np.nan, 3.0], "n": [3, 4, 5]}),
        ("inf_df", {"var": [1.0, 2.0], "df": [np.inf, np.inf], "multiplier": [1, 1]}),
    ],
)
def test_pool_var_matches_r(case, kwargs):
    """Satterthwaite's df divided directly, as R: 0/0 is NaN, an NA
    variance NA and infinite df Inf."""
    from pylimma import pool_var

    r = pd.read_csv(FIXTURES_DIR / "R_pool_var.csv").set_index("case").loc[case]
    out = pool_var(**kwargs)
    for key in ("var", "df", "multiplier"):
        np.testing.assert_allclose(out[key], r[key], rtol=1e-10)


def test_r_as_character_matches_r():
    """R: as.character(c(1, 0.5, 0.1 + 0.2, NaN, Inf, -Inf, TRUE)); NA (None) stays missing."""
    from pylimma.utils import _r_as_character

    result = _r_as_character([1.0, 0.5, 0.1 + 0.2, np.nan, np.inf, -np.inf, True, None, "a"])
    assert list(result) == ["1", "0.5", "0.3", "NaN", "Inf", "-Inf", "TRUE", None, "a"]


def test_pool_var_reads_a_matrix_as_its_values():
    """R: as.vector(var), column by column (fixture script)."""
    from pylimma import pool_var

    by_matrix = pool_var(
        np.array([[1.2, 2.5], [0.7, 1.1]]), df=[3, 4, 5, 6], multiplier=[1, 2, 1, 2]
    )
    by_vector = pool_var([1.2, 0.7, 2.5, 1.1], df=[3, 4, 5, 6], multiplier=[1, 2, 1, 2])
    assert by_matrix == by_vector
    # also when the defaults come from a matrix n
    from_n = pool_var(np.array([[1.2, 2.5], [0.7, 1.1]]), n=np.array([[4, 6], [5, 7]]))
    assert from_n == pool_var([1.2, 0.7, 2.5, 1.1], n=[4, 5, 6, 7])

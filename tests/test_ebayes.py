"""Tests for pylimma ebayes module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylimma.ebayes import _tmixture_vector, e_bayes, treat
from pylimma.lmfit import lm_fit

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestEBayes:
    """Tests for e_bayes function."""

    def test_r_parity(self):
        """Test e_bayes matches R limma's eBayes."""
        # Load R fixtures
        expr = pd.read_csv(FIXTURES_DIR / "ebayes_expr.csv", index_col=0).values
        design = pd.read_csv(FIXTURES_DIR / "ebayes_design.csv").values
        ref_stats = pd.read_csv(FIXTURES_DIR / "ebayes_stats.csv", index_col=0)
        ref_global = pd.read_csv(FIXTURES_DIR / "ebayes_global.csv")

        # Fit in Python
        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        # Compare t-statistics (column 2 = groupB effect)
        np.testing.assert_allclose(fit["t"][:, 1], ref_stats["t"].values, rtol=1e-6)

        # Compare p-values
        np.testing.assert_allclose(fit["p_value"][:, 1], ref_stats["p_value"].values, rtol=1e-6)

        # Compare s2.post
        np.testing.assert_allclose(fit["s2_post"], ref_stats["s2_post"].values, rtol=1e-6)

        np.testing.assert_allclose(fit["lods"][:, 1], ref_stats["lods"].values, rtol=1e-6)
        np.testing.assert_allclose(fit["df_total"], ref_stats["df_total"].values, rtol=1e-6)

        # Compare hyperparameters and the F-test
        np.testing.assert_allclose(fit["s2_prior"], ref_global["s2_prior"].iloc[0], rtol=1e-6)
        np.testing.assert_allclose(fit["df_prior"], ref_global["df_prior"].iloc[0], rtol=1e-6)
        np.testing.assert_allclose(fit["F"], ref_global["F_stat"].values, rtol=1e-6)
        np.testing.assert_allclose(fit["F_p_value"], ref_global["F_p_value"].values, rtol=1e-6)

        ref_coef = pd.read_csv(FIXTURES_DIR / "ebayes_coef.csv", index_col=0).values
        np.testing.assert_allclose(fit["coefficients"], ref_coef, rtol=1e-10)

    def test_moderated_t_smaller_than_ordinary(self):
        """Test that moderated t-stats have smaller variance than ordinary."""
        np.random.seed(42)
        n_genes, n_samples = 100, 10
        expr = np.random.randn(n_genes, n_samples)
        design = np.column_stack([np.ones(n_samples), np.random.randn(n_samples)])

        fit = lm_fit(expr, design)
        ordinary_t = fit["coefficients"][:, 1] / (fit["stdev_unscaled"][:, 1] * fit["sigma"])

        fit = e_bayes(fit)
        moderated_t = fit["t"][:, 1]

        # Moderated t should have smaller variance due to shrinkage
        assert np.var(moderated_t) < np.var(ordinary_t)

    def test_anndata_input(self):
        """Test e_bayes with AnnData input."""
        pytest.importorskip("anndata")
        import anndata as ad

        np.random.seed(42)
        adata = ad.AnnData(X=np.random.randn(8, 50))
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        lm_fit(adata, design)
        result = e_bayes(adata)

        assert result is None
        assert "t" in adata.uns["pylimma"]
        assert "p_value" in adata.uns["pylimma"]
        assert "lods" in adata.uns["pylimma"]

    def test_lods_sign(self):
        """Test that lods are higher for genes with larger effects."""
        np.random.seed(42)
        n_genes, n_samples = 100, 8
        expr = np.random.randn(n_genes, n_samples)
        # Add strong effects to first 10 genes
        expr[:10, 4:] += 3

        design = np.column_stack([np.ones(n_samples), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        # DE genes should have higher lods
        de_lods = fit["lods"][:10, 1]
        null_lods = fit["lods"][10:, 1]

        assert np.mean(de_lods) > np.mean(null_lods)

    def test_requires_fit_components(self):
        """Test that e_bayes validates fit object."""
        with pytest.raises(ValueError, match="stdev_unscaled, sigma, df_residual not found in fit"):
            e_bayes({"coefficients": np.array([[1, 2]])})

    def test_trend_requires_amean(self):
        """Test that trend=True requires Amean in fit."""
        np.random.seed(42)
        expr = np.random.randn(50, 8)
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        # Remove Amean if present
        fit.pop("Amean", None)

        with pytest.raises(ValueError, match="Amean"):
            e_bayes(fit, trend=True)

    def test_trend_vs_no_trend(self):
        """Test that trend and no-trend give different results."""
        np.random.seed(42)
        n_genes, n_samples = 100, 10

        # Create data with strong variance-mean relationship
        means = np.linspace(5, 15, n_genes)
        expr = np.zeros((n_genes, n_samples))
        for i in range(n_genes):
            # Variance increases with mean
            expr[i, :] = np.random.randn(n_samples) * (0.2 + 0.1 * means[i]) + means[i]

        design = np.column_stack([np.ones(n_samples), [0] * 5 + [1] * 5])

        fit1 = lm_fit(expr.copy(), design)
        fit1["Amean"] = np.mean(expr, axis=1)
        fit1 = e_bayes(fit1, trend=False)

        fit2 = lm_fit(expr.copy(), design)
        fit2["Amean"] = np.mean(expr, axis=1)
        fit2 = e_bayes(fit2, trend=True)

        # Results should differ
        assert not np.allclose(fit1["s2_post"], fit2["s2_post"])


class TestTreat:
    """Tests for treat function (TREAT test)."""

    def test_fc_parameter(self):
        """Test that fc parameter works (converts to log2)."""
        np.random.seed(42)
        expr = np.random.randn(50, 8)
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = treat(fit, fc=2.0)  # fc=2 means lfc=1

        assert np.isclose(fit["treat_lfc"], 1.0, rtol=1e-10, atol=1e-12)

    def test_higher_threshold_gives_fewer_significant(self):
        """Test that higher lfc threshold gives fewer significant results."""
        np.random.seed(42)
        expr = np.random.randn(100, 8)
        expr[:20, 4:] += 2  # Add moderate effects
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit1 = lm_fit(expr.copy(), design)
        fit1 = treat(fit1, lfc=0.5)

        fit2 = lm_fit(expr.copy(), design)
        fit2 = treat(fit2, lfc=1.5)

        sig1 = np.sum(fit1["p_value"][:, 1] < 0.05)
        sig2 = np.sum(fit2["p_value"][:, 1] < 0.05)

        assert sig2 <= sig1

    def test_t_stat_direction(self):
        """Test that t-statistics have correct direction."""
        np.random.seed(42)
        expr = np.random.randn(50, 8)
        expr[:10, 4:] += 5  # Strong positive effect
        expr[10:20, 4:] -= 5  # Strong negative effect
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = treat(fit, lfc=1.0)

        # Up-regulated genes should have positive t
        assert np.mean(fit["t"][:10, 1]) > 0
        # Down-regulated genes should have negative t
        assert np.mean(fit["t"][10:20, 1]) < 0

    def test_anndata_input(self):
        """Test treat with AnnData input."""
        pytest.importorskip("anndata")
        import anndata as ad

        np.random.seed(42)
        adata = ad.AnnData(X=np.random.randn(8, 50))
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        lm_fit(adata, design)
        result = treat(adata, lfc=0.5)

        assert result is None
        assert "treat_lfc" in adata.uns["pylimma"]

    def test_upshot_parameter(self):
        """Test that upshot parameter produces valid p-values."""
        np.random.seed(42)
        expr = np.random.randn(50, 8)
        expr[:10, 4:] += 3  # Add effects
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit_upshot = treat(fit.copy(), lfc=0.5, upshot=True)

        # P-values should be bounded
        assert np.all(fit_upshot["p_value"] >= 0)
        assert np.all(fit_upshot["p_value"] <= 1)

        # Upshot should give slightly different p-values than standard
        fit_standard = treat(fit.copy(), lfc=0.5, upshot=False)
        # The p-values should not be identical (averaging vs boundary)
        assert not np.allclose(fit_upshot["p_value"], fit_standard["p_value"])


class TestTmixtureVector:
    """Tests for _tmixture_vector helper."""

    def test_returns_positive(self):
        """Test that estimated variance is positive."""
        np.random.seed(42)
        tstat = np.random.standard_t(df=5, size=100)
        stdev = np.ones(100)
        df = 5

        v0 = _tmixture_vector(tstat, stdev, df, proportion=0.1)

        assert v0 >= 0 or np.isnan(v0)

    def test_handles_missing(self):
        """Test that NaN values are handled."""
        tstat = np.array([1.0, np.nan, 2.0, 3.0])
        stdev = np.array([1.0, 1.0, 1.0, 1.0])
        df = 5

        # Should not raise
        v0 = _tmixture_vector(tstat, stdev, df, proportion=0.1)
        assert not np.isnan(v0) or v0 >= 0

    def test_extreme_t_statistics_no_underflow(self):
        """Test that extreme t-statistics don't cause precision loss.

        R uses log.p=TRUE throughout; Python must handle values where
        exp(logsf) would underflow to 0.
        """
        # Create t-statistics with varying df that need adjustment
        n = 100
        np.random.seed(42)
        # Include some very extreme values that would cause underflow
        tstat = np.concatenate(
            [
                np.random.standard_t(df=5, size=90),
                np.array([30, 40, 50, 60, 70, 80, 90, 100, 150, 200]),  # Extreme values
            ]
        )
        stdev = np.ones(n)
        # Use varying df to force adjustment
        df = np.concatenate([np.full(50, 3), np.full(50, 10)])

        # Should not raise and should return finite result
        v0 = _tmixture_vector(tstat, stdev, df, proportion=0.1)

        # Result should be finite (not inf due to underflow handling)
        assert np.isfinite(v0)


class TestEBayesArgumentChecks:
    """R behaviour recorded in the fixture script."""

    def _fit(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=(50, 6))
        y[:, 3:] += 1
        return lm_fit(y, np.column_stack([np.ones(6), [0.0, 0, 0, 1, 1, 1]]))

    def test_numpy_logical_trend_is_logical(self):
        fit = self._fit()
        np.testing.assert_array_equal(
            e_bayes(fit, trend=np.True_)["t"], e_bayes(fit, trend=True)["t"]
        )
        np.testing.assert_array_equal(e_bayes(fit, trend=np.False_)["t"], e_bayes(fit)["t"])

    def test_fit_without_genes_raises_r_error(self):
        fit = self._fit()[:0, :]
        for fn in (e_bayes, treat):
            with pytest.raises(ValueError, match="No finite residual standard deviations"):
                fn(fit)

    def test_treat_without_fc_or_lfc_raises(self):
        with pytest.raises(ValueError, match="fc must be a number"):
            treat(self._fit(), fc=None)

    @pytest.mark.parametrize("lim", [(0.1,), (0.1, 4.0, 10.0)])
    def test_stdev_coef_lim_of_wrong_length_raises(self, lim):
        with pytest.raises(ValueError, match="stdev_coef_lim must have length 2"):
            e_bayes(self._fit(), stdev_coef_lim=lim)

    @pytest.mark.parametrize("not_a_fit", [5, "abc", np.ones((3, 3))])
    def test_input_that_is_not_a_fit_raises(self, not_a_fit):
        for fn in (e_bayes, treat):
            with pytest.raises(TypeError, match="fit must be an MArrayLM"):
                fn(not_a_fit)

    def test_stdev_unscaled_of_another_shape_raises(self):
        fit = self._fit()
        fit["stdev_unscaled"] = fit["stdev_unscaled"][:, :1]
        for fn in (e_bayes, treat):
            with pytest.raises(ValueError, match="have different dimensions"):
                fn(fit)

    def test_vector_coefficients_are_one_column(self):
        """R's treat reads coefficients and stdev_unscaled with as.matrix
        (t is one column); eBayes takes vectors too (R's t is then a vector,
        pylimma's one column, as the other functions read it)."""
        fit = self._fit()
        vec = dict(fit, coefficients=fit["coefficients"][:, 1], stdev_unscaled=fit["stdev_unscaled"][:, 1])
        mixed = dict(fit, coefficients=fit["coefficients"][:, 1:], stdev_unscaled=fit["stdev_unscaled"][:, 1])
        one = dict(fit, coefficients=fit["coefficients"][:, 1:], stdev_unscaled=fit["stdev_unscaled"][:, 1:])
        for fn in (e_bayes, treat):
            expected = fn(one)
            for other in (vec, mixed):
                result = fn(other)
                for slot in ("coefficients", "t", "p_value"):
                    np.testing.assert_array_equal(result[slot], expected[slot])
            # decide_tests reads the slots with one shape
            from pylimma import decide_tests

            assert decide_tests(fn(vec)).shape == expected["t"].shape
            # Integer coefficients are read as numbers, not truncated
            ints = dict(one, coefficients=np.round(one["coefficients"] * 10).astype(int))
            floats = dict(ints, coefficients=ints["coefficients"].astype(float))
            np.testing.assert_array_equal(fn(ints)["t"], fn(floats)["t"])
            # A DataFrame of coefficients keeps its row names
            framed = dict(one, coefficients=pd.DataFrame(one["coefficients"], index=list("abcdefghij") * 5))
            assert fn(framed)["coefficients"].index.equals(framed["coefficients"].index)

    def test_treat_removes_lods(self):
        assert "lods" not in treat(e_bayes(self._fit()))

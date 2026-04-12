"""Tests for pylimma ebayes module."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from pylimma.ebayes import e_bayes, treat, _tmixture_vector
from pylimma.lmfit import lm_fit
from pylimma.contrasts import make_contrasts, contrasts_fit


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
        np.testing.assert_allclose(
            fit["t"][:, 1], ref_stats["t"].values, rtol=1e-5
        )

        # Compare p-values
        np.testing.assert_allclose(
            fit["p_value"][:, 1], ref_stats["p_value"].values, rtol=1e-5
        )

        # Compare s2.post
        np.testing.assert_allclose(
            fit["s2_post"], ref_stats["s2_post"].values, rtol=1e-5
        )

        # Compare hyperparameters
        np.testing.assert_allclose(
            fit["s2_prior"], ref_global["s2_prior"].iloc[0], rtol=1e-5
        )

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
        with pytest.raises(ValueError, match="must contain"):
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

    def test_trend_basic_functionality(self):
        """Test that trend=True runs without error."""
        np.random.seed(42)
        n_genes, n_samples = 100, 10
        expr = np.random.randn(n_genes, n_samples)
        # Add variance-mean relationship
        means = np.random.uniform(5, 15, n_genes)
        for i in range(n_genes):
            expr[i, :] = expr[i, :] * (0.1 + 0.05 * means[i]) + means[i]

        design = np.column_stack([np.ones(n_samples), np.random.randn(n_samples)])

        fit = lm_fit(expr, design)
        # Ensure Amean is present
        fit["Amean"] = np.mean(expr, axis=1)
        fit = e_bayes(fit, trend=True)

        assert "t" in fit
        assert "p_value" in fit
        assert "s2_prior" in fit
        # s2_prior should be array when trend=True
        assert isinstance(fit["s2_prior"], np.ndarray)

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

        design = np.column_stack([np.ones(n_samples), [0]*5 + [1]*5])

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

        assert np.isclose(fit["treat_lfc"], 1.0, rtol=1e-10)

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
        tstat = np.concatenate([
            np.random.standard_t(df=5, size=90),
            np.array([30, 40, 50, 60, 70, 80, 90, 100, 150, 200])  # Extreme values
        ])
        stdev = np.ones(n)
        # Use varying df to force adjustment
        df = np.concatenate([np.full(50, 3), np.full(50, 10)])

        # Should not raise and should return finite result
        v0 = _tmixture_vector(tstat, stdev, df, proportion=0.1)

        # Result should be finite (not inf due to underflow handling)
        assert np.isfinite(v0)

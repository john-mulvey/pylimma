"""Tests for pylimma decide_tests module."""

import numpy as np

from pylimma.contrasts import contrasts_fit, make_contrasts
from pylimma.decide_tests import classify_tests_f, decide_tests
from pylimma.ebayes import e_bayes
from pylimma.lmfit import lm_fit


class TestDecideTests:
    """Tests for decide_tests function."""

    def test_basic_functionality(self):
        """Test basic decide_tests output."""
        np.random.seed(42)
        expr = np.random.randn(50, 8)
        expr[:10, 4:] += 3  # Strong effects
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)
        result = decide_tests(fit, p_value=0.05)

        assert result.shape == (50, 2)
        assert set(np.unique(result)).issubset({-1, 0, 1})

    def test_lfc_threshold(self):
        """Test log fold-change threshold."""
        np.random.seed(42)
        expr = np.random.randn(50, 8)
        expr[:10, 4:] += 3
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        result_no_lfc = decide_tests(fit, p_value=0.05, lfc=0)
        result_with_lfc = decide_tests(fit, p_value=0.05, lfc=2)

        # With lfc threshold, should have fewer significant genes
        assert np.sum(result_with_lfc != 0) <= np.sum(result_no_lfc != 0)

    def test_direction_encoding(self):
        """Test that direction is correctly encoded."""
        np.random.seed(42)
        expr = np.random.randn(50, 8)
        expr[:5, 4:] += 5  # Up-regulated
        expr[5:10, 4:] -= 5  # Down-regulated
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)
        result = decide_tests(fit, p_value=0.01)

        # Check some up-regulated genes are marked as 1
        assert np.any(result[:5, 1] == 1)
        # Check some down-regulated genes are marked as -1
        assert np.any(result[5:10, 1] == -1)

    def test_pvalue_matrix_input(self):
        """Test decide_tests with p-value matrix input."""
        np.random.seed(42)
        p_values = np.random.uniform(0, 1, (50, 2))
        p_values[:5, :] = 0.001  # Some very significant

        result = decide_tests(p_values, p_value=0.05)

        assert result.shape == (50, 2)
        # Without coefficients, all significant are marked as 1 (not -1)
        assert np.all(result[:5, :] == 1)

    def test_auto_runs_ebayes(self):
        """Test that decide_tests auto-runs e_bayes if not already run (R parity)."""
        np.random.seed(42)
        expr = np.random.randn(50, 8)
        design = np.ones((8, 1))

        fit = lm_fit(expr, design)
        # Don't run e_bayes - should auto-run

        result = decide_tests(fit)
        assert result.shape == (50, 1)


class TestClassifyTestsF:
    """Tests for classify_tests_f function."""

    def test_fstat_only(self):
        """Test F-statistic computation."""
        np.random.seed(42)
        expr = np.random.randn(30, 12)
        expr[:5, 4:8] += 2
        expr[:5, 8:] += 3

        design = np.zeros((12, 3))
        design[:4, 0] = 1
        design[4:8, 1] = 1
        design[8:, 2] = 1

        fit = lm_fit(expr, design)
        contrasts = make_contrasts("x1-x0", "x2-x0", levels=["x0", "x1", "x2"])
        fit = contrasts_fit(fit, contrasts)
        fit = e_bayes(fit)

        f_stat, df1, df2 = classify_tests_f(fit, fstat_only=True)

        assert len(f_stat) == 30
        assert df1 >= 1
        assert np.all(f_stat >= 0)

    def test_classification_output(self):
        """Test classification matrix output."""
        np.random.seed(42)
        expr = np.random.randn(30, 12)
        expr[:5, 4:8] += 3
        expr[:5, 8:] += 4

        design = np.zeros((12, 3))
        design[:4, 0] = 1
        design[4:8, 1] = 1
        design[8:, 2] = 1

        fit = lm_fit(expr, design)
        contrasts = make_contrasts("x1-x0", "x2-x0", levels=["x0", "x1", "x2"])
        fit = contrasts_fit(fit, contrasts)
        fit = e_bayes(fit)

        result = classify_tests_f(fit, p_value=0.01, fstat_only=False)

        assert result.shape == (30, 2)
        assert set(np.unique(result)).issubset({-1, 0, 1})

    def test_single_coefficient(self):
        """Test with single coefficient (reduces to t-test)."""
        np.random.seed(42)
        expr = np.random.randn(50, 8)
        expr[:10, 4:] += 3
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        # Subset to single coefficient
        fit_single = {
            "t": fit["t"][:, 1:2],
            "df_prior": fit["df_prior"],
            "df_residual": fit["df_residual"],
        }

        f_stat, df1, df2 = classify_tests_f(fit_single, fstat_only=True)

        assert len(f_stat) == 50
        assert df1 == 1
        # F = t^2 for single coefficient
        np.testing.assert_allclose(f_stat, fit["t"][:, 1] ** 2, rtol=1e-10)

    def test_na_handling(self):
        """Test that NA t-statistics produce NA results (R parity).

        R limma sets result[i,] <- NA when any t-statistic is NA.
        """
        np.random.seed(42)
        expr = np.random.randn(10, 12)
        expr[:3, 4:8] += 3
        expr[:3, 8:] += 4

        design = np.zeros((12, 3))
        design[:4, 0] = 1
        design[4:8, 1] = 1
        design[8:, 2] = 1

        fit = lm_fit(expr, design)
        contrasts = make_contrasts("x1-x0", "x2-x0", levels=["x0", "x1", "x2"])
        fit = contrasts_fit(fit, contrasts)
        fit = e_bayes(fit)

        # Inject NA into t-statistics for gene 5
        fit["t"][5, 0] = np.nan

        result = classify_tests_f(fit, p_value=0.05, fstat_only=False)

        # Gene 5 should have all NA results (R parity)
        assert np.all(np.isnan(result[5, :]))
        # Other genes should have valid results
        assert not np.any(np.isnan(result[0, :]))


class TestDecideTestsAnnDataAutoEBayes:
    """decide_tests(adata) auto-runs e_bayes when the stored fit lacks
    a p_value slot. Pre-fix, the auto-run result was rebound locally
    but never written back to adata.uns[key], so a subsequent
    top_table(adata) raised ``"Need to run e_bayes() first"``.
    """

    def _make_adata(self):
        import anndata as ad

        rng = np.random.default_rng(0)
        counts = rng.integers(10, 1000, size=(8, 50)).astype(float)
        adata = ad.AnnData(X=counts)
        return adata

    def test_decide_tests_persists_auto_ebayes_to_uns(self):
        import pylimma

        adata = self._make_adata()
        design = np.column_stack([np.ones(8), [0] * 4 + [1] * 4])
        pylimma.voom(adata, design=design)
        pylimma.lm_fit(adata, design=design, layer="voom_E")
        # intentionally skip explicit e_bayes
        out = decide_tests(adata)
        assert out.shape == (50, 2)
        # The auto-run e_bayes result must be persisted.
        assert "t" in adata.uns["pylimma"]
        assert "p_value" in adata.uns["pylimma"]
        # top_table must now succeed.
        tt = pylimma.top_table(adata, coef=1, number=5)
        assert len(tt) == 5

    def test_decide_tests_does_not_rerun_if_already_ebayes(self):
        import pylimma

        adata = self._make_adata()
        design = np.column_stack([np.ones(8), [0] * 4 + [1] * 4])
        pylimma.voom(adata, design=design)
        pylimma.lm_fit(adata, design=design, layer="voom_E")
        pylimma.e_bayes(adata)
        t_before = np.asarray(adata.uns["pylimma"]["t"]).copy()
        decide_tests(adata)
        t_after = np.asarray(adata.uns["pylimma"]["t"])
        # No re-run, no drift.
        np.testing.assert_array_equal(t_before, t_after)

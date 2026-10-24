"""Tests for pylimma decide_tests module."""

import numpy as np
import pytest

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

    @pytest.mark.parametrize("method", ["hierarchical", "nestedF"])
    def test_f_methods_without_f_statistics_raise(self, method):
        """Rank-deficient design: e_bayes computes no F-statistics. R returns
        an all-zero result by accident (generate_all_fixtures.R records it);
        pylimma raises (docs/validation/known_differences.rst)."""
        rng = np.random.default_rng(20261022)
        group = np.repeat([0.0, 1.0], 3)
        design = np.column_stack([np.ones(6), group, group])
        with pytest.warns(UserWarning):
            fit = e_bayes(lm_fit(rng.normal(size=(30, 6)), design))
        assert "F_p_value" not in fit
        with pytest.raises(ValueError, match="needs F-statistics"):
            decide_tests(fit, method=method)


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
    a p_value slot and writes the moderated fit back to adata.uns[key],
    so a subsequent top_table(adata) finds the e_bayes slots.
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


class TestUnusableArguments:
    """R gives NA results or fails deep in qf / eigen / if() (fixture
    script); pylimma raises clear errors."""

    @pytest.fixture
    def fit(self):
        rng = np.random.default_rng(1)
        design = np.column_stack([np.ones(6), [0.0, 0, 0, 1, 1, 1], [0.0, 1, 0, 1, 0, 1]])
        return e_bayes(lm_fit(rng.normal(size=(20, 6)), design))

    def test_decide_tests_p_value_nan(self, fit):
        with pytest.raises(ValueError, match="p_value must be a number"):
            decide_tests(fit, p_value=np.nan)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"p_value": np.nan}, "between 0 and 1"),
            ({"p_value": 2}, "between 0 and 1"),
            ({"df": 0}, "df must be positive"),
            ({"df": np.nan}, "df must be positive"),
            ({"cor_matrix": np.full((3, 3), np.nan)}, "missing or infinite"),
            ({"cor_matrix": np.zeros((3, 3))}, "no positive eigenvalue"),
        ],
    )
    def test_classify_tests_f(self, fit, kwargs, match):
        with pytest.raises(ValueError, match=match):
            classify_tests_f(fit, **kwargs)

    def test_scalar_p_value_is_one_by_one(self):
        np.testing.assert_array_equal(decide_tests(0.5), [[0]])

    def test_summarize_test_results_checks_its_input(self):
        from pylimma.decide_tests import summarize_test_results

        with pytest.raises(ValueError, match="results is None"):
            summarize_test_results(None)
        with pytest.raises(ValueError, match="must be a matrix"):
            summarize_test_results(np.zeros((2, 2, 2)))
        with pytest.raises(ValueError, match="1 names for 2 columns"):
            summarize_test_results(np.zeros((3, 2)), coef_names=["a"])


def test_classify_tests_f_threshold_uses_r_qf_for_large_df():
    """R's qf takes the chi-squared limit for df > 4e5, slightly below the
    exact F quantile; a gene between the two is significant in R
    (generate_all_fixtures.R, "classifyTestsF takes its threshold from qf")."""
    x = np.full((1, 2), np.sqrt(4.60518))
    assert np.all(classify_tests_f(x, df=1e6, p_value=0.01) != 0)


def test_classify_tests_f_one_coefficient_takes_any_p_value():
    """R's one-coefficient branch never calls qf: sign(t) * (p < p.value)
    (generate_all_fixtures.R, "decideTests / classifyTestsF with unusable")."""
    result = classify_tests_f(np.array([[-2.0], [0.5], [3.0]]), df=5, p_value=2)
    np.testing.assert_array_equal(np.asarray(result).ravel(), [-1, 1, 1])

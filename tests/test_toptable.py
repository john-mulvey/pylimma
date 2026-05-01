"""Tests for pylimma toptable module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylimma.contrasts import contrasts_fit, make_contrasts
from pylimma.ebayes import e_bayes
from pylimma.lmfit import lm_fit
from pylimma.toptable import top_table

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestTopTable:
    """Tests for top_table function."""

    def test_r_parity(self):
        """Test top_table matches R limma's topTable."""
        # Load R fixtures - use DataFrame to preserve gene names
        expr = pd.read_csv(FIXTURES_DIR / "ebayes_expr.csv", index_col=0)
        design = pd.read_csv(FIXTURES_DIR / "ebayes_design.csv").values
        ref = pd.read_csv(FIXTURES_DIR / "toptable_output.csv")

        # Fit in Python
        fit = lm_fit(expr, design)
        fit = e_bayes(fit)
        result = top_table(fit, coef=1, number=20, sort_by="B")

        # Compare top genes - should match exactly
        ref_top10 = list(ref["gene"].head(10))
        result_top10 = list(result.index[:10])
        assert ref_top10 == result_top10

        # Compare statistics for top gene
        gene1 = ref["gene"].iloc[0]
        np.testing.assert_allclose(result.loc[gene1, "log_fc"], ref["log_fc"].iloc[0], rtol=1e-5)
        np.testing.assert_allclose(result.loc[gene1, "t"], ref["t"].iloc[0], rtol=1e-5)
        np.testing.assert_allclose(result.loc[gene1, "b"], ref["b"].iloc[0], rtol=1e-5)

        assert "adj_p_value" in result.columns

    def test_number_parameter(self):
        """Test that number parameter limits output rows."""
        np.random.seed(42)
        expr = np.random.randn(100, 8)
        design = np.ones((8, 1))

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        result5 = top_table(fit, coef=0, number=5)
        result20 = top_table(fit, coef=0, number=20)

        assert len(result5) == 5
        assert len(result20) == 20

    def test_sort_by_options(self):
        """Test different sort_by options."""
        np.random.seed(42)
        expr = np.random.randn(50, 8)
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        # Sort by p-value
        result_p = top_table(fit, coef=1, sort_by="p", number=50)
        assert result_p["p_value"].is_monotonic_increasing

        # Sort by absolute t-statistic (descending)
        result_t = top_table(fit, coef=1, sort_by="t", number=50)
        assert np.all(np.diff(np.abs(result_t["t"].values)) <= 1e-10)

    def test_p_value_filter(self):
        """Test p-value filtering."""
        np.random.seed(42)
        expr = np.random.randn(100, 8)
        # Add strong effects to some genes
        expr[:10, 4:] += 4
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        result = top_table(fit, coef=1, p_value=0.05, number=100)

        # All returned genes should have adj_p_value <= 0.05
        assert np.all(result["adj_p_value"] <= 0.05)

    def test_lfc_filter(self):
        """Test log fold-change filtering."""
        np.random.seed(42)
        expr = np.random.randn(100, 8)
        expr[:20, 4:] += 2  # Add effects
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        result = top_table(fit, coef=1, lfc=1.0, number=100)

        # All returned genes should have |log_fc| >= 1.0
        assert np.all(np.abs(result["log_fc"]) >= 1.0)

    def test_fc_filter(self):
        """Test fold-change filtering with fc parameter."""
        np.random.seed(42)
        expr = np.random.randn(100, 8)
        expr[:20, 4:] += 2  # Add effects
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        # fc=2 is equivalent to lfc=1
        result = top_table(fit, coef=1, fc=2.0, number=100)

        # All returned genes should have |log_fc| >= log2(2) = 1.0
        assert np.all(np.abs(result["log_fc"]) >= 1.0)

    def test_fc_equivalent_to_lfc(self):
        """Test that fc=2 gives same result as lfc=1."""
        np.random.seed(42)
        expr = np.random.randn(100, 8)
        expr[:20, 4:] += 2
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        result_fc = top_table(fit, coef=1, fc=2.0, number=100)
        result_lfc = top_table(fit, coef=1, lfc=1.0, number=100)

        pd.testing.assert_frame_equal(result_fc, result_lfc)

    def test_fc_takes_precedence_over_lfc(self):
        """Test that fc takes precedence when both are specified."""
        np.random.seed(42)
        expr = np.random.randn(100, 8)
        expr[:20, 4:] += 2
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        # fc=4 (log2=2) should take precedence over lfc=0.5
        result = top_table(fit, coef=1, fc=4.0, lfc=0.5, number=100)

        # All returned genes should have |log_fc| >= log2(4) = 2.0
        assert np.all(np.abs(result["log_fc"]) >= 2.0)

    def test_fc_less_than_one_raises(self):
        """Test that fc < 1 raises ValueError."""
        np.random.seed(42)
        expr = np.random.randn(50, 8)
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        with pytest.raises(ValueError, match="fc must be >= 1"):
            top_table(fit, coef=1, fc=0.5)

    def test_anndata_input(self):
        """Test top_table with AnnData input."""
        pytest.importorskip("anndata")
        import anndata as ad

        np.random.seed(42)
        adata = ad.AnnData(X=np.random.randn(8, 50))
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        lm_fit(adata, design)
        e_bayes(adata)
        result = top_table(adata, coef=1)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10  # default number

    def test_empty_result(self):
        """Test that empty DataFrame is returned when no genes pass filters."""
        np.random.seed(42)
        expr = np.random.randn(50, 8)  # No true effects
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        # Very stringent filter
        result = top_table(fit, coef=1, p_value=1e-10, lfc=5)

        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_requires_ebayes(self):
        """Test that top_table requires e_bayes to be run first."""
        np.random.seed(42)
        expr = np.random.randn(50, 8)
        design = np.ones((8, 1))

        fit = lm_fit(expr, design)
        # Don't run e_bayes

        with pytest.raises(ValueError, match="e_bayes"):
            top_table(fit, coef=0)

    def test_multiple_coefficients_f_test(self):
        """Test F-test with multiple coefficients."""
        np.random.seed(42)
        expr = np.random.randn(30, 12)
        expr[:5, 4:8] += 2  # Effect in group B
        expr[:5, 8:] += 3  # Effect in group C

        # Three-group design (cell means model)
        design = np.zeros((12, 3))
        design[:4, 0] = 1
        design[4:8, 1] = 1
        design[8:, 2] = 1

        fit = lm_fit(expr, design)
        contrasts = make_contrasts("x1-x0", "x2-x0", levels=["x0", "x1", "x2"])
        fit = contrasts_fit(fit, contrasts)
        fit = e_bayes(fit)

        # F-test (coef=None means all coefficients)
        result = top_table(fit, coef=None, number=10)

        assert "F" in result.columns
        assert "p_value" in result.columns
        assert len(result) == 10

    def test_confint_parameter(self):
        """Test that confint adds confidence interval columns."""
        np.random.seed(42)
        expr = np.random.randn(50, 8)
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        result = top_table(fit, coef=1, number=10, confint=True)

        assert "ci_l" in result.columns
        assert "ci_r" in result.columns
        # CI should contain log_fc
        assert np.all(result["ci_l"] <= result["log_fc"])
        assert np.all(result["ci_r"] >= result["log_fc"])

    def test_confint_custom_level(self):
        """Test confint with custom confidence level."""
        np.random.seed(42)
        expr = np.random.randn(50, 8)
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        # 99% CI should be wider than 95% CI
        result_95 = top_table(fit, coef=1, number=10, confint=True)
        result_99 = top_table(fit, coef=1, number=10, confint=0.99)

        width_95 = result_95["ci_r"] - result_95["ci_l"]
        width_99 = result_99["ci_r"] - result_99["ci_l"]
        assert np.all(width_99 > width_95)

    def test_genelist_parameter(self):
        """Test genelist adds custom annotations."""
        np.random.seed(42)
        expr = np.random.randn(50, 8)
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        genelist = pd.DataFrame(
            {
                "symbol": [f"GENE{i}" for i in range(50)],
                "description": [f"Description {i}" for i in range(50)],
            }
        )

        result = top_table(fit, coef=1, number=10, genelist=genelist)

        assert "symbol" in result.columns
        assert "description" in result.columns

    def test_resort_by_parameter(self):
        """Test resort_by applies secondary sorting using signed values (R behaviour)."""
        np.random.seed(42)
        expr = np.random.randn(50, 8)
        expr[:10, 4:] += 3  # Add effects
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        # Sort by p-value, then resort by logFC
        result_resort = top_table(fit, coef=1, number=50, sort_by="P", resort_by="logFC")

        # After resort, should be sorted by signed logFC descending (R behaviour)
        # R's resort.by uses signed values, not absolute (unlike sort.by)
        logfc = result_resort["log_fc"].values
        assert np.all(logfc[:-1] >= logfc[1:])

    def test_duplicated_rownames_handling(self):
        """Test that duplicated gene names are handled correctly (R parity).

        R limma's topTable moves duplicated rownames to an ID column and
        replaces the index with 1-indexed integers.
        """
        # Load R fixtures for true parity test
        expr_df = pd.read_csv(FIXTURES_DIR / "R_toptable_duplicated_expr.csv", index_col=0)
        design = pd.read_csv(FIXTURES_DIR / "R_toptable_duplicated_design.csv").values
        ref = pd.read_csv(FIXTURES_DIR / "R_toptable_duplicated.csv", index_col=0)

        # Extract expression matrix and gene names from R data
        expr = expr_df.values
        dup_genes = list(expr_df.index)

        fit = lm_fit(expr, design)
        fit["genes"] = dup_genes  # Set gene names in fit
        fit = e_bayes(fit)

        result = top_table(fit, coef=1, number=10, sort_by="none")

        # Index should be unique 1-indexed integers (R parity)
        assert result.index.is_unique
        assert list(result.index) == list(range(1, 11))

        # Original names should be in ID column (R parity)
        assert "ID" in result.columns
        assert list(result["ID"]) == list(ref["ID"])

        # Statistics should match R
        np.testing.assert_allclose(result["log_fc"].values, ref["logFC"].values, rtol=1e-5)
        np.testing.assert_allclose(result["t"].values, ref["t"].values, rtol=1e-5)
        np.testing.assert_allclose(result["p_value"].values, ref["P.Value"].values, rtol=1e-5)

    def test_duplicated_rownames_with_existing_id_column(self):
        """Test that ID0 is used when genelist already has an ID column."""
        np.random.seed(42)
        expr = np.random.randn(10, 8)
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        # Genelist with existing ID column and duplicated index
        genelist = pd.DataFrame(
            {
                "ID": [f"EXISTING_{i}" for i in range(10)],
                "symbol": ["A", "A", "B", "C", "C", "D", "E", "E", "F", "G"],
            }
        )
        genelist.index = [
            "GeneA",
            "GeneA",
            "GeneB",
            "GeneC",
            "GeneC",
            "GeneD",
            "GeneE",
            "GeneE",
            "GeneF",
            "GeneG",
        ]

        result = top_table(fit, coef=1, number=10, genelist=genelist)

        # Index should be unique
        assert result.index.is_unique

        # Original ID column preserved, original index in ID0
        assert "ID" in result.columns
        assert "ID0" in result.columns

    def test_explicit_vector_genelist_creates_id_column(self):
        """Explicit vector genelist becomes an ID column (R toptable.R:168).

        Mirrors R: `topTable(..., genelist=ids)` wraps `ids` as
        `data.frame(ID=ids)`; the row index stays as
        `rownames(fit$coefficients)` (or 1:N when absent).
        """
        np.random.seed(42)
        expr = np.random.randn(10, 8)
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit = e_bayes(fit)

        unique_genes = [f"Gene{i}" for i in range(10)]
        result = top_table(fit, coef=1, number=10, genelist=unique_genes)

        # ID column carries the explicit genelist values
        assert "ID" in result.columns
        assert set(result["ID"]) == set(unique_genes)
        # Row index is integer 1..N (no rownames on the input)
        assert set(result.index) == set(range(1, 11))

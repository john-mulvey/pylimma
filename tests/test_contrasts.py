"""Tests for pylimma contrasts module."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from pylimma.contrasts import make_contrasts, contrasts_fit
from pylimma.lmfit import lm_fit


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestMakeContrasts:
    """Tests for make_contrasts function."""

    def test_simple_contrast(self):
        """Test simple two-level contrast."""
        cm = make_contrasts("B-A", levels=["A", "B"])
        expected = np.array([[-1], [1]])
        np.testing.assert_array_equal(cm, expected)

    def test_multiple_contrasts(self):
        """Test multiple contrasts."""
        cm = make_contrasts("B-A", "C-A", "C-B", levels=["A", "B", "C"])
        expected = np.array([
            [-1, -1, 0],
            [1, 0, -1],
            [0, 1, 1]
        ])
        np.testing.assert_array_equal(cm, expected)

    def test_average_contrast(self):
        """Test contrast with averaging."""
        cm = make_contrasts("(B+C)/2-A", levels=["A", "B", "C"])
        expected = np.array([[-1], [0.5], [0.5]])
        np.testing.assert_allclose(cm, expected)

    def test_r_parity_contrast_matrix(self):
        """Test make_contrasts matches R limma's makeContrasts."""
        ref = pd.read_csv(FIXTURES_DIR / "contrast_matrix.csv", index_col=0)

        cm = make_contrasts("B-A", "C-A", "C-B", levels=["A", "B", "C"])

        np.testing.assert_allclose(cm, ref.values, rtol=1e-10)

    def test_design_matrix_input(self):
        """Test levels from design matrix columns."""
        design = pd.DataFrame(
            np.eye(3),
            columns=["A", "B", "C"]
        )
        cm = make_contrasts("B-A", levels=design)
        expected = np.array([[-1], [1], [0]])
        np.testing.assert_array_equal(cm, expected)

    def test_returns_dataframe(self):
        """Test that make_contrasts returns a DataFrame."""
        cm = make_contrasts("B-A", "C-A", levels=["A", "B", "C"])
        assert isinstance(cm, pd.DataFrame)
        assert list(cm.index) == ["A", "B", "C"]
        assert list(cm.columns) == ["B-A", "C-A"]

    def test_named_contrasts(self):
        """Test named contrasts via kwargs."""
        cm = make_contrasts(
            TreatmentVsControl="B-A",
            DrugVsDMSO="C-A",
            levels=["A", "B", "C"]
        )
        assert isinstance(cm, pd.DataFrame)
        assert list(cm.columns) == ["TreatmentVsControl", "DrugVsDMSO"]
        # Values should be correct
        np.testing.assert_array_equal(cm["TreatmentVsControl"].values, [-1, 1, 0])
        np.testing.assert_array_equal(cm["DrugVsDMSO"].values, [-1, 0, 1])

    def test_mixed_named_unnamed_contrasts(self):
        """Test mixing unnamed and named contrasts."""
        cm = make_contrasts(
            "C-B",  # Unnamed - expression becomes name
            AvsRest="A-(B+C)/2",  # Named
            levels=["A", "B", "C"]
        )
        assert list(cm.columns) == ["C-B", "AvsRest"]
        np.testing.assert_array_equal(cm["C-B"].values, [0, -1, 1])
        np.testing.assert_allclose(cm["AvsRest"].values, [1, -0.5, -0.5])


class TestContrastsFit:
    """Tests for contrasts_fit function."""

    def test_r_parity(self):
        """Test contrasts_fit matches R limma's contrasts.fit."""
        # Load R fixtures
        expr = pd.read_csv(FIXTURES_DIR / "contrasts_expr.csv", index_col=0).values
        design = pd.read_csv(FIXTURES_DIR / "contrasts_design.csv").values
        contrast_matrix = pd.read_csv(FIXTURES_DIR / "contrast_matrix.csv", index_col=0).values

        ref_coef = pd.read_csv(FIXTURES_DIR / "contrasts_fit2_coef.csv", index_col=0).values
        ref_stdev = pd.read_csv(FIXTURES_DIR / "contrasts_fit2_stdev.csv", index_col=0).values

        # Fit in Python
        fit = lm_fit(expr, design)
        fit2 = contrasts_fit(fit, contrast_matrix)

        # Compare
        np.testing.assert_allclose(fit2["coefficients"], ref_coef, rtol=1e-10)
        np.testing.assert_allclose(fit2["stdev_unscaled"], ref_stdev, rtol=1e-10)

    def test_removes_test_statistics(self):
        """Test that contrasts_fit removes previous test statistics."""
        np.random.seed(42)
        expr = np.random.randn(10, 6)
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])

        fit = lm_fit(expr, design)
        # Simulate having run eBayes
        fit["t"] = np.random.randn(10, 2)
        fit["p_value"] = np.random.rand(10, 2)

        contrasts = np.array([[0], [1]])
        fit2 = contrasts_fit(fit, contrasts)

        assert "t" not in fit2
        assert "p_value" not in fit2

    def test_identity_contrast(self):
        """Test that identity contrast preserves coefficients."""
        np.random.seed(42)
        expr = np.random.randn(10, 6)
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])

        fit = lm_fit(expr, design)
        original_coef = fit["coefficients"].copy()

        # Identity contrast
        contrasts = np.eye(2)
        fit2 = contrasts_fit(fit, contrasts)

        np.testing.assert_allclose(fit2["coefficients"], original_coef)

    def test_single_contrast(self):
        """Test applying a single contrast."""
        np.random.seed(42)
        expr = np.random.randn(10, 6)
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])

        fit = lm_fit(expr, design)

        # Extract second coefficient only
        contrasts = np.array([[0], [1]])
        fit2 = contrasts_fit(fit, contrasts)

        assert fit2["coefficients"].shape == (10, 1)
        np.testing.assert_allclose(
            fit2["coefficients"][:, 0],
            fit["coefficients"][:, 1]
        )

    def test_anndata_input(self):
        """Test contrasts_fit with AnnData input."""
        pytest.importorskip("anndata")
        import anndata as ad

        np.random.seed(42)
        adata = ad.AnnData(X=np.random.randn(6, 10))
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])

        lm_fit(adata, design)
        contrasts = np.array([[0], [1]])
        result = contrasts_fit(adata, contrasts)

        assert result is None  # Returns None for AnnData
        assert adata.uns["pylimma"]["coefficients"].shape == (10, 1)
        assert "contrasts" in adata.uns["pylimma"]

    def test_empty_contrasts(self):
        """Test handling of empty contrast matrix."""
        np.random.seed(42)
        expr = np.random.randn(10, 6)
        design = np.ones((6, 1))

        fit = lm_fit(expr, design)
        contrasts = np.zeros((1, 0))
        fit2 = contrasts_fit(fit, contrasts)

        assert fit2["coefficients"].shape == (10, 0)

    def test_preserves_contrast_names_from_dataframe(self):
        """Test that contrasts_fit preserves contrast names from DataFrame."""
        np.random.seed(42)
        expr = np.random.randn(10, 6)
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])

        fit = lm_fit(expr, design)

        # Create named contrasts
        contrasts = make_contrasts(
            TreatmentEffect="x1",
            levels=["x0", "x1"]
        )
        fit2 = contrasts_fit(fit, contrasts)

        assert "contrast_names" in fit2
        assert fit2["contrast_names"] == ["TreatmentEffect"]

    def test_generates_default_names_for_array(self):
        """Test that contrasts_fit generates default names for array input."""
        np.random.seed(42)
        expr = np.random.randn(10, 6)
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])

        fit = lm_fit(expr, design)

        # Plain array, no names
        contrasts = np.array([[0, 1], [1, -1]])
        fit2 = contrasts_fit(fit, contrasts)

        assert "contrast_names" in fit2
        assert fit2["contrast_names"] == ["contrast0", "contrast1"]

    def test_coefficients_parameter_by_index(self):
        """Test coefficients parameter with integer indices."""
        np.random.seed(42)
        expr = np.random.randn(10, 9)
        design = np.column_stack([np.ones(9), [0]*3 + [1]*3 + [0]*3, [0]*6 + [1]*3])

        fit = lm_fit(expr, design)
        original_coef = fit["coefficients"].copy()

        # Select coefficients 1 and 2 (skip intercept)
        fit2 = contrasts_fit(fit, coefficients=[1, 2])

        assert fit2["coefficients"].shape == (10, 2)
        np.testing.assert_allclose(fit2["coefficients"][:, 0], original_coef[:, 1])
        np.testing.assert_allclose(fit2["coefficients"][:, 1], original_coef[:, 2])

    def test_coefficients_parameter_single_int(self):
        """Test coefficients parameter with single integer."""
        np.random.seed(42)
        expr = np.random.randn(10, 6)
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])

        fit = lm_fit(expr, design)
        original_coef = fit["coefficients"].copy()

        # Select single coefficient
        fit2 = contrasts_fit(fit, coefficients=1)

        assert fit2["coefficients"].shape == (10, 1)
        np.testing.assert_allclose(fit2["coefficients"][:, 0], original_coef[:, 1])

    def test_coefficients_parameter_by_name(self):
        """Test coefficients parameter with names."""
        np.random.seed(42)
        expr = np.random.randn(10, 6)
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])

        fit = lm_fit(expr, design)
        fit["coef_names"] = ["Intercept", "Treatment"]
        original_coef = fit["coefficients"].copy()

        # Select by name
        fit2 = contrasts_fit(fit, coefficients="Treatment")

        assert fit2["coefficients"].shape == (10, 1)
        np.testing.assert_allclose(fit2["coefficients"][:, 0], original_coef[:, 1])
        assert fit2["contrast_names"] == ["Treatment"]

    def test_coefficients_equivalent_to_identity_contrast(self):
        """Test that coefficients param gives same result as identity contrast."""
        np.random.seed(42)
        expr = np.random.randn(10, 6)
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])

        fit = lm_fit(expr, design)

        # Using coefficients parameter
        fit_coef = contrasts_fit(fit.copy(), coefficients=[1])

        # Using equivalent contrast matrix
        contrast_matrix = np.array([[0], [1]])
        fit_contrast = contrasts_fit(fit.copy(), contrasts=contrast_matrix)

        np.testing.assert_allclose(
            fit_coef["coefficients"], fit_contrast["coefficients"]
        )
        np.testing.assert_allclose(
            fit_coef["stdev_unscaled"], fit_contrast["stdev_unscaled"]
        )

    def test_both_contrasts_and_coefficients_raises(self):
        """Test that specifying both contrasts and coefficients raises error."""
        np.random.seed(42)
        expr = np.random.randn(10, 6)
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])

        fit = lm_fit(expr, design)

        with pytest.raises(ValueError, match="Cannot specify both"):
            contrasts_fit(fit, contrasts=np.array([[0], [1]]), coefficients=[1])

    def test_neither_contrasts_nor_coefficients_raises(self):
        """Test that specifying neither contrasts nor coefficients raises error."""
        np.random.seed(42)
        expr = np.random.randn(10, 6)
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])

        fit = lm_fit(expr, design)

        with pytest.raises(ValueError, match="Must specify either"):
            contrasts_fit(fit)


class TestContrastNamesInTopTable:
    """Tests for contrast name support in top_table."""

    def test_top_table_accepts_contrast_name(self):
        """Test that top_table accepts contrast names for coef parameter."""
        from pylimma.ebayes import e_bayes
        from pylimma.toptable import top_table

        np.random.seed(42)
        expr = np.random.randn(50, 8)
        expr[:10, 4:] += 3  # Add effect
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        contrasts = make_contrasts(
            TreatmentEffect="x1",
            levels=["x0", "x1"]
        )
        fit2 = contrasts_fit(fit, contrasts)
        fit2 = e_bayes(fit2)

        # Should work with name
        result = top_table(fit2, coef="TreatmentEffect", number=10)
        assert len(result) == 10

        # Should also work with index
        result_idx = top_table(fit2, coef=0, number=10)
        pd.testing.assert_frame_equal(result, result_idx)

    def test_top_table_invalid_contrast_name_raises(self):
        """Test that top_table raises for invalid contrast name."""
        from pylimma.ebayes import e_bayes
        from pylimma.toptable import top_table

        np.random.seed(42)
        expr = np.random.randn(50, 8)
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])

        fit = lm_fit(expr, design)
        contrasts = make_contrasts(
            TreatmentEffect="x1",
            levels=["x0", "x1"]
        )
        fit2 = contrasts_fit(fit, contrasts)
        fit2 = e_bayes(fit2)

        with pytest.raises(ValueError, match="not found"):
            top_table(fit2, coef="NonExistent")

    def test_f_test_uses_contrast_names_in_columns(self):
        """Test that F-test output uses contrast names for coefficient columns."""
        from pylimma.ebayes import e_bayes
        from pylimma.toptable import top_table

        np.random.seed(42)
        expr = np.random.randn(30, 12)
        expr[:5, 4:8] += 2
        expr[:5, 8:] += 3

        design = np.zeros((12, 3))
        design[:4, 0] = 1
        design[4:8, 1] = 1
        design[8:, 2] = 1

        fit = lm_fit(expr, design)
        contrasts = make_contrasts(
            BvsA="x1-x0",
            CvsA="x2-x0",
            levels=["x0", "x1", "x2"]
        )
        fit2 = contrasts_fit(fit, contrasts)
        fit2 = e_bayes(fit2)

        # F-test with all contrasts
        result = top_table(fit2, coef=None, number=10)

        # Columns should use contrast names
        assert "BvsA" in result.columns
        assert "CvsA" in result.columns

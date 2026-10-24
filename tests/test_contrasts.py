"""Tests for pylimma contrasts module."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import patsy
import pytest

from pylimma.contrasts import (
    contrasts_fit,
    make_contrasts,
    model_matrix,
    model_matrix_with_names,
)
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
        expected = np.array([[-1, -1, 0], [1, 0, -1], [0, 1, 1]])
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
        design = pd.DataFrame(np.eye(3), columns=["A", "B", "C"])
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
        cm = make_contrasts(TreatmentVsControl="B-A", DrugVsDMSO="C-A", levels=["A", "B", "C"])
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
            levels=["A", "B", "C"],
        )
        assert list(cm.columns) == ["C-B", "AvsRest"]
        np.testing.assert_array_equal(cm["C-B"].values, [0, -1, 1])
        np.testing.assert_allclose(cm["AvsRest"].values, [1, -0.5, -0.5])


class TestMakeContrastsLevels:
    """Parameter names taken from levels, as R's makeContrasts reads them."""

    def test_single_string_is_one_level(self):
        cm = make_contrasts("2*AB", levels="AB")
        assert list(cm.index) == ["AB"]
        np.testing.assert_array_equal(cm.values, [[2.0]])

    def test_design_matrix_without_names_raises(self):
        # R fails with "argument is of length zero"; pylimma used to invent x0, x1
        with pytest.raises(ValueError, match="levels must be the parameter names"):
            make_contrasts("x1-x0", levels=np.eye(2))

    def test_numeric_levels_raise(self):
        with pytest.raises(ValueError, match="levels must be the parameter names"):
            make_contrasts("B-A", levels=[1, 2])

    def test_numeric_categories_are_not_valid_names(self):
        with pytest.raises(ValueError, match="names: 1, 2"):
            make_contrasts("B-A", levels=pd.Categorical([1, 2]))

    def test_duplicated_levels_raise(self):
        # R silently gives the indicator to the last copy only
        with pytest.raises(ValueError, match="duplicated: A"):
            make_contrasts("A", levels=["A", "B", "A"])


class TestMakeContrastsExpressions:
    """Contrasts are R arithmetic on the levels; anything else raises."""

    def test_unknown_name_raises(self):
        # R finds an unrelated object (stats::D) and fails obscurely
        with pytest.raises(ValueError, match="'D' is not one of the levels"):
            make_contrasts("B-D", levels=["A", "B"])

    def test_r_constants_and_functions_raise(self):
        with pytest.raises(ValueError, match="'pi' is not one of the levels"):
            make_contrasts("B-A*pi", levels=["A", "B"])
        with pytest.raises(ValueError, match="'mean' is not one of the levels"):
            make_contrasts("mean(A)", levels=["A", "B"])

    def test_other_syntax_raises(self):
        for expr in ("A %% 2", "A, B", "A[0]", "A == B"):
            with pytest.raises(ValueError, match="unsupported syntax"):
                make_contrasts(expr, levels=["A", "B"])

    def test_malformed_expression_raises(self):
        for expr in ("A B", "A-", "", "(A", "A // B", "A(B)"):
            with pytest.raises(ValueError, match="Could not evaluate contrast expression"):
                make_contrasts(expr, levels=["A", "B"])

    def test_expression_over_several_lines_raises(self):
        # R evaluates only the last line: "A + B\n- C" is -C
        with pytest.raises(ValueError, match="spans several lines"):
            make_contrasts("A + B\n- C", levels=["A", "B", "C"])

    def test_non_string_expression_raises(self):
        # R accepts makeContrasts(B = 1, ...) and fills the column with ones
        for kwargs in ({"B": 1}, {"contrasts": ["B-A", None]}):
            with pytest.raises(ValueError, match="must be strings"):
                make_contrasts(levels=["A", "B"], **kwargs)

    def test_number_alone_fills_the_column(self):
        # R: cm[, j] <- 2 recycles to every level
        cm = make_contrasts("2", levels=["A", "B"])
        np.testing.assert_array_equal(cm["2"].values, [2.0, 2.0])


class TestMakeContrastsArgument:
    """The contrasts= argument, as R's makeContrasts(contrasts=)."""

    def test_single_string_is_one_contrast(self):
        cm = make_contrasts(contrasts="B-A", levels=["A", "B"])
        assert list(cm.columns) == ["B-A"]
        np.testing.assert_array_equal(cm["B-A"].values, [-1.0, 1.0])

    def test_series_with_default_index_is_unnamed(self):
        cm = make_contrasts(contrasts=pd.Series(["B-A", "A"]), levels=["A", "B"])
        assert list(cm.columns) == ["B-A", "A"]

    def test_only_an_empty_name_falls_back_to_the_expression(self):
        cm = make_contrasts(contrasts={0: "B-A", "": "A"}, levels=["A", "B"])
        assert list(cm.columns) == [0, "A"]

    def test_keyword_contrasts_with_contrasts_argument_raise(self):
        # R: "Can't specify both ... and contrasts" (named ... arguments count)
        with pytest.raises(ValueError, match="Cannot specify both"):
            make_contrasts(BvsA="B-A", contrasts=["B-A"], levels=["A", "B"])

    def test_no_contrasts_raise(self):
        # R returns a 2 x 0 matrix for contrasts=character(0) and fails for an empty ...
        for kwargs in ({}, {"contrasts": []}):
            with pytest.raises(ValueError, match="No contrasts specified"):
                make_contrasts(levels=["A", "B"], **kwargs)


class TestFormulaCaret:
    """^ in a formula raises: patsy reads it as XOR, R as a power or crossing."""

    @pytest.fixture
    def data(self):
        return pd.DataFrame({"age": np.arange(1, 7), "g": list("aabbcc")})

    def test_caret_raises_in_every_formula_entry_point(self, data):
        adata = ad.AnnData(
            X=np.random.default_rng(0).normal(size=(6, 4)),
            obs=data.set_axis([f"s{i}" for i in range(6)]),
        )
        builders = (
            lambda f: model_matrix(f, data),
            lambda f: model_matrix_with_names(f, data),
            lambda f: lm_fit(adata, f),
        )
        for build in builders:
            # an integer column gave silently wrong numbers (1^2 is 3)
            with pytest.raises(ValueError, match=r"use \*\* instead"):
                build("~ I(age^2)")

    def test_caret_in_a_quoted_name_is_allowed(self):
        data = pd.DataFrame({"a^b": [1.0, 2.0, 3.0]})
        np.testing.assert_array_equal(model_matrix('~ Q("a^b")', data)[:, 1], [1, 2, 3])


class TestFormulaChecks:
    """R's model.matrix needs two levels for every factor term, also without
    an intercept (fixture script); a formula that is not a string raises."""

    def test_single_level_factor_raises(self):
        data = pd.DataFrame({"f": ["a"] * 4, "x": np.arange(4.0)})
        for formula in ("~ f", "~ 0 + f", "~ x + f"):
            with pytest.raises(ValueError, match="2 or more levels"):
                model_matrix(formula, data)
        assert model_matrix("~ x", data).shape == (4, 2)

    def test_formula_must_be_a_string(self):
        with pytest.raises(TypeError, match="formula must be a string"):
            model_matrix(None, pd.DataFrame({"x": [1.0, 2.0]}))


class TestFormulaNames:
    """Formula names: data columns, patsy built-ins, np and pd only."""

    def test_numpy_and_patsy_builtins_resolve(self):
        data = pd.DataFrame({"dose": [1.0, 2.0, 4.0]})
        design = model_matrix("~ np.log2(dose) + center(dose)", data)
        np.testing.assert_allclose(design[:, 1:], [[0, -4 / 3], [1, -1 / 3], [2, 5 / 3]])

    def test_module_names_do_not_resolve(self):
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        for name in ("tokenize", "io", "warnings"):
            with pytest.raises(patsy.PatsyError, match=name):
                model_matrix(f"~ x + {name}", data)


class TestFormulaOrderedFactor:
    """An ordered categorical coded with treatment contrasts raises (R: contr.poly)."""

    @pytest.fixture
    def data(self):
        return pd.DataFrame(
            {
                "o": pd.Categorical(
                    ["lo", "mid", "hi"] * 2, categories=["lo", "mid", "hi"], ordered=True
                ),
                "g": list("aabbcc"),
            }
        )

    def test_bare_ordered_factor_raises(self, data):
        with pytest.raises(ValueError, match=r"ordered categorical 'o'.*C\(o, Poly\)"):
            model_matrix("~ o", data)
        with pytest.raises(ValueError, match="ordered categorical 'o'"):
            model_matrix_with_names("~ g + o", data)
        # wrapped without naming a contrast, patsy still uses treatment coding
        for formula in ("~ C(o)", '~ Q("o")', '~ C(o, levels=["hi", "mid", "lo"])'):
            with pytest.raises(ValueError, match="without a named contrast"):
                model_matrix(formula, data)

    def test_explicit_coding_is_kept(self, data):
        names = list(model_matrix_with_names("~ C(o, Treatment)", data).columns)
        assert names == ["Intercept", "C(o, Treatment)[T.mid]", "C(o, Treatment)[T.hi]"]
        assert model_matrix("~ C(o, contrast=Poly)", data).shape == (6, 3)

    def test_term_derived_from_an_ordered_column_is_unaffected(self, data):
        names = list(model_matrix_with_names('~ I(o == "lo")', data).columns)
        assert names == ["Intercept", 'I(o == "lo")[T.True]']

    def test_unordered_factor_is_unaffected(self, data):
        data["o"] = data["o"].cat.as_unordered()
        assert model_matrix("~ o", data).shape == (6, 3)


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

        # Compare the fit before and after the contrasts
        for key, name in [("coefficients", "coef"), ("stdev_unscaled", "stdev")]:
            ref_fit = pd.read_csv(FIXTURES_DIR / f"contrasts_fit_{name}.csv", index_col=0).values
            np.testing.assert_allclose(fit[key], ref_fit, rtol=1e-10)
        np.testing.assert_allclose(fit2["coefficients"], ref_coef, rtol=1e-10)
        np.testing.assert_allclose(fit2["stdev_unscaled"], ref_stdev, rtol=1e-10)
        ref_cov = pd.read_csv(FIXTURES_DIR / "contrasts_fit2_cov.csv", index_col=0).values
        np.testing.assert_allclose(fit2["cov_coefficients"], ref_cov, rtol=1e-10)

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
        np.testing.assert_allclose(fit2["coefficients"][:, 0], fit["coefficients"][:, 1])

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
        contrasts = make_contrasts(TreatmentEffect="x2", levels=["x1", "x2"])
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
        design = np.column_stack([np.ones(9), [0] * 3 + [1] * 3 + [0] * 3, [0] * 6 + [1] * 3])

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

        np.testing.assert_allclose(fit_coef["coefficients"], fit_contrast["coefficients"])
        np.testing.assert_allclose(fit_coef["stdev_unscaled"], fit_contrast["stdev_unscaled"])

    def test_both_contrasts_and_coefficients_raises(self):
        """Test that specifying both contrasts and coefficients raises error."""
        np.random.seed(42)
        expr = np.random.randn(10, 6)
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])

        fit = lm_fit(expr, design)

        with pytest.raises(
            ValueError, match="Must specify exactly one of contrasts or coefficients"
        ):
            contrasts_fit(fit, contrasts=np.array([[0], [1]]), coefficients=[1])

    def test_neither_contrasts_nor_coefficients_raises(self):
        """Test that specifying neither contrasts nor coefficients raises error."""
        np.random.seed(42)
        expr = np.random.randn(10, 6)
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])

        fit = lm_fit(expr, design)

        with pytest.raises(
            ValueError, match="Must specify exactly one of contrasts or coefficients"
        ):
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
        contrasts = make_contrasts(TreatmentEffect="x2", levels=["x1", "x2"])
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
        contrasts = make_contrasts(TreatmentEffect="x2", levels=["x1", "x2"])
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
        contrasts = make_contrasts(BvsA="x2-x1", CvsA="x3-x1", levels=["x1", "x2", "x3"])
        fit2 = contrasts_fit(fit, contrasts)
        fit2 = e_bayes(fit2)

        # F-test with all contrasts
        result = top_table(fit2, coef=None, number=10)

        # Columns should use contrast names
        assert "BvsA" in result.columns
        assert "CvsA" in result.columns


class TestContrastsFitArguments:
    """R behaviour recorded in the fixture script."""

    def _fit(self):
        from pylimma import lm_fit

        design = np.column_stack([np.ones(6), [0.0, 0, 0, 1, 1, 1]])
        return lm_fit(np.random.default_rng(1).normal(size=(10, 6)), design)

    def test_float_coefficient_subsets(self):
        fit = self._fit()
        np.testing.assert_array_equal(
            contrasts_fit(fit, coefficients=1.0)["coefficients"],
            contrasts_fit(fit, coefficients=1)["coefficients"],
        )

    def test_wrong_sized_cov_coefficients_raises(self):
        fit = self._fit()
        fit["cov_coefficients"] = np.eye(3)
        with pytest.raises(ValueError, match=r"has shape \(3, 3\), but the fit has 2"):
            contrasts_fit(fit, np.array([[0.0], [1.0]]))
        fit["cov_coefficients"] = 1.0
        with pytest.raises(ValueError, match=r"cov_coefficients has shape \(\)"):
            contrasts_fit(fit, np.array([[0.0], [1.0]]))

    def test_missing_fit_raises_r_error(self):
        with pytest.raises(ValueError, match="fit must contain coefficients component"):
            contrasts_fit(None, np.array([[0.0], [1.0]]))

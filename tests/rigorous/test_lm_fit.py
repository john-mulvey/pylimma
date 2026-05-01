"""
Rigorous per-branch parity tests for pylimma.lmfit.lm_fit.

Each test exercises a specific R branch of lmFit() in R limma's
lmfit.R.

These tests were added by a rigorous single-function audit on
2026-04-23. They are intentionally tight (rtol=1e-8) and mostly run
a live R subprocess via helpers.run_r_comparison so that any
regression surfaces immediately. Tests that only assert a warning /
error (not a numerical output) do not need live R because the
trigger condition is documented in the R source.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pylimma.lmfit import lm_fit

from ..helpers import (
    compare_arrays,
    limma_available,
    run_r_comparison,
)

pytestmark = pytest.mark.skipif(not limma_available(), reason="R/limma not available")


# -----------------------------------------------------------------------------
# Small shared data
# -----------------------------------------------------------------------------


def _two_group_expr(rng=None, n_genes=20, n_samples=8):
    rng = rng if rng is not None else np.random.default_rng(0)
    expr = rng.standard_normal((n_genes, n_samples))
    design = np.column_stack([np.ones(n_samples), np.array([0] * 4 + [1] * 4, dtype=float)])
    return expr, design


class TestRigorousLmFit:
    """One class per function, one test per uncovered/partial branch."""

    # ------------------------------------------------------------------
    # R-B2: data.frame with exactly one non-numeric column treated as
    # gene IDs. R emits message() and uses as.matrix(object[,-1]).
    # ------------------------------------------------------------------
    def test_dataframe_with_gene_id_column(self):
        """Exercises R-B2: lmfit.R:15-22 (1 non-numeric col = gene IDs)."""
        rng = np.random.default_rng(1)
        expr_vals = rng.standard_normal((6, 4))
        df = pd.DataFrame(expr_vals, columns=[f"s{i}" for i in range(4)])
        df.insert(0, "gene_id", [f"g{i}" for i in range(6)])
        design = np.column_stack([np.ones(4), np.array([0, 0, 1, 1], dtype=float)])

        # Python currently rejects a DataFrame with a mix of numeric +
        # one non-numeric column with a bespoke error. R accepts the
        # same input silently (emits only a message). The differential
        # test asserts pylimma matches R's coefficients; if the error
        # path is kept it will fail -- that IS the divergence.
        py_coef = None
        py_error = None
        try:
            fit = lm_fit(df, design)
            py_coef = np.asarray(fit["coefficients"], dtype=float)
        except Exception as exc:  # pragma: no cover - reported below
            py_error = exc

        r_code = """
        library(limma)
        expr <- read.csv('{tmpdir}/expr.csv', row.names=1,
                         stringsAsFactors = FALSE)
        design <- as.matrix(read.csv('{tmpdir}/design.csv', row.names=1))
        fit <- lmFit(expr, design)
        coef <- fit$coefficients
        """
        r_results = run_r_comparison(
            py_data={"expr": df, "design": design},
            r_code_template=r_code,
            output_vars=["coef"],
        )
        r_coef = np.asarray(r_results["coef"], dtype=float)

        assert py_error is None, (
            f"lm_fit raised on DataFrame+gene-ID column; R accepts it: {py_error!r}"
        )
        result = compare_arrays(r_coef, py_coef, rtol=1e-8)
        assert result["match"], (
            f"Gene-ID DataFrame lmFit coefficients differ: max_rel={result['max_rel_diff']:.2e}"
        )

    # ------------------------------------------------------------------
    # R-B3: data.frame with >1 non-numeric columns -> stop()
    # ------------------------------------------------------------------
    def test_dataframe_multiple_nonnumeric_errors(self):
        """Exercises R-B3: lmfit.R:23-25 (>=2 non-numeric cols = stop)."""
        df = pd.DataFrame(
            {
                "gene_id": ["g1", "g2", "g3"],
                "other": ["a", "b", "c"],
                "s1": [1.0, 2.0, 3.0],
                "s2": [2.0, 3.0, 4.0],
            }
        )
        design = np.column_stack([np.ones(2), np.array([0, 1], dtype=float)])

        # R: stop("Expression object should be numeric, instead it is a
        #         data.frame with 2 non-numeric columns")
        with pytest.raises((TypeError, ValueError)):
            lm_fit(df, design)

    # ------------------------------------------------------------------
    # R-B5: zero-row expression matrix -> stop()
    # ------------------------------------------------------------------
    def test_zero_row_expression_errors(self):
        """Exercises R-B5: lmfit.R:31 ('expression matrix has zero rows')."""
        expr = np.zeros((0, 4))
        design = np.column_stack([np.ones(4), np.array([0, 0, 1, 1], dtype=float)])

        # R's behaviour is definitive: stop("expression matrix has zero
        # rows"). pylimma currently has no such guard; the test will
        # FAIL, which is the documented divergence.
        with pytest.raises((ValueError, IndexError)):
            lm_fit(expr, design)

    # ------------------------------------------------------------------
    # R-B8 sub-check: anyNA(design) -> stop()
    # ------------------------------------------------------------------
    def test_na_in_design_errors(self):
        """Exercises R-B8: lmfit.R:41 ('NAs not allowed in design matrix')."""
        expr, design = _two_group_expr()
        design = design.copy()
        design[0, 1] = np.nan

        with pytest.raises((ValueError, TypeError, RuntimeError)):
            lm_fit(expr, design)

    # ------------------------------------------------------------------
    # R-B8 sub-check: non-numeric design (mode(design) != "numeric")
    # ------------------------------------------------------------------
    def test_nonnumeric_design_errors(self):
        """Exercises R-B8: lmfit.R:39 ('design must be a numeric matrix')."""
        expr, _ = _two_group_expr()
        design = np.array([["a"] * 2] * 8, dtype=object)

        with pytest.raises((ValueError, TypeError)):
            lm_fit(expr, design)

    # ------------------------------------------------------------------
    # R-B15: match.arg(method, c("ls","robust")) -> stop on bad method
    # ------------------------------------------------------------------
    def test_invalid_method_errors(self):
        """Exercises R-B15: lmfit.R:56 (match.arg on method)."""
        expr, design = _two_group_expr()
        with pytest.raises(ValueError):
            lm_fit(expr, design, method="bogus")

    # ------------------------------------------------------------------
    # R-B10/R-B12: ndups / spacing default from printer metadata
    # N/A against R for EList: R's getEAWP (getEAWP at lmfit.R:438-453)
    # populates y$printer only for unclassed MAList-style lists, never
    # for EList. Since MAList / RGList wrappers are deliberately
    # out-of-scope for pylimma (policy_data_class_wrappers), this
    # branch has no reachable R pathway in pylimma-accepted inputs
    # and is documented rather than tested. (pylimma's _printer_attr
    # will happily read from an EList dict, which is a harmless
    # Python-only extension, not an R divergence.)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # R-B16: ndups>1 reduces Amean and probes
    #   R does:  y$Amean <- rowMeans(unwrapdups(y$Amean, ndups, spacing))
    #            y$probes <- uniquegenelist(y$probes, ndups, spacing)
    # so fit$Amean has length n_genes/ndups to match fit$coefficients.
    # ------------------------------------------------------------------
    def test_ndups_greater_than_1_reduces_amean(self):
        """Exercises R-B16: lmfit.R:59-62 (Amean + probes collapsed)."""
        rng = np.random.default_rng(3)
        ndups = 2
        n_blocks = 5
        n_arrays = 4
        n_genes = n_blocks * ndups
        expr = rng.standard_normal((n_genes, n_arrays))
        design = np.column_stack([np.ones(n_arrays), np.array([0, 0, 1, 1], dtype=float)])

        fit = lm_fit(expr, design, ndups=ndups, correlation=0.2)
        py_coef = np.asarray(fit["coefficients"], dtype=float)
        py_amean = np.asarray(fit["Amean"], dtype=float)

        # R reference.
        r_code = """
        library(limma)
        expr <- as.matrix(read.csv('{tmpdir}/expr.csv', row.names=1))
        design <- as.matrix(read.csv('{tmpdir}/design.csv', row.names=1))
        fit <- lmFit(expr, design, ndups=2, correlation=0.2)
        coef <- fit$coefficients
        amean <- matrix(fit$Amean, ncol=1)
        """
        r_results = run_r_comparison(
            py_data={"expr": expr, "design": design},
            r_code_template=r_code,
            output_vars=["coef", "amean"],
        )
        r_coef = np.asarray(r_results["coef"], dtype=float)
        r_amean = np.asarray(r_results["amean"], dtype=float).ravel()

        # If pylimma forgets to reduce Amean, py_amean will have
        # n_genes elements and r_amean n_blocks. That shape mismatch
        # fails here first.
        assert py_amean.shape == r_amean.shape, (
            f"Amean shape mismatch: R={r_amean.shape} Py={py_amean.shape} "
            f"(pylimma may be skipping the ndups>1 Amean collapse)"
        )
        # And the coefficients should also agree.
        assert py_coef.shape == r_coef.shape, (
            f"Coefficient shape mismatch: R={r_coef.shape} Py={py_coef.shape}"
        )
        assert compare_arrays(r_coef, py_coef, rtol=1e-8)["match"]
        assert compare_arrays(r_amean, py_amean, rtol=1e-8)["match"]

    # ------------------------------------------------------------------
    # R-B18: method="robust" combined with block or ndups>1 emits a
    # warning in R: warning("Correlation cannot be combined with
    # robust regression..."). pylimma should emit the equivalent.
    # ------------------------------------------------------------------
    def test_robust_with_block_emits_warning(self):
        """Exercises R-B18: lmfit.R:66 (warning on robust+blocking)."""
        rng = np.random.default_rng(4)
        expr = rng.standard_normal((6, 4))
        design = np.column_stack([np.ones(4), np.array([0, 0, 1, 1], dtype=float)])
        block = np.array([1, 1, 2, 2])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lm_fit(expr, design, method="robust", block=block)
        messages = [str(w.message) for w in caught]
        assert any("Correlation cannot be combined" in m for m in messages), (
            f"Expected 'Correlation cannot be combined' warning; got {messages}"
        )

    # ------------------------------------------------------------------
    # R-B22: Partial NA coefficients warning
    #   if(NCOL(fit$coefficients)>1) {
    #     n <- rowSums(is.na(fit$coefficients))
    #     n <- sum(n>0 & n<NCOL(fit$coefficients))
    #     if(n>0) warning("Partial NA coefficients for ",n," probe(s)")
    #   }
    # Triggered when some genes have partial missing observations that
    # leave some but not all coefficients estimable.
    # ------------------------------------------------------------------
    def test_partial_na_coefficients_warning(self):
        """Exercises R-B22: lmfit.R:77-81 ('Partial NA coefficients')."""
        # Construct a dataset that forces partial-NA coefficients. With
        # a 4-column group-means design, if one group has all-NA for a
        # gene, only that coefficient becomes NA; the other three are
        # estimable. This is precisely the "partial NA" case R warns
        # about.
        n_samples = 12
        groups = np.repeat(np.arange(4), 3)
        design = np.zeros((n_samples, 4))
        for i, g in enumerate(groups):
            design[i, g] = 1.0

        rng = np.random.default_rng(5)
        expr = rng.standard_normal((5, n_samples))
        # Gene 0: group 3 entirely NA -> coefficient 3 becomes NA, the
        # other three remain estimable.
        expr[0, groups == 3] = np.nan
        # Gene 1: group 0 entirely NA -> coefficient 0 becomes NA.
        expr[1, groups == 0] = np.nan

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = lm_fit(expr, design)
        messages = [str(w.message) for w in caught]

        # R reference: emits a warning
        r_code = """
        library(limma)
        expr <- as.matrix(read.csv('{tmpdir}/expr.csv', row.names=1))
        design <- as.matrix(read.csv('{tmpdir}/design.csv', row.names=1))
        w <- character(0)
        wh <- withCallingHandlers(
            lmFit(expr, design),
            warning = function(cond) {{
                w <<- c(w, conditionMessage(cond))
                invokeRestart('muffleWarning')
            }}
        )
        partial_hit <- matrix(
            as.integer(any(grepl('Partial NA', w))), 1, 1
        )
        """
        r_results = run_r_comparison(
            py_data={"expr": expr, "design": design},
            r_code_template=r_code,
            output_vars=["partial_hit"],
        )
        r_partial_hit = bool(int(r_results["partial_hit"].ravel()[0]))

        # Sanity: the fit should indeed have partial NA coefficients.
        coefs = np.asarray(fit["coefficients"], dtype=float)
        row_nas = np.sum(np.isnan(coefs), axis=1)
        partial_rows = int(np.sum((row_nas > 0) & (row_nas < coefs.shape[1])))
        assert partial_rows > 0, (
            f"Test setup failed to produce partial-NA coefficients; per-row NA counts: {row_nas}"
        )

        assert r_partial_hit, (
            "R sanity check failed: expected a 'Partial NA' warning from R "
            "on this input but did not see one."
        )
        assert any("Partial NA" in m for m in messages), (
            f"Expected pylimma to emit 'Partial NA coefficients' warning "
            f"(R does). Got warnings: {messages}"
        )

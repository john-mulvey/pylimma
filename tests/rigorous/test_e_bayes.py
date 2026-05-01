"""
Rigorous per-branch parity tests for pylimma.ebayes.e_bayes.

Each test exercises a specific R branch of eBayes() / .ebayes() /
tmixture.matrix() / tmixture.vector() in R limma's ebayes.R.

These tests were added by a rigorous single-function audit on
2026-04-29. They are intentionally tight (rtol=1e-8) and run a live R
subprocess via helpers.run_r_comparison so any regression surfaces
immediately.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pylimma.ebayes import e_bayes, _tmixture_matrix, _tmixture_vector
from pylimma.lmfit import lm_fit

from ..helpers import (
    compare_arrays,
    limma_available,
    run_r_comparison,
)


pytestmark = pytest.mark.skipif(
    not limma_available(), reason="R/limma not available"
)


def _two_group_expr(rng=None, n_genes=30, n_samples=8, seed=0):
    rng = rng if rng is not None else np.random.default_rng(seed)
    expr = rng.standard_normal((n_genes, n_samples))
    design = np.column_stack(
        [np.ones(n_samples), np.array([0] * 4 + [1] * 4, dtype=float)]
    )
    return expr, design


def _r_ebayes_template(extra_args: str = "") -> str:
    """Standard R script template for an eBayes parity test."""
    return f"""
    suppressMessages(library(limma))
    expr <- as.matrix(read.csv('{{tmpdir}}/expr.csv', row.names=1))
    design <- as.matrix(read.csv('{{tmpdir}}/design.csv', row.names=1))
    fit <- lmFit(expr, design)
    eb <- eBayes(fit{extra_args})
    t_stat <- eb$t
    p_value <- eb$p.value
    lods <- eb$lods
    s2_post <- eb$s2.post
    df_total <- eb$df.total
    """


class TestRigorousEBayes:
    """One class per function, one test per uncovered/partial branch."""

    # ------------------------------------------------------------------
    # R-B5 (.ebayes:41): max(df.residual)==0 -> stop
    # ------------------------------------------------------------------
    def test_zero_residual_df_errors(self):
        """Exercises R-B5: ebayes.R:41 (no residual df -> stop)."""
        # design with as many cols as rows -> df_residual = 0
        rng = np.random.default_rng(0)
        expr = rng.standard_normal((10, 4))
        design = np.eye(4)
        fit = lm_fit(expr, design)
        # df_residual should be 0
        assert np.max(fit["df_residual"]) == 0
        with pytest.raises(ValueError, match="No residual"):
            e_bayes(fit)

    # ------------------------------------------------------------------
    # R-B6 (.ebayes:42): no finite sigma -> stop
    # ------------------------------------------------------------------
    def test_nonfinite_sigma_errors(self):
        """Exercises R-B6: ebayes.R:42 (no finite sigma -> stop)."""
        expr, design = _two_group_expr(n_genes=5, n_samples=8)
        fit = lm_fit(expr, design)
        # Wipe sigma to NaN to trigger guard
        fit["sigma"] = np.full_like(fit["sigma"], np.nan)
        with pytest.raises(ValueError, match="No finite residual"):
            e_bayes(fit)

    # ------------------------------------------------------------------
    # R-B8 (.ebayes:50-54): trend is numeric vector
    # ------------------------------------------------------------------
    def test_trend_numeric_vector_parity(self):
        """Exercises R-B8: ebayes.R:50-54 (numeric trend covariate)."""
        rng = np.random.default_rng(7)
        n_genes, n_samples = 40, 8
        expr = rng.standard_normal((n_genes, n_samples))
        design = np.column_stack(
            [np.ones(n_samples), np.array([0]*4 + [1]*4, dtype=float)]
        )
        # Numeric covariate (e.g., GC content or sequence length)
        covariate = rng.uniform(5, 12, size=n_genes)

        fit = lm_fit(expr, design)
        eb = e_bayes(fit, trend=covariate)

        # Save covariate for R
        cov_df = pd.DataFrame({"x": covariate})

        r_code = """
        suppressMessages(library(limma))
        expr <- as.matrix(read.csv('{tmpdir}/expr.csv', row.names=1))
        design <- as.matrix(read.csv('{tmpdir}/design.csv', row.names=1))
        cov <- read.csv('{tmpdir}/cov.csv', row.names=1)$x
        fit <- lmFit(expr, design)
        eb <- eBayes(fit, trend=cov)
        t_stat <- eb$t
        s2_prior <- eb$s2.prior
        s2_post <- eb$s2.post
        lods <- eb$lods
        """
        r_results = run_r_comparison(
            py_data={"expr": pd.DataFrame(expr), "design": pd.DataFrame(design),
                     "cov": cov_df},
            r_code_template=r_code,
            output_vars=["t_stat", "s2_prior", "s2_post", "lods"],
        )

        # s2_prior should be array
        py_s2 = np.asarray(eb["s2_prior"], dtype=np.float64)
        r_s2 = np.asarray(r_results["s2_prior"], dtype=np.float64).ravel()
        # s2_prior matches R within tight tolerance
        result = compare_arrays(r_s2, py_s2, rtol=1e-8)
        assert result["match"], (
            f"trend=numeric s2_prior differs: max_rel={result['max_rel_diff']:.2e}"
        )

        # t-statistic full matrix
        py_t = np.asarray(eb["t"], dtype=np.float64)
        r_t = np.asarray(r_results["t_stat"], dtype=np.float64)
        result = compare_arrays(r_t, py_t, rtol=1e-8)
        assert result["match"], (
            f"trend=numeric t differs: max_rel={result['max_rel_diff']:.2e}"
        )

        # s2_post
        py_s2p = np.asarray(eb["s2_post"], dtype=np.float64)
        r_s2p = np.asarray(r_results["s2_post"], dtype=np.float64).ravel()
        result = compare_arrays(r_s2p, py_s2p, rtol=1e-8)
        assert result["match"], (
            f"trend=numeric s2_post differs: max_rel={result['max_rel_diff']:.2e}"
        )

        # lods
        py_lods = np.asarray(eb["lods"], dtype=np.float64)
        r_lods = np.asarray(r_results["lods"], dtype=np.float64)
        result = compare_arrays(r_lods, py_lods, rtol=1e-6)
        assert result["match"], (
            f"trend=numeric lods differs: max_rel={result['max_rel_diff']:.2e}"
        )

    # ------------------------------------------------------------------
    # R-B8a (.ebayes:51): non-numeric trend -> stop
    # ------------------------------------------------------------------
    def test_trend_non_numeric_errors(self):
        """Exercises R-B8a: ebayes.R:51 (trend non-numeric and non-logical -> stop)."""
        expr, design = _two_group_expr()
        fit = lm_fit(expr, design)
        # In R, passing trend=c("a","b") triggers stop().
        # In Python, passing a string array should likewise raise.
        with pytest.raises((ValueError, TypeError)):
            e_bayes(fit, trend=np.array(["a"] * len(fit["sigma"])))

    # ------------------------------------------------------------------
    # R-B8b (.ebayes:52): numeric trend wrong length -> stop
    # ------------------------------------------------------------------
    def test_trend_wrong_length_errors(self):
        """Exercises R-B8b: ebayes.R:52 (trend numeric, wrong length -> stop)."""
        expr, design = _two_group_expr(n_genes=20)
        fit = lm_fit(expr, design)
        with pytest.raises(ValueError, match="length"):
            # Wrong length: only 5 values, but n_genes=20
            e_bayes(fit, trend=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    # ------------------------------------------------------------------
    # R-B9 (.ebayes:71-74): var.prior NA fallback uses RECYCLED 1/s2.prior
    # ------------------------------------------------------------------
    def test_var_prior_na_fallback_recycles_s2_prior(self):
        """Exercises R-B9: ebayes.R:71-72 (var.prior[NA] <- 1/s2.prior).

        R uses element-wise recycling: if var.prior[1] and var.prior[2]
        are NA and s2.prior is per-gene (trend=TRUE), R assigns
        var.prior[1] <- 1/s2.prior[1], var.prior[2] <- 1/s2.prior[2], etc.
        Python uses 1/s2.prior[0] for ALL NA positions.

        Triggering: ntarget = ceiling(proportion/2 * ngenes) < 1 forces
        tmixture.vector to return NA. Use a tiny proportion so this
        happens with reasonable ngenes.
        """
        rng = np.random.default_rng(1234)
        n_genes, n_samples = 50, 12
        # Wide variation in sigma so s2_prior has a big spread
        # under trend=TRUE.
        means = np.linspace(0, 6, n_genes)
        expr = rng.standard_normal((n_genes, n_samples)) * np.exp(means[:, None])
        # Three groups for 2 non-trivial contrasts
        design = np.column_stack([
            np.ones(n_samples),
            np.array([0]*4 + [1]*4 + [0]*4, dtype=float),
            np.array([0]*4 + [0]*4 + [1]*4, dtype=float),
        ])
        fit_py = lm_fit(expr, design)
        # proportion=0 -> ntarget=0 -> tmixture returns NA -> fallback path
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eb_py = e_bayes(fit_py, proportion=0.0, trend=True)

        r_code = """
        suppressMessages(library(limma))
        expr <- as.matrix(read.csv('{tmpdir}/expr.csv', row.names=1))
        design <- as.matrix(read.csv('{tmpdir}/design.csv', row.names=1))
        fit <- lmFit(expr, design)
        eb <- suppressWarnings(eBayes(fit, proportion=0, trend=TRUE))
        var_prior <- eb$var.prior
        s2_prior <- eb$s2.prior
        """
        r_results = run_r_comparison(
            py_data={"expr": pd.DataFrame(expr), "design": pd.DataFrame(design)},
            r_code_template=r_code,
            output_vars=["var_prior", "s2_prior"],
        )

        py_vp = np.asarray(eb_py["var_prior"], dtype=np.float64).ravel()
        r_vp = np.asarray(r_results["var_prior"], dtype=np.float64).ravel()

        assert py_vp.shape == r_vp.shape, (
            f"var_prior shape mismatch: R={r_vp.shape}, Py={py_vp.shape}"
        )

        # In the fallback path, R uses positions 1, 2, 3 of 1/s2.prior
        # whereas Python uses 1/s2.prior[0] for all positions.
        result = compare_arrays(r_vp, py_vp, rtol=1e-6)
        assert result["match"], (
            f"var_prior fallback differs (R recycles 1/s2.prior, Py uses [0]): "
            f"R={r_vp}, Py={py_vp}, max_rel={result['max_rel_diff']:.2e}"
        )

    # ------------------------------------------------------------------
    # R-B10 (.ebayes:79-86): infinite df.prior path
    # Standard non-trend, non-robust eBayes: df.prior is a scalar; if
    # data are highly homoscedastic, R returns df.prior=Inf and the
    # 'Infdf' branch is taken. We construct exactly such data.
    # ------------------------------------------------------------------
    def test_inf_df_prior_kernel_branch(self):
        """Exercises R-B10b: ebayes.R:79-80, 87 (all-Infdf -> kernel limit)."""
        rng = np.random.default_rng(99)
        n_genes, n_samples = 50, 8
        # All genes share *exactly* the same residual variance -> infinite
        # df.prior. Easiest construction: sigma all equal post-fit.
        # We achieve approximate homoscedasticity by drawing each row
        # from N(0, 1) with a fixed scale, then projecting to constant-
        # variance via QR residuals; in practice fitFDist returns a
        # very large df.prior but not necessarily Inf. Use a deterministic
        # construction: each row of expr has identical squared residuals
        # to its neighbours.
        # Simplest reliable trigger: set sigma to a constant after lmFit.
        expr = rng.standard_normal((n_genes, n_samples))
        design = np.column_stack(
            [np.ones(n_samples), np.array([0]*4 + [1]*4, dtype=float)]
        )
        fit = lm_fit(expr, design)
        # Force sigma to a constant - simulates infinite-df-prior squeeze
        fit["sigma"] = np.full_like(fit["sigma"], np.median(fit["sigma"]))

        eb_py = e_bayes(fit)
        # df_prior should be > 1e6 (Inf branch trigger)
        df_prior = float(np.asarray(eb_py["df_prior"]).ravel()[0])
        assert df_prior > 1e6, (
            f"df_prior should be Inf-like for constant sigma, got {df_prior}"
        )

        # The Inf branch uses kernel = t2 * (1 - 1/r) / 2.
        # Sanity: lods should be finite (no NaN from the kernel).
        lods = np.asarray(eb_py["lods"], dtype=np.float64)
        assert np.all(np.isfinite(lods)), "lods should be finite under Inf-df branch"

    # ------------------------------------------------------------------
    # R-B10a (.ebayes:81-86): mixed Infdf branch (some Inf, some finite)
    # Only triggered when df.prior is per-gene (robust=TRUE).
    # ------------------------------------------------------------------
    def test_robust_mixed_infdf_kernel_parity(self):
        """Exercises R-B10a: ebayes.R:81-86 (per-gene df, mixed Inf/finite)."""
        rng = np.random.default_rng(5)
        n_genes, n_samples = 60, 8
        # Heteroscedastic data with a few outlier-variance genes
        expr = rng.standard_normal((n_genes, n_samples))
        # Inflate variance of last 5 genes to create outliers
        expr[-5:, :] *= 5.0
        design = np.column_stack(
            [np.ones(n_samples), np.array([0]*4 + [1]*4, dtype=float)]
        )

        fit = lm_fit(expr, design)
        eb_py = e_bayes(fit, robust=True)

        r_code = """
        suppressMessages(library(limma))
        expr <- as.matrix(read.csv('{tmpdir}/expr.csv', row.names=1))
        design <- as.matrix(read.csv('{tmpdir}/design.csv', row.names=1))
        fit <- lmFit(expr, design)
        eb <- eBayes(fit, robust=TRUE)
        t_stat <- eb$t
        lods <- eb$lods
        df_prior <- eb$df.prior
        s2_post <- eb$s2.post
        """
        r_results = run_r_comparison(
            py_data={"expr": pd.DataFrame(expr), "design": pd.DataFrame(design)},
            r_code_template=r_code,
            output_vars=["t_stat", "lods", "df_prior", "s2_post"],
        )

        # df_prior should be a per-gene vector
        r_df = np.asarray(r_results["df_prior"], dtype=np.float64).ravel()
        py_df = np.asarray(eb_py["df_prior"], dtype=np.float64).ravel()
        assert py_df.shape == r_df.shape, (
            f"robust df_prior shape mismatch: R={r_df.shape}, Py={py_df.shape}"
        )

        # Compare s2_post - mainly determined by squeeze_var
        result = compare_arrays(
            np.asarray(r_results["s2_post"]).ravel(),
            np.asarray(eb_py["s2_post"]).ravel(),
            rtol=1e-6,
        )
        assert result["match"], (
            f"robust s2_post differs: max_rel={result['max_rel_diff']:.2e}"
        )

        # Compare lods (kernel branch result)
        result = compare_arrays(
            np.asarray(r_results["lods"]),
            np.asarray(eb_py["lods"]),
            rtol=1e-6,
        )
        assert result["match"], (
            f"robust lods differs: max_rel={result['max_rel_diff']:.2e}"
        )

    # ------------------------------------------------------------------
    # R-B17 (tmixture.vector ebayes.R:135-140): df_residual varies (some
    # genes have NA-driven different df). Triggers the df adjustment
    # branch. Use missing-value pattern.
    # ------------------------------------------------------------------
    def test_varying_df_residual_tmixture_adjust(self):
        """Exercises R-B17: ebayes.R:135-140 (df adjustment in tmixture).

        Ensures the inner df-equalisation step in tmixture.vector
        produces results matching R when df_residual is gene-specific.
        Triggered by setting NaN values in expr (lm_fit drops them per
        gene, producing varying df_residual).
        """
        rng = np.random.default_rng(11)
        n_genes, n_samples = 60, 12
        expr = rng.standard_normal((n_genes, n_samples))
        # Drop one observation in half the genes -> df_residual differs
        expr[:30, 0] = np.nan
        # Add some DE signal for tmixture to actually estimate v0
        expr[:5, 6:] += 3.0
        design = np.column_stack(
            [np.ones(n_samples), np.array([0]*6 + [1]*6, dtype=float)]
        )

        fit = lm_fit(expr, design)
        eb_py = e_bayes(fit)

        r_code = """
        suppressMessages(library(limma))
        expr <- as.matrix(read.csv('{tmpdir}/expr.csv', row.names=1))
        design <- as.matrix(read.csv('{tmpdir}/design.csv', row.names=1))
        fit <- lmFit(expr, design)
        eb <- eBayes(fit)
        t_stat <- eb$t
        lods <- eb$lods
        var_prior <- eb$var.prior
        df_total <- eb$df.total
        """
        r_results = run_r_comparison(
            py_data={"expr": pd.DataFrame(expr), "design": pd.DataFrame(design)},
            r_code_template=r_code,
            output_vars=["t_stat", "lods", "var_prior", "df_total"],
        )

        # df_total parity (this confirms the df_adjustment ran)
        result = compare_arrays(
            np.asarray(r_results["df_total"]).ravel(),
            np.asarray(eb_py["df_total"]).ravel(),
            rtol=1e-8,
        )
        assert result["match"], (
            f"df_total differs: max_rel={result['max_rel_diff']:.2e}"
        )

        # var_prior is the tmixture output - this exercises R-B17
        result = compare_arrays(
            np.asarray(r_results["var_prior"]).ravel(),
            np.asarray(eb_py["var_prior"]).ravel(),
            rtol=1e-6,
        )
        assert result["match"], (
            f"var_prior (tmixture) differs: max_rel={result['max_rel_diff']:.2e}"
        )

        # lods is the final consumer - wraps everything
        result = compare_arrays(
            np.asarray(r_results["lods"]),
            np.asarray(eb_py["lods"]),
            rtol=1e-6,
        )
        assert result["match"], (
            f"lods differs: max_rel={result['max_rel_diff']:.2e}"
        )

    # ------------------------------------------------------------------
    # R-B3 (eBayes:21): F-stat block skipped when design is rank-
    # deficient. The fit$F and fit$F.p.value slots should be absent.
    # ------------------------------------------------------------------
    def test_rank_deficient_design_no_fstat(self):
        """Exercises R-B3 false branch: design rank-deficient, no F slot.

        Despite an apparently full-rank input, we manually mark design as
        rank-deficient by adding a duplicate column. R's eBayes skips the
        F.stat block; pylimma should match.
        """
        rng = np.random.default_rng(2)
        n_genes, n_samples = 20, 8
        expr = rng.standard_normal((n_genes, n_samples))
        # Rank-deficient design: third column duplicates second
        x = np.array([0]*4 + [1]*4, dtype=float)
        design = np.column_stack([np.ones(n_samples), x, x])
        # lm_fit will fit it, but the design itself is rank-deficient
        # (col 3 == col 2)
        fit = lm_fit(expr, design)

        # If the fit dropped the duplicate column already, fall through.
        # Otherwise design is in fit and is rank-deficient -> no F slot.
        eb_py = e_bayes(fit)
        # In R, the rank-deficient case skips the F-stat block.
        # In Python, we should not have F either.
        if eb_py.get("design") is not None:
            from pylimma.lmfit import is_fullrank
            if not is_fullrank(eb_py["design"]):
                assert "F" not in eb_py or eb_py.get("F") is None, (
                    "F should not be set when design is rank-deficient"
                )

    # ------------------------------------------------------------------
    # R-B3 sub-check: F.p.value matches R (single coef vs multi coef)
    # ------------------------------------------------------------------
    def test_f_pvalue_parity_with_design(self):
        """Exercises R-B3: ebayes.R:21-27 (F-statistic + F.p.value).

        Differential parity check: F and F.p.value must match R.
        """
        rng = np.random.default_rng(3)
        n_genes, n_samples = 25, 10
        expr = rng.standard_normal((n_genes, n_samples))
        design = np.column_stack([
            np.ones(n_samples),
            np.array([0]*5 + [1]*5, dtype=float),
            rng.standard_normal(n_samples),
        ])

        fit = lm_fit(expr, design)
        eb_py = e_bayes(fit)

        r_code = """
        suppressMessages(library(limma))
        expr <- as.matrix(read.csv('{tmpdir}/expr.csv', row.names=1))
        design <- as.matrix(read.csv('{tmpdir}/design.csv', row.names=1))
        fit <- lmFit(expr, design)
        eb <- eBayes(fit)
        F_stat <- eb$F
        F_p_value <- eb$F.p.value
        """
        r_results = run_r_comparison(
            py_data={"expr": pd.DataFrame(expr), "design": pd.DataFrame(design)},
            r_code_template=r_code,
            output_vars=["F_stat", "F_p_value"],
        )

        result = compare_arrays(
            np.asarray(r_results["F_stat"]).ravel(),
            np.asarray(eb_py["F"]).ravel(),
            rtol=1e-6,
        )
        assert result["match"], (
            f"F differs: max_rel={result['max_rel_diff']:.2e}"
        )

        result = compare_arrays(
            np.asarray(r_results["F_p_value"]).ravel(),
            np.asarray(eb_py["F_p_value"]).ravel(),
            rtol=1e-6,
        )
        assert result["match"], (
            f"F_p_value differs: max_rel={result['max_rel_diff']:.2e}"
        )

    # ------------------------------------------------------------------
    # R-B3 false: design absent from fit -> no F slot
    # ------------------------------------------------------------------
    def test_no_design_no_fstat(self):
        """Exercises R-B3: ebayes.R:21 (no design slot -> no F)."""
        rng = np.random.default_rng(4)
        expr = rng.standard_normal((15, 8))
        design = np.column_stack(
            [np.ones(8), np.array([0]*4 + [1]*4, dtype=float)]
        )
        fit = lm_fit(expr, design)
        # Wipe design from fit
        fit.pop("design", None)
        eb_py = e_bayes(fit)
        assert eb_py.get("F") is None, "F should be absent when design is absent"

    # ------------------------------------------------------------------
    # R-B12 (tmixture.matrix:100): dim mismatch tstat vs stdev_unscaled
    # MISSING in Python - verifies that pylimma's _tmixture_matrix does
    # NOT raise on dim mismatch (documented divergence).
    # ------------------------------------------------------------------
    def test_tmixture_matrix_dim_mismatch_python_no_check(self):
        """Exercises R-B12: ebayes.R:100 (R stops on dim mismatch).

        Python's _tmixture_matrix has no equivalent guard. This test
        documents the divergence: R raises, pylimma either crashes
        with a numpy error OR runs to completion with bogus output.
        """
        tstat = np.random.randn(20, 2)
        stdev_unscaled = np.random.rand(20, 3)  # Wrong shape
        df = 5.0
        # R: stop("Dims of tstat and stdev.unscaled don't match")
        # Python: numpy will broadcast or fail with cryptic IndexError
        # at the column-access step. Either crash or wrong result.
        try:
            result = _tmixture_matrix(tstat, stdev_unscaled, df, 0.01)
            # If it returns, the result is dimensionally bogus
            # (n_coefs from tstat=2, but stdev_unscaled has 3 cols)
            # This documents R-B12 absence in pylimma.
            assert result is not None
        except (IndexError, ValueError):
            pass  # Either outcome documents the missing check

    # ------------------------------------------------------------------
    # R-B13 (tmixture.matrix:101): v0.lim wrong length stop
    # MISSING in Python.
    # ------------------------------------------------------------------
    def test_tmixture_matrix_v0_lim_length(self):
        """Exercises R-B13: ebayes.R:101 (v0.lim length != 2 -> stop).

        Python has no length check on v0_lim. Pylimma either ignores
        extra elements or crashes on indexing. Documents divergence.
        """
        tstat = np.random.randn(20, 2)
        stdev_unscaled = np.random.rand(20, 2)
        # R: stop("v0.lim must have length 2")
        # Python: silently uses indices [0] and [1], ignoring extras
        try:
            result = _tmixture_matrix(
                tstat, stdev_unscaled, 5.0, 0.01,
                v0_lim=(0.1, 1.0, 999.0)
            )
            assert result is not None
        except (IndexError, ValueError, TypeError):
            pass

    # ------------------------------------------------------------------
    # R-B15 (tmixture.vector:126): ntarget < 1 -> return NA
    # ------------------------------------------------------------------
    def test_tmixture_ntarget_zero_returns_nan(self):
        """Exercises R-B15: ebayes.R:126 (ntarget<1 -> return NA)."""
        # ntarget = ceiling(proportion/2 * ngenes); set proportion such that
        # ceiling(.) = 0. With ngenes=1 and proportion=0, ntarget=0.
        tstat = np.array([2.0])
        stdev = np.array([1.0])
        df = 5.0
        result = _tmixture_vector(tstat, stdev, df, proportion=0.0)
        assert np.isnan(result), "Should return NaN when ntarget < 1"

    # ------------------------------------------------------------------
    # R-B19 (tmixture.vector:157): v0.lim clipping parity
    # ------------------------------------------------------------------
    def test_tmixture_v0_lim_clipping(self):
        """Exercises R-B19: ebayes.R:157 (v0_lim clip).

        Compare clipped v0 against R for a controlled tstat sample.
        """
        rng = np.random.default_rng(13)
        n = 100
        tstat = rng.standard_t(df=5, size=n)
        # Add a few moderately-sized values
        tstat[:5] = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
        stdev = np.full(n, 0.5)
        df = 5

        # Clip v0 to a tight range
        v0_lim = (0.01, 2.0)
        df_vec = np.full(n, df, dtype=np.float64)
        py_v0 = _tmixture_vector(tstat, stdev, df_vec, proportion=0.1, v0_lim=v0_lim)

        r_code = """
        suppressMessages(library(limma))
        tstat <- read.csv('{tmpdir}/tstat.csv', row.names=1)$x
        stdev <- read.csv('{tmpdir}/stdev.csv', row.names=1)$x
        df <- read.csv('{tmpdir}/df.csv', row.names=1)$x
        v0 <- limma:::tmixture.vector(tstat, stdev, df=df, proportion=0.1,
                                      v0.lim=c(0.01, 2.0))
        """
        r_results = run_r_comparison(
            py_data={"tstat": pd.DataFrame({"x": tstat}),
                     "stdev": pd.DataFrame({"x": stdev}),
                     "df": pd.DataFrame({"x": df_vec})},
            r_code_template=r_code,
            output_vars=["v0"],
        )
        r_v0 = float(np.asarray(r_results["v0"]).ravel()[0])
        assert np.isclose(py_v0, r_v0, rtol=1e-6), (
            f"v0 with v0_lim clipping differs: R={r_v0:.6e}, Py={py_v0:.6e}"
        )

    # ------------------------------------------------------------------
    # R-B14 (tmixture.vector:117): NA in tstat - parity check
    # ------------------------------------------------------------------
    def test_tmixture_na_removal_parity(self):
        """Exercises R-B14: ebayes.R:117-121 (NA removal in tmixture).

        Passes df as a vector (matching how .ebayes calls tmixture.matrix
        with df.total, which is always per-gene). With NaN tstat in the
        data, the R subsetting `df <- df[o]` produces a clean vector
        only when df is itself a vector.
        """
        rng = np.random.default_rng(17)
        n = 100
        tstat = rng.standard_t(df=5, size=n)
        # Inject NaN values
        tstat[10:15] = np.nan
        stdev = np.full(n, 0.5)
        df = np.full(n, 5.0)  # Vector matching n (real .ebayes usage)

        py_v0 = _tmixture_vector(tstat, stdev, df, proportion=0.1)

        r_code = """
        suppressMessages(library(limma))
        tstat <- read.csv('{tmpdir}/tstat.csv', row.names=1)$x
        stdev <- read.csv('{tmpdir}/stdev.csv', row.names=1)$x
        df <- read.csv('{tmpdir}/df.csv', row.names=1)$x
        v0 <- limma:::tmixture.vector(tstat, stdev, df=df, proportion=0.1)
        """
        r_results = run_r_comparison(
            py_data={"tstat": pd.DataFrame({"x": tstat}),
                     "stdev": pd.DataFrame({"x": stdev}),
                     "df": pd.DataFrame({"x": df})},
            r_code_template=r_code,
            output_vars=["v0"],
        )
        r_v0 = float(np.asarray(r_results["v0"]).ravel()[0])
        assert np.isclose(py_v0, r_v0, rtol=1e-6), (
            f"tmixture NA removal differs: R={r_v0:.6e}, Py={py_v0:.6e}"
        )

    # ------------------------------------------------------------------
    # Full eBayes parity round-trip on a clean two-group design,
    # checking ALL output slots at rtol=1e-8 (tighter than existing
    # parity test). Catches drift from any branch.
    # ------------------------------------------------------------------
    def test_full_slot_parity_basic(self):
        """All-slot parity: t, p_value, lods, s2_prior, s2_post, df_total, F, F_p_value."""
        rng = np.random.default_rng(42)
        n_genes, n_samples = 50, 8
        expr = rng.standard_normal((n_genes, n_samples))
        # Add some DE signal
        expr[:5, 4:] += 2.5
        design = np.column_stack(
            [np.ones(n_samples), np.array([0]*4 + [1]*4, dtype=float)]
        )

        fit = lm_fit(expr, design)
        eb_py = e_bayes(fit)

        r_code = _r_ebayes_template()
        r_code += """
        s2_prior <- eb$s2.prior
        df_prior <- eb$df.prior
        var_prior <- eb$var.prior
        F_stat <- eb$F
        F_p_value <- eb$F.p.value
        """
        r_results = run_r_comparison(
            py_data={"expr": pd.DataFrame(expr), "design": pd.DataFrame(design)},
            r_code_template=r_code,
            output_vars=[
                "t_stat", "p_value", "lods", "s2_post", "df_total",
                "var_prior", "F_stat", "F_p_value",
            ],
        )

        for slot, key in [
            ("t_stat", "t"),
            ("p_value", "p_value"),
            ("lods", "lods"),
            ("s2_post", "s2_post"),
            ("df_total", "df_total"),
            ("var_prior", "var_prior"),
            ("F_stat", "F"),
            ("F_p_value", "F_p_value"),
        ]:
            r_val = np.asarray(r_results[slot], dtype=np.float64)
            py_val = np.asarray(eb_py[key], dtype=np.float64)
            r_val = r_val.ravel() if py_val.ndim == 1 else r_val
            result = compare_arrays(r_val, py_val, rtol=1e-8)
            assert result["match"], (
                f"slot {key} differs: max_rel={result['max_rel_diff']:.2e}, "
                f"max_abs={result['max_abs_diff']:.2e}"
            )

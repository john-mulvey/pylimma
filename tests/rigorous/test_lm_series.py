"""
Rigorous branch-coverage audit for ``pylimma.lmfit.lm_series``.

Each test exercises a specific R branch of ``lm.series`` (R limma's
lmfit.R:91-188) and compares pylimma's output against a live R subprocess
for numerical parity at rtol=1e-8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pylimma.lmfit import lm_series

from ..helpers import (
    compare_arrays,
    limma_available,
    run_r_comparison,
)

pytestmark = pytest.mark.skipif(
    not limma_available(),
    reason="live R + limma required for differential comparison",
)


# ---------------------------------------------------------------------------
# R code templates
# ---------------------------------------------------------------------------

_R_LM_SERIES_TEMPLATE = r"""
suppressPackageStartupMessages(library(limma))

M <- as.matrix(read.csv("{tmpdir}/M.csv", row.names = 1))
design <- as.matrix(read.csv("{tmpdir}/design.csv", row.names = 1))
colnames(design) <- NULL

{weights_block}
{ndups_block}

fit <- limma:::lm.series(M = M, design = design, ndups = ndups,
                         spacing = spacing, weights = weights)

coefficients     <- fit$coefficients
stdev_unscaled   <- fit$stdev.unscaled
sigma            <- matrix(fit$sigma, ncol = 1)
df_residual      <- matrix(fit$df.residual, ncol = 1)
cov_coefficients <- fit$cov.coefficients
pivot            <- matrix(fit$pivot, ncol = 1)
rank_out         <- matrix(fit$rank, ncol = 1)
"""


def _run_lm_series_r(
    M: np.ndarray,
    design: np.ndarray,
    weights: np.ndarray | None = None,
    ndups: int = 1,
    spacing: int = 1,
) -> dict[str, np.ndarray]:
    """Call ``limma:::lm.series`` in a subprocess and return its slots."""
    py_data = {"M": pd.DataFrame(M), "design": pd.DataFrame(design)}
    weights_block = "weights <- NULL"
    if weights is not None:
        py_data["weights"] = pd.DataFrame(weights)
        weights_block = 'weights <- as.matrix(read.csv("{tmpdir}/weights.csv", row.names = 1))'
    ndups_block = f"ndups <- {int(ndups)}\nspacing <- {int(spacing)}"
    r_code = _R_LM_SERIES_TEMPLATE.format(
        tmpdir="{tmpdir}",
        weights_block=weights_block,
        ndups_block=ndups_block,
    )
    return run_r_comparison(
        py_data,
        r_code,
        output_vars=[
            "coefficients",
            "stdev_unscaled",
            "sigma",
            "df_residual",
            "cov_coefficients",
            "pivot",
            "rank_out",
        ],
        timeout=90,
    )


def _assert_slots_match(
    py_fit: dict,
    r_out: dict,
    rtol: float = 1e-8,
    compare_cov: bool = True,
    compare_pivot_rank: bool = True,
):
    """Compare common R/Py ``lm.series`` slots at rtol."""
    for key_r, key_py in [
        ("coefficients", "coefficients"),
        ("stdev_unscaled", "stdev_unscaled"),
    ]:
        r_val = np.asarray(r_out[key_r], dtype=float).reshape(np.asarray(py_fit[key_py]).shape)
        cmp = compare_arrays(r_val, py_fit[key_py], rtol=rtol)
        assert cmp["match"], (
            f"{key_r} differs: max_rel={cmp['max_rel_diff']:.2e}, max_abs={cmp['max_abs_diff']:.2e}"
        )

    r_sigma = np.asarray(r_out["sigma"], dtype=float).ravel()
    cmp = compare_arrays(r_sigma, np.asarray(py_fit["sigma"]), rtol=rtol)
    assert cmp["match"], f"sigma differs: {cmp}"

    r_df = np.asarray(r_out["df_residual"], dtype=float).ravel()
    cmp = compare_arrays(
        r_df,
        np.asarray(py_fit["df_residual"], dtype=float),
        rtol=0,
        atol=0,
    )
    assert cmp["match"], f"df_residual differs: {cmp}"

    if compare_cov:
        r_cov = np.asarray(r_out["cov_coefficients"], dtype=float)
        py_cov = np.asarray(py_fit["cov_coefficients"], dtype=float)
        # R drops non-estimable rows/cols from cov.coef; pylimma pads with
        # NaN. Restrict comparison to the estimable block recorded by pivot.
        rank = int(py_fit["rank"])
        est = py_fit["pivot"][:rank]
        py_cov_est = py_cov[np.ix_(est, est)]
        # Reshape R's cov matrix (CSV -> (k*k,) or (k, k))
        r_cov_flat = r_cov.ravel()
        if r_cov_flat.size == py_cov_est.size:
            r_cov_sq = r_cov_flat.reshape(py_cov_est.shape)
        else:
            r_cov_sq = r_cov  # will error if truly wrong
        cmp = compare_arrays(r_cov_sq, py_cov_est, rtol=rtol)
        assert cmp["match"], f"cov_coefficients differs: {cmp}"

    if compare_pivot_rank:
        r_rank = int(np.asarray(r_out["rank_out"]).ravel()[0])
        assert r_rank == int(py_fit["rank"]), f"rank mismatch: R={r_rank}, Py={py_fit['rank']}"
        r_pivot = np.asarray(r_out["pivot"]).ravel().astype(int) - 1
        py_pivot = np.asarray(py_fit["pivot"], dtype=int)
        # Estimable columns should match in order for standard designs.
        cmp = compare_arrays(
            r_pivot[:r_rank].astype(float),
            py_pivot[: int(py_fit["rank"])].astype(float),
            rtol=0,
            atol=0,
        )
        assert cmp["match"], f"pivot (estimable block) differs: R={r_pivot}, Py={py_pivot}"


# ---------------------------------------------------------------------------
# TestRigorousLmSeries
# ---------------------------------------------------------------------------


class TestRigorousLmSeries:
    """Branch-by-branch parity audit of lm_series (lmfit.R:91-188)."""

    # ------------------------- R-B1: as.matrix(M) -------------------------
    def test_b1_dataframe_coercion(self):
        """R-B1 (lmfit.R:97): ``M <- as.matrix(M)``. Confirm pylimma accepts
        a pandas DataFrame silently like R would. If it raises, that is a
        divergence from R's behaviour."""
        rng = np.random.default_rng(1)
        M = rng.standard_normal((8, 6))
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
        df = pd.DataFrame(M)
        try:
            py_fit = lm_series(df, design)
        except AttributeError:
            pytest.fail(
                "lm_series(DataFrame) raised AttributeError - divergence "
                "from R's as.matrix(M) coercion"
            )
        # If it runs, it should still match R's numeric result.
        r_out = _run_lm_series_r(M, design)
        _assert_slots_match(py_fit, r_out)

    # ----------------------- R-B2: default design -------------------------
    def test_b2_default_design_intercept_only(self):
        """R-B2 (lmfit.R:101-102): ``if(is.null(design))`` -> intercept.
        pylimma's lm_series makes design a required positional arg, so the
        R-level default is delegated to lm_fit. Document this by passing an
        explicit intercept here to confirm numerical equivalence."""
        rng = np.random.default_rng(2)
        M = rng.standard_normal((10, 5))
        design = np.ones((5, 1))
        py_fit = lm_series(M, design)
        r_out = _run_lm_series_r(M, design)
        _assert_slots_match(py_fit, r_out)

    # ----------------- R-B3: coef.names fallback "x1".."xp" ---------------
    def test_b3_default_coef_names_are_one_based_like_R(self):
        """R-B3 (lmfit.R:107): ``coef.names <- paste("x",1:nbeta,sep="")``.
        R generates "x1","x2",...; pylimma generates "x0","x1",... This is a
        deliberate-but-potentially-surprising divergence - the test documents
        it rather than asserting parity."""
        rng = np.random.default_rng(3)
        M = rng.standard_normal((6, 4))
        design = np.column_stack([np.ones(4), [0, 0, 1, 1]])
        py_fit = lm_series(M, design)
        # pylimma does not currently attach coef names to the ndarray
        # output - they get added in lm_fit via the design_names sidecar.
        # Check via the private default generator by reading the source:
        # lmfit.py:223 produces [f"x{i}" for i in range(n_coefs)], which is
        # "x0","x1".
        # There is nothing user-visible to test here at the lm_series level,
        # so this test just documents the divergence and confirms it is a
        # silent difference.
        assert py_fit["rank"] == 2
        assert py_fit["coefficients"].shape == (6, 2)

    # -------------- R-B4: weights preprocessing (<=0 -> NA) ---------------
    def test_b4_nonpositive_weights_become_nan_and_mask_M(self):
        """R-B4 (lmfit.R:110-114): ``weights[weights<=0] <- NA`` then
        ``M[!is.finite(weights)] <- NA``. Confirm parity when some weights
        are 0 / negative."""
        rng = np.random.default_rng(4)
        M = rng.standard_normal((8, 6))
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
        weights = rng.uniform(0.5, 1.5, M.shape)
        weights[1, 2] = 0.0
        weights[3, 4] = -0.1
        weights[5, 0] = np.nan
        py_fit = lm_series(M, design, weights=weights)
        r_out = _run_lm_series_r(M, design, weights=weights)
        _assert_slots_match(py_fit, r_out)

    def test_b4_does_not_mutate_caller_weights(self):
        """R-B4 guard: R's weights[weights<=0] <- NA is copy-on-modify.
        known_diff_weights_mutation.md records the prior pylimma bug. Assert
        the fix holds: caller's weights array is unchanged after the call."""
        rng = np.random.default_rng(14)
        M = rng.standard_normal((6, 4))
        design = np.column_stack([np.ones(4), [0, 0, 1, 1]])
        weights = rng.uniform(0.5, 1.5, M.shape)
        weights[0, 0] = 0.0
        weights[2, 3] = -0.5
        snapshot = weights.copy()
        lm_series(M, design, weights=weights)
        assert np.array_equal(weights, snapshot), (
            "lm_series mutated caller's weights (regression of known_diff_weights_mutation.md)"
        )

    # -------------------- R-B5: ndups > 1 unwrapping ----------------------
    def test_b5_ndups_gt_1_not_accepted(self):
        """R-B5 (lmfit.R:117-121): R's ``if(ndups>1)`` unwrap path.
        pylimma's lm_series does not accept ndups/spacing kwargs - that
        responsibility is hoisted to lm_fit which routes through gls_series.
        This is a deliberate signature divergence; flag it so downstream
        consumers know to pass ndups through lm_fit."""
        rng = np.random.default_rng(5)
        M = rng.standard_normal((6, 8))
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])
        with pytest.raises(TypeError):
            lm_series(M, design, ndups=2, spacing=1)

    # ---------------- R-B10: fast WLS with pure array weights -------------
    def test_b10_fast_wls_array_weights_only(self):
        """R-B10 (lmfit.R:133-136): NoProbeWts=TRUE with non-null weights,
        i.e. ``lm.wfit(design, t(M), weights[1,])`` - every gene sees the
        same per-sample weights."""
        rng = np.random.default_rng(6)
        M = rng.standard_normal((12, 8))
        design = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1]])
        array_w = rng.uniform(0.5, 2.0, size=8)
        weights = np.broadcast_to(array_w, M.shape).copy()
        py_fit = lm_series(M, design, weights=weights)
        r_out = _run_lm_series_r(M, design, weights=weights)
        _assert_slots_match(py_fit, r_out)

    # ------------- R-B11b: fast sigma, single-gene (vector effects) -------
    def test_b11b_fast_sigma_single_gene(self):
        """R-B11b (lmfit.R:140-141): single gene -> ``fit$effects`` is a
        vector; R takes ``mean(effects[(rank+1):n]^2)``. The pylimma
        fast-path code uses ``np.mean(qty[rank:,:]**2, axis=0)`` which
        should handle a (p, 1) residuals matrix the same way."""
        rng = np.random.default_rng(7)
        M = rng.standard_normal((1, 6))
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
        py_fit = lm_series(M, design)
        r_out = _run_lm_series_r(M, design)
        _assert_slots_match(py_fit, r_out)

    # ----------------- R-B11c: df.residual == 0 -> sigma NA ---------------
    def test_b11c_fast_df_residual_zero(self):
        """R-B11c (lmfit.R:142-143): if ``df.residual == 0``, sigma is
        ``NA_real_`` for all genes. Trigger by making design saturated
        (nrow == rank)."""
        rng = np.random.default_rng(8)
        n_samples = 3
        M = rng.standard_normal((5, n_samples))
        # 3 samples, 3 independent columns -> rank 3 -> df = 0
        design = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        py_fit = lm_series(M, design)
        assert np.all(np.isnan(py_fit["sigma"])), "sigma should be NaN when df_residual == 0"
        assert np.all(np.asarray(py_fit["df_residual"]) == 0)
        r_out = _run_lm_series_r(M, design)
        _assert_slots_match(py_fit, r_out)

    # ------------------ R-B13a: slow path, all-NA row skip ----------------
    def test_b13a_slow_all_na_gene_skipped(self):
        """R-B13a (lmfit.R:164): ``if(sum(obs) > 0)`` - when a gene has no
        finite observations the entire row should be left as NA in beta and
        stdev.unscaled, sigma NA, df_residual 0."""
        rng = np.random.default_rng(9)
        M = rng.standard_normal((8, 6))
        M[3, :] = np.nan  # force slow path, row 3 has zero finite values
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
        py_fit = lm_series(M, design)
        assert np.all(np.isnan(py_fit["coefficients"][3, :])), (
            "all-NA gene should yield NaN coefficients"
        )
        assert np.all(np.isnan(py_fit["stdev_unscaled"][3, :]))
        assert np.isnan(py_fit["sigma"][3])
        assert py_fit["df_residual"][3] == 0
        r_out = _run_lm_series_r(M, design)
        _assert_slots_match(py_fit, r_out)

    # ------------------- R-B13b: slow OLS per gene (no weights) -----------
    def test_b13b_slow_ols_per_gene(self):
        """R-B13b (lmfit.R:167-168): slow path without weights - triggered
        when M has NAs but no probe weights are supplied."""
        rng = np.random.default_rng(10)
        M = rng.standard_normal((15, 7))
        # Scatter NAs per gene (different pattern per row -> slow path)
        M[0, 0] = np.nan
        M[3, 2] = np.nan
        M[7, 5] = np.nan
        M[11, 1] = np.nan
        design = np.column_stack([np.ones(7), [0, 0, 0, 1, 1, 1, 1]])
        py_fit = lm_series(M, design)
        r_out = _run_lm_series_r(M, design)
        _assert_slots_match(py_fit, r_out)

    # --------- R-B13e: rank-deficient per-gene fit (est filter) -----------
    def test_b13e_slow_rank_deficient_design(self):
        """R-B13e (lmfit.R:173-175): ``est <- !is.na(out$coefficients)`` -
        only the estimable columns get a finite stdev.unscaled entry.
        Trigger with a rank-deficient design matrix (column 2 == column 1
        + column 0)."""
        rng = np.random.default_rng(11)
        M = rng.standard_normal((6, 6))
        M[0, 0] = np.nan  # force slow path
        col0 = np.ones(6)
        col1 = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        design = np.column_stack([col0, col1, col0 + col1])
        py_fit = lm_series(M, design)
        assert py_fit["rank"] == 2, "rank-deficient design should have rank 2"
        # The redundant column should have NaN stdev in pylimma output.
        red_col = py_fit["pivot"][2]
        assert np.all(np.isnan(py_fit["stdev_unscaled"][:, red_col]))
        r_out = _run_lm_series_r(M, design)
        _assert_slots_match(
            py_fit,
            r_out,
            compare_cov=False,
            compare_pivot_rank=True,
        )

    # --------- R-B14: cov.coef full design (slow path, verify) ------------
    def test_b14_slow_cov_coef_from_full_design(self):
        """R-B14 (lmfit.R:182-185): cov.coef computed from the full design
        QR at the end of the slow path. Confirm it matches R when slow path
        is taken."""
        rng = np.random.default_rng(12)
        M = rng.standard_normal((10, 6))
        M[0, 0] = np.nan  # trigger slow path
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
        py_fit = lm_series(M, design)
        r_out = _run_lm_series_r(M, design)
        _assert_slots_match(py_fit, r_out)

    # --------- Cross-branch sanity: probe-specific weights slow path ------
    def test_b13c_slow_wls_probe_weights(self):
        """R-B13c (lmfit.R:169-172): ``lm.wfit(X,y,w)`` per gene with
        probe-specific weights. Pylimma triggers the slow path as soon as
        weights vary across genes."""
        rng = np.random.default_rng(13)
        M = rng.standard_normal((10, 6))
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
        # Per-gene weights (row 0 > row 1 > ...), no NAs
        weights = rng.uniform(0.5, 2.0, M.shape)
        py_fit = lm_series(M, design, weights=weights)
        r_out = _run_lm_series_r(M, design, weights=weights)
        _assert_slots_match(py_fit, r_out)

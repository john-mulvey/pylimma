"""
Rigorous branch-coverage audit for ``pylimma.lmfit.gls_series``.

Each test exercises a specific R branch of ``gls.series`` (R limma's
lmfit.R:240-377) and compares pylimma's output against a live R subprocess
for numerical parity at rtol=1e-8.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pylimma.lmfit import gls_series

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
# R code template for limma:::gls.series
# ---------------------------------------------------------------------------

_R_GLS_TEMPLATE = r"""
suppressPackageStartupMessages(library(limma))

M <- as.matrix(read.csv("{tmpdir}/M.csv", row.names = 1))

{design_block}
{weights_block}
{block_block}
{ndups_block}
{corr_block}

fit <- limma:::gls.series(M = M, design = design, ndups = ndups,
                          spacing = spacing, block = block,
                          correlation = correlation, weights = weights)

coefficients     <- fit$coefficients
stdev_unscaled   <- fit$stdev.unscaled
sigma            <- matrix(fit$sigma, ncol = 1)
df_residual      <- matrix(fit$df.residual, ncol = 1)
cov_coefficients <- fit$cov.coefficients
pivot            <- matrix(fit$qr$pivot, ncol = 1)
rank_out         <- matrix(fit$qr$rank, ncol = 1)
"""

# Slow-path R returns rank/pivot at the top level (no fit$qr).
_R_GLS_SLOW_TEMPLATE = r"""
suppressPackageStartupMessages(library(limma))

M <- as.matrix(read.csv("{tmpdir}/M.csv", row.names = 1))

{design_block}
{weights_block}
{block_block}
{ndups_block}
{corr_block}

fit <- limma:::gls.series(M = M, design = design, ndups = ndups,
                          spacing = spacing, block = block,
                          correlation = correlation, weights = weights)

coefficients     <- fit$coefficients
stdev_unscaled   <- fit$stdev.unscaled
sigma            <- matrix(fit$sigma, ncol = 1)
df_residual      <- matrix(fit$df.residual, ncol = 1)
cov_coefficients <- fit$cov.coefficients
pivot            <- matrix(fit$pivot, ncol = 1)
rank_out         <- matrix(fit$rank, ncol = 1)
"""


def _run_gls_r(
    M: np.ndarray,
    design: np.ndarray | None,
    correlation: float,
    block: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    ndups: int = 1,
    spacing: int = 1,
    template: str = _R_GLS_TEMPLATE,
) -> dict[str, np.ndarray]:
    """Call ``limma:::gls.series`` in a subprocess and return its slots."""
    py_data = {"M": pd.DataFrame(M)}

    if design is None:
        design_block = "design <- NULL"
    else:
        py_data["design"] = pd.DataFrame(design)
        design_block = (
            'design <- as.matrix(read.csv("{tmpdir}/design.csv", '
            "row.names = 1)); colnames(design) <- NULL"
        )

    if weights is None:
        weights_block = "weights <- NULL"
    else:
        wts = np.asarray(weights, dtype=float)
        if wts.ndim == 1:
            # Array weights of length narrays passed as a vector.
            py_data["weights"] = pd.DataFrame(wts.reshape(-1, 1))
            weights_block = (
                'weights <- as.numeric(read.csv("{tmpdir}/weights.csv", '
                "row.names = 1)[,1])"
            )
        else:
            py_data["weights"] = pd.DataFrame(wts)
            weights_block = (
                'weights <- as.matrix(read.csv("{tmpdir}/weights.csv", '
                "row.names = 1))"
            )

    if block is None:
        block_block = "block <- NULL"
    else:
        py_data["block"] = pd.DataFrame({"block": block})
        block_block = (
            'block <- read.csv("{tmpdir}/block.csv", row.names = 1)$block'
        )

    ndups_block = f"ndups <- {int(ndups)}\nspacing <- {int(spacing)}"
    corr_block = f"correlation <- {repr(float(correlation))}"

    r_code = template.format(
        tmpdir="{tmpdir}",
        design_block=design_block,
        weights_block=weights_block,
        block_block=block_block,
        ndups_block=ndups_block,
        corr_block=corr_block,
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
        timeout=120,
    )


def _assert_slots_match(
    py_fit: dict,
    r_out: dict,
    rtol: float = 1e-8,
    compare_cov: bool = True,
):
    """Compare R/Py ``gls.series`` slots at rtol."""
    # coefficients (n_genes, n_coefs)
    r_coef = np.asarray(r_out["coefficients"], dtype=float)
    py_coef = np.asarray(py_fit["coefficients"], dtype=float)
    if r_coef.shape != py_coef.shape:
        r_coef = r_coef.reshape(py_coef.shape)
    cmp = compare_arrays(r_coef, py_coef, rtol=rtol)
    assert cmp["match"], (
        f"coefficients differ: max_rel={cmp['max_rel_diff']:.2e}, "
        f"max_abs={cmp['max_abs_diff']:.2e}"
    )

    # stdev_unscaled (n_genes, n_coefs)
    r_se = np.asarray(r_out["stdev_unscaled"], dtype=float)
    py_se = np.asarray(py_fit["stdev_unscaled"], dtype=float)
    if r_se.shape != py_se.shape:
        r_se = r_se.reshape(py_se.shape)
    cmp = compare_arrays(r_se, py_se, rtol=rtol)
    assert cmp["match"], f"stdev_unscaled differs: {cmp}"

    # sigma
    r_sigma = np.asarray(r_out["sigma"], dtype=float).ravel()
    cmp = compare_arrays(r_sigma, np.asarray(py_fit["sigma"]), rtol=rtol)
    assert cmp["match"], f"sigma differs: {cmp}"

    # df_residual
    r_df = np.asarray(r_out["df_residual"], dtype=float).ravel()
    cmp = compare_arrays(
        r_df, np.asarray(py_fit["df_residual"], dtype=float),
        rtol=0, atol=0,
    )
    assert cmp["match"], f"df_residual differs: {cmp}"

    if compare_cov:
        r_cov = np.asarray(r_out["cov_coefficients"], dtype=float)
        py_cov = np.asarray(py_fit["cov_coefficients"], dtype=float)
        rank = int(py_fit["rank"])
        est = py_fit["pivot"][:rank]
        py_cov_est = py_cov[np.ix_(est, est)]
        r_flat = r_cov.ravel()
        if r_flat.size == py_cov_est.size:
            r_cov_sq = r_flat.reshape(py_cov_est.shape)
        else:
            r_cov_sq = r_cov
        cmp = compare_arrays(r_cov_sq, py_cov_est, rtol=rtol)
        assert cmp["match"], f"cov_coefficients differs: {cmp}"


# ---------------------------------------------------------------------------
# TestRigorousGlsSeries
# ---------------------------------------------------------------------------


class TestRigorousGlsSeries:
    """Branch-by-branch parity audit of gls_series (lmfit.R:240-377)."""

    # --------------- R-B1: as.matrix(M) accepts DataFrame -----------------
    def test_b1_dataframe_input(self):
        """Exercises R-B1 (lmfit.R:247): ``M <- as.matrix(M)``.
        Confirm pylimma accepts a pandas DataFrame the way R's
        ``as.matrix`` would."""
        rng = np.random.default_rng(1)
        M = rng.standard_normal((10, 8))
        design = np.column_stack([np.ones(8), rng.standard_normal(8)])
        block = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        try:
            py_fit = gls_series(
                pd.DataFrame(M), design,
                block=block, correlation=0.4,
            )
        except (AttributeError, TypeError) as exc:
            pytest.fail(
                f"gls_series(DataFrame) raised {type(exc).__name__}: {exc} "
                f"- divergence from R's as.matrix(M) coercion"
            )
        r_out = _run_gls_r(
            M, design, correlation=0.4, block=block,
            template=_R_GLS_SLOW_TEMPLATE,
        )
        _assert_slots_match(py_fit, r_out)

    # --------------- R-B2: default design = intercept-only ----------------
    def test_b2_default_design(self):
        """Exercises R-B2 (lmfit.R:252): ``if(is.null(design)) design <-
        matrix(1, narrays, 1)``. pylimma must supply intercept-only."""
        rng = np.random.default_rng(2)
        M = rng.standard_normal((10, 8))
        block = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        py_fit = gls_series(M, design=None, block=block, correlation=0.3)
        r_out = _run_gls_r(
            M, design=None, correlation=0.3, block=block,
            template=_R_GLS_SLOW_TEMPLATE,
        )
        _assert_slots_match(py_fit, r_out)

    # --------------- R-B4: design row mismatch error ----------------------
    def test_b4_design_row_mismatch_error(self):
        """Exercises R-B4 (lmfit.R:254): ``stop("Number of rows of design
        matrix does not match number of arrays")``."""
        M = np.zeros((5, 6))
        bad_design = np.ones((5, 1))  # 5 rows, expected 6
        block = np.array([1, 1, 2, 2, 3, 3])
        with pytest.raises(ValueError, match="match number of arrays"):
            gls_series(M, bad_design, block=block, correlation=0.3)

    # --------------- R-B7: correlation>=1 error ---------------------------
    def test_b7_correlation_one_error(self):
        """Exercises R-B7 (lmfit.R:260): ``if(abs(correlation) >= 1)
        stop("correlation is 1 or -1...")``."""
        rng = np.random.default_rng(7)
        M = rng.standard_normal((4, 6))
        design = np.ones((6, 1))
        block = np.array([1, 1, 2, 2, 3, 3])
        with pytest.raises(ValueError, match="degenerate"):
            gls_series(M, design, block=block, correlation=1.0)
        with pytest.raises(ValueError, match="degenerate"):
            gls_series(M, design, block=block, correlation=-1.0)

    # --------- R-B9: ndups>=2 within-array correlated duplicates ----------
    def test_b9_ndups_path_no_weights(self):
        """Exercises R-B9 (lmfit.R:271-285): ``if(is.null(block))`` with
        ndups>=2 - within-array duplicate spots; cormatrix is the
        block-diagonal Kronecker product, design is unwrapped via
        ``design %x% rep_len(1, ndups)``."""
        rng = np.random.default_rng(9)
        ndups = 2
        n_arrays = 6
        n_genes_unique = 5
        n_spots = n_genes_unique * ndups
        M = rng.standard_normal((n_spots, n_arrays))
        design = np.column_stack([np.ones(n_arrays), [0, 0, 0, 1, 1, 1]])

        py_fit = gls_series(
            M, design, ndups=ndups, spacing=1,
            correlation=0.4, block=None,
        )
        r_out = _run_gls_r(
            M, design, correlation=0.4, block=None,
            ndups=ndups, spacing=1,
            template=_R_GLS_TEMPLATE,
        )
        _assert_slots_match(py_fit, r_out)

    def test_b9_ndups_path_with_array_weights(self):
        """Exercises R-B9 (ndups path) interaction with array-weight gate
        decision. Empirical R behaviour (verified live): in the ndups
        path, ``unwrapdups`` strips the ``arrayweights`` attribute set
        by ``asMatrixWeights``, so R takes the SLOW path despite the
        weights being constant across probes. pylimma uses a value-check
        across rows post-unwrap, which still says "array weights" because
        unwrap preserves the row-equal pattern, so pylimma takes the
        FAST path. Numerical results must still agree slot-for-slot."""
        rng = np.random.default_rng(91)
        ndups = 2
        n_arrays = 6
        n_genes_unique = 4
        n_spots = n_genes_unique * ndups
        M = rng.standard_normal((n_spots, n_arrays))
        design = np.column_stack([np.ones(n_arrays), [0, 1, 0, 1, 0, 1]])
        # Array weights: length n_arrays - R will broadcast and tag,
        # but unwrapdups will drop the tag and force slow path.
        array_w = np.array([1.0, 2.0, 1.5, 0.5, 1.2, 0.8])

        py_fit = gls_series(
            M, design, ndups=ndups, spacing=1,
            correlation=0.3, block=None,
            weights=array_w,
        )
        # R returns a slow-path list (no fit$qr) because unwrapdups drops
        # the arrayweights attribute.
        r_out = _run_gls_r(
            M, design, correlation=0.3, block=None,
            weights=array_w, ndups=ndups, spacing=1,
            template=_R_GLS_SLOW_TEMPLATE,
        )
        _assert_slots_match(py_fit, r_out)

    # --------- R-B13/R-B14: fast path, block, no weights ------------------
    def test_b14_fast_path_block_no_weights(self):
        """Exercises R-B14 fast path (lmfit.R:302-336) via the block
        path with no NAs and no weights -> NoProbeWts=TRUE."""
        rng = np.random.default_rng(14)
        n_genes = 30
        n_arrays = 12
        M = rng.standard_normal((n_genes, n_arrays))
        design = np.column_stack(
            [np.ones(n_arrays), np.tile([0, 1], n_arrays // 2)]
        )
        block = np.repeat(np.arange(6), 2)
        py_fit = gls_series(
            M, design, block=block, correlation=0.3,
        )
        r_out = _run_gls_r(
            M, design, correlation=0.3, block=block,
            template=_R_GLS_TEMPLATE,
        )
        _assert_slots_match(py_fit, r_out)

    # --------- R-B14 + array weights: NoProbeWts=TRUE ---------------------
    def test_b14_fast_path_block_with_array_weights(self):
        """Exercises R-B14 fast path with the array-weights branch
        (lmfit.R:304-307). R uses ``attr(weights, "arrayweights")`` set
        by asMatrixWeights to gate the fast path; pylimma uses a
        value-equality check across rows. This test passes a length-n
        weights vector that R will broadcast and tag."""
        rng = np.random.default_rng(141)
        n_genes = 25
        n_arrays = 12
        M = rng.standard_normal((n_genes, n_arrays))
        design = np.column_stack(
            [np.ones(n_arrays), np.tile([0, 1], n_arrays // 2)]
        )
        block = np.repeat(np.arange(6), 2)
        array_w = rng.uniform(0.5, 2.0, n_arrays)
        py_fit = gls_series(
            M, design, block=block, correlation=0.4,
            weights=array_w,
        )
        r_out = _run_gls_r(
            M, design, correlation=0.4, block=block,
            weights=array_w,
            template=_R_GLS_TEMPLATE,
        )
        _assert_slots_match(py_fit, r_out)

    # --------- R-B14 with rank-deficient design ---------------------------
    def test_b14_fast_path_rank_deficient(self):
        """Exercises R-B14 with a rank-deficient design (lmfit.R:323-326).
        ``chol2inv(fit$qr$qr, size=fit$qr$rank)`` and ``est <-
        fit$qr$pivot[1:fit$qr$rank]`` are the rank-deficient code paths."""
        rng = np.random.default_rng(142)
        n_genes = 15
        n_arrays = 10
        M = rng.standard_normal((n_genes, n_arrays))
        # Collinear columns: col 3 = col 1 + col 2
        col1 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
        col2 = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1], dtype=float)
        col3 = col1 + col2
        design = np.column_stack([np.ones(n_arrays), col1, col2, col3])
        block = np.repeat(np.arange(5), 2)
        py_fit = gls_series(
            M, design, block=block, correlation=0.25,
        )
        r_out = _run_gls_r(
            M, design, correlation=0.25, block=block,
            template=_R_GLS_TEMPLATE,
        )
        _assert_slots_match(py_fit, r_out)

    # --------- R-B14 ndups path zero-correlation reduces to OLS ----------
    def test_b14_ndups_zero_correlation(self):
        """Exercises R-B14 fast path on the ndups branch with
        correlation=0; cormatrix should reduce to identity and the
        fit must match plain OLS on the unwrapped matrix."""
        rng = np.random.default_rng(140)
        ndups = 2
        n_arrays = 6
        n_spots = 8 * ndups
        M = rng.standard_normal((n_spots, n_arrays))
        design = np.column_stack(
            [np.ones(n_arrays), [0, 0, 0, 1, 1, 1]]
        )
        py_fit = gls_series(
            M, design, ndups=ndups, spacing=1,
            correlation=0.0, block=None,
        )
        r_out = _run_gls_r(
            M, design, correlation=0.0, block=None,
            ndups=ndups, spacing=1,
            template=_R_GLS_TEMPLATE,
        )
        _assert_slots_match(py_fit, r_out)

    # --------- R-B15: slow path with NAs only (no weights) ---------------
    def test_b15_slow_path_nas_no_weights(self):
        """Exercises R-B15 slow path (lmfit.R:339-370) with NAs only;
        no probe weights. NoProbeWts is FALSE because is.finite(M) is
        FALSE."""
        rng = np.random.default_rng(15)
        n_genes = 20
        n_arrays = 12
        M = rng.standard_normal((n_genes, n_arrays))
        # Sprinkle NAs
        M[1, 0] = np.nan
        M[5, 3] = np.nan
        M[5, 4] = np.nan
        M[10, 7] = np.nan
        design = np.column_stack(
            [np.ones(n_arrays), np.tile([0, 1], n_arrays // 2)]
        )
        block = np.repeat(np.arange(6), 2)
        py_fit = gls_series(
            M, design, block=block, correlation=0.3,
        )
        r_out = _run_gls_r(
            M, design, correlation=0.3, block=block,
            template=_R_GLS_SLOW_TEMPLATE,
        )
        _assert_slots_match(py_fit, r_out)

    # --------- R-B15 slow path with probe weights ------------------------
    def test_b15_slow_path_probe_weights_no_nas(self):
        """Exercises R-B15 slow path triggered by probe-varying weights
        (NoProbeWts is FALSE because asMatrixWeights does not tag a full
        (G, N) matrix)."""
        rng = np.random.default_rng(151)
        n_genes = 18
        n_arrays = 10
        M = rng.standard_normal((n_genes, n_arrays))
        design = np.column_stack(
            [np.ones(n_arrays), np.tile([0, 1], n_arrays // 2)]
        )
        block = np.repeat(np.arange(5), 2)
        # Full (G, N) weight matrix - probe-varying.
        weights = rng.uniform(0.5, 1.5, M.shape)
        py_fit = gls_series(
            M, design, block=block, correlation=0.35,
            weights=weights,
        )
        r_out = _run_gls_r(
            M, design, correlation=0.35, block=block,
            weights=weights,
            template=_R_GLS_SLOW_TEMPLATE,
        )
        _assert_slots_match(py_fit, r_out)

    # --------- R-B15 with weights that are NaN-masked --------------------
    def test_b15_slow_path_weight_zero_masks_M(self):
        """Exercises R-B8c (lmfit.R:266-267): weights < 1e-15 mask
        ``M`` to NA. Combined with R-B15 slow-path observation logic."""
        rng = np.random.default_rng(152)
        n_genes = 12
        n_arrays = 8
        M = rng.standard_normal((n_genes, n_arrays))
        weights = rng.uniform(0.5, 1.5, M.shape)
        # Inject zero weights -> R will set M[i,j] <- NA in those slots.
        weights[3, 0] = 0.0
        weights[7, 5] = 0.0
        design = np.column_stack(
            [np.ones(n_arrays), np.tile([0, 1], n_arrays // 2)]
        )
        block = np.repeat(np.arange(4), 2)
        py_fit = gls_series(
            M, design, block=block, correlation=0.3,
            weights=weights,
        )
        r_out = _run_gls_r(
            M, design, correlation=0.3, block=block,
            weights=weights,
            template=_R_GLS_SLOW_TEMPLATE,
        )
        _assert_slots_match(py_fit, r_out)

    # --------- R-B9a warning + correlation->0 fall back -----------------
    def test_b9a_ndups_lt_2_warning(self):
        """Exercises R-B9a (lmfit.R:273-277): ``warning("No duplicates
        (ndups<2)")`` then ``ndups <- 1; correlation <- 0``."""
        rng = np.random.default_rng(90)
        n_arrays = 8
        M = rng.standard_normal((6, n_arrays))
        design = np.column_stack(
            [np.ones(n_arrays), np.tile([0, 1], n_arrays // 2)]
        )

        with pytest.warns(UserWarning, match="No duplicates"):
            py_fit = gls_series(
                M, design, ndups=1, spacing=1,
                correlation=0.5, block=None,
            )
        # After fallback: correlation forced to 0, ndups forced to 1.
        assert py_fit["ndups"] == 1
        assert py_fit["correlation"] == 0

        # R parity: r should produce the same result as ols on M.
        # Match against R's gls.series with ndups=1.
        r_out = _run_gls_r(
            M, design, correlation=0.5, block=None,
            ndups=1, spacing=1,
            template=_R_GLS_TEMPLATE,
        )
        _assert_slots_match(py_fit, r_out)

    # --------- R-B6: correlation auto-estimate via duplicateCorrelation --
    def test_b6_correlation_auto_estimate(self):
        """Exercises R-B6 (lmfit.R:259): ``if(is.null(correlation))
        correlation <- duplicateCorrelation(...)$consensus.correlation``.
        Note: pylimma's gls_series mirrors this auto-estimate path."""
        rng = np.random.default_rng(6)
        n_genes = 30
        n_arrays = 12
        M = rng.standard_normal((n_genes, n_arrays))
        design = np.column_stack(
            [np.ones(n_arrays), np.tile([0, 1], n_arrays // 2)]
        )
        block = np.repeat(np.arange(6), 2)
        # pylimma path: pass correlation=None; gls_series will call
        # duplicate_correlation internally.
        py_fit = gls_series(
            M, design, block=block, correlation=None, ndups=1,
        )
        # R path: supply the same correlation value to gls.series.
        # (duplicate_correlation parity is its own audit; we simply check
        # the full chain matches.)
        r_template = _R_GLS_TEMPLATE.replace(
            "correlation = correlation,",
            "correlation = correlation,",
        )
        r_template = r"""
suppressPackageStartupMessages(library(limma))

M <- as.matrix(read.csv("{tmpdir}/M.csv", row.names = 1))

{design_block}
{weights_block}
{block_block}
{ndups_block}

# Auto-estimate correlation just like gls.series's NULL branch.
fit <- limma:::gls.series(M = M, design = design, ndups = ndups,
                          spacing = spacing, block = block,
                          correlation = NULL, weights = weights)

coefficients     <- fit$coefficients
stdev_unscaled   <- fit$stdev.unscaled
sigma            <- matrix(fit$sigma, ncol = 1)
df_residual      <- matrix(fit$df.residual, ncol = 1)
cov_coefficients <- fit$cov.coefficients
pivot            <- matrix(fit$qr$pivot, ncol = 1)
rank_out         <- matrix(fit$qr$rank, ncol = 1)
correlation_out  <- matrix(fit$correlation, ncol = 1)
"""
        py_data = {
            "M": pd.DataFrame(M),
            "design": pd.DataFrame(design),
            "block": pd.DataFrame({"block": block}),
        }
        design_block = (
            'design <- as.matrix(read.csv("{tmpdir}/design.csv", '
            "row.names = 1)); colnames(design) <- NULL"
        )
        block_block = (
            'block <- read.csv("{tmpdir}/block.csv", row.names = 1)$block'
        )
        weights_block = "weights <- NULL"
        ndups_block = "ndups <- 1\nspacing <- 1"
        r_code = r_template.format(
            tmpdir="{tmpdir}",
            design_block=design_block,
            weights_block=weights_block,
            block_block=block_block,
            ndups_block=ndups_block,
        )
        r_out = run_r_comparison(
            py_data,
            r_code,
            output_vars=[
                "coefficients", "stdev_unscaled", "sigma",
                "df_residual", "cov_coefficients", "pivot",
                "rank_out", "correlation_out",
            ],
            timeout=180,
        )
        # Correlations should match closely.
        r_corr = float(np.asarray(r_out["correlation_out"]).ravel()[0])
        assert abs(py_fit["correlation"] - r_corr) < 1e-6, (
            f"auto-estimated correlation differs: R={r_corr}, "
            f"Py={py_fit['correlation']}"
        )
        _assert_slots_match(py_fit, r_out)

    # --------- R-B14j: df.residual broadcast across genes ----------------
    def test_b14_df_residual_per_gene(self):
        """Exercises R-B14j (lmfit.R:328): ``fit$df.residual <-
        rep_len(fit$df.residual, length.out=ngenes)``. Confirm pylimma
        returns a length-n_genes vector with all entries equal."""
        rng = np.random.default_rng(14000)
        n_genes = 12
        n_arrays = 8
        M = rng.standard_normal((n_genes, n_arrays))
        design = np.column_stack(
            [np.ones(n_arrays), np.tile([0, 1], n_arrays // 2)]
        )
        block = np.repeat(np.arange(4), 2)
        py_fit = gls_series(
            M, design, block=block, correlation=0.2,
        )
        df = np.asarray(py_fit["df_residual"], dtype=float)
        assert df.shape == (n_genes,), (
            f"df_residual shape: expected ({n_genes},), got {df.shape}"
        )
        assert np.all(df == df[0]), "fast-path df_residual should be constant"

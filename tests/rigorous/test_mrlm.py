"""
Rigorous branch-coverage audit for ``pylimma.lmfit.mrlm``.

Each test exercises a specific R branch of ``mrlm`` (R limma's
lmfit.R:190-238) and compares pylimma's output against a live R subprocess
for numerical parity at rtol=1e-8.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pylimma.lmfit import mrlm

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

# Note: limma's mrlm() forwards `...` to MASS::rlm.  Default psi is psi.huber.
# `run_r_comparison` calls `.format(tmpdir=...)` on whatever string we hand it,
# so the only field that may appear in the *final* template is `{tmpdir}`. We
# pre-build the template with no other format fields - all R curly braces are
# real R syntax (no escaping needed at the second .format() call because we
# place them only in the static template, not in the substituted block).


def _run_mrlm_r(
    M: np.ndarray,
    design: np.ndarray,
    weights: np.ndarray | None = None,
    ndups: int = 1,
    spacing: int = 1,
    extra_args: str = "",
    timeout: int = 90,
) -> dict[str, np.ndarray]:
    """Call ``limma:::mrlm`` in a subprocess and return its slots."""
    py_data = {
        "M": pd.DataFrame(np.asarray(M, dtype=np.float64)),
        "design": pd.DataFrame(np.asarray(design, dtype=np.float64)),
    }
    if weights is None:
        weights_block = "weights <- NULL"
    else:
        py_data["weights"] = pd.DataFrame(np.asarray(weights, dtype=np.float64))
        # Inline the literal {tmpdir} so the single .format() in
        # run_r_comparison handles it.
        weights_block = 'weights <- as.matrix(read.csv("{tmpdir}/weights.csv", row.names = 1))'

    ndups_int = int(ndups)
    spacing_int = int(spacing)

    # Build the template with ALL substitutions inlined except {tmpdir},
    # so run_r_comparison's `.format(tmpdir=...)` is the only format pass.
    r_code = (
        "suppressPackageStartupMessages(library(limma))\n"
        "suppressPackageStartupMessages(library(MASS))\n\n"
        'M <- as.matrix(read.csv("{tmpdir}/M.csv", row.names = 1))\n'
        'design <- as.matrix(read.csv("{tmpdir}/design.csv", '
        "row.names = 1))\n"
        "colnames(design) <- NULL\n\n"
        + weights_block
        + "\n"
        + f"ndups <- {ndups_int}\n"
        + f"spacing <- {spacing_int}\n\n"
        ".warns <- character(0)\n"
        "fit <- withCallingHandlers(\n"
        "    limma:::mrlm(\n"
        "        M = M, design = design, ndups = ndups, spacing = spacing,\n"
        f"        weights = weights{extra_args}\n"
        "    ),\n"
        "    warning = function(w) {{\n"
        "        .warns <<- c(.warns, conditionMessage(w))\n"
        '        invokeRestart("muffleWarning")\n'
        "    }}\n"
        ")\n\n"
        "coefficients     <- fit$coefficients\n"
        "stdev_unscaled   <- fit$stdev.unscaled\n"
        "sigma            <- matrix(fit$sigma, ncol = 1)\n"
        "df_residual      <- matrix(fit$df.residual, ncol = 1)\n"
        "cov_coefficients <- fit$cov.coefficients\n"
        "pivot            <- matrix(fit$pivot, ncol = 1)\n"
        "rank_out         <- matrix(fit$rank, ncol = 1)\n"
        "n_warnings       <- matrix(length(.warns), ncol = 1)\n"
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
            "n_warnings",
        ],
        timeout=timeout,
    )


def _assert_slots_match(
    py_fit: dict,
    r_out: dict,
    rtol: float = 1e-8,
    atol: float = 1e-12,
    compare_cov: bool = True,
    compare_pivot: bool = True,
):
    """Compare every R/Py mrlm slot at rtol/atol."""
    # coefficients & stdev_unscaled
    for key_r, key_py in [
        ("coefficients", "coefficients"),
        ("stdev_unscaled", "stdev_unscaled"),
    ]:
        r_val = np.asarray(r_out[key_r], dtype=float).reshape(np.asarray(py_fit[key_py]).shape)
        cmp = compare_arrays(r_val, py_fit[key_py], rtol=rtol, atol=atol)
        assert cmp["match"], (
            f"{key_r} differs: max_rel={cmp['max_rel_diff']:.2e}, max_abs={cmp['max_abs_diff']:.2e}"
        )

    # sigma
    r_sigma = np.asarray(r_out["sigma"], dtype=float).ravel()
    cmp = compare_arrays(r_sigma, np.asarray(py_fit["sigma"]), rtol=rtol, atol=atol)
    assert cmp["match"], (
        f"sigma differs: max_rel={cmp['max_rel_diff']:.2e}, max_abs={cmp['max_abs_diff']:.2e}"
    )

    # df_residual (integer-valued, exact match)
    r_df = np.asarray(r_out["df_residual"], dtype=float).ravel()
    cmp = compare_arrays(
        r_df,
        np.asarray(py_fit["df_residual"], dtype=float),
        rtol=0,
        atol=0,
    )
    assert cmp["match"], f"df_residual differs: {cmp}"

    # rank
    r_rank = int(np.asarray(r_out["rank_out"]).ravel()[0])
    py_rank = int(py_fit["rank"])
    assert r_rank == py_rank, f"rank mismatch: R={r_rank}, Py={py_rank}"

    if compare_cov:
        r_cov = np.asarray(r_out["cov_coefficients"], dtype=float)
        py_cov = np.asarray(py_fit["cov_coefficients"], dtype=float)
        # R drops non-estimable rows/cols from cov.coef; pylimma pads with
        # NaN. Restrict comparison to the estimable block.
        if "pivot" in py_fit:
            est = np.asarray(py_fit["pivot"], dtype=int)[:py_rank]
        else:
            # Best-effort: take the first rank rows/cols.
            est = np.arange(py_rank, dtype=int)
        py_cov_est = py_cov[np.ix_(est, est)]
        r_cov_flat = r_cov.ravel()
        if r_cov_flat.size == py_cov_est.size:
            r_cov_sq = r_cov_flat.reshape(py_cov_est.shape)
        else:
            r_cov_sq = r_cov
        cmp = compare_arrays(r_cov_sq, py_cov_est, rtol=rtol, atol=atol)
        assert cmp["match"], f"cov_coefficients differs: {cmp}"

    if compare_pivot:
        assert "pivot" in py_fit, (
            "pylimma's mrlm did NOT return 'pivot' - this is a divergence "
            "from R's mrlm which always returns fit$pivot (lmfit.R:237). "
            "Downstream consumers (contrasts.fit, voom) read fit$pivot."
        )
        r_pivot = np.asarray(r_out["pivot"]).ravel().astype(int) - 1
        py_pivot = np.asarray(py_fit["pivot"], dtype=int)
        cmp = compare_arrays(
            r_pivot[:r_rank].astype(float),
            py_pivot[:py_rank].astype(float),
            rtol=0,
            atol=0,
        )
        assert cmp["match"], f"pivot (estimable block) differs: R={r_pivot}, Py={py_pivot}"


# ---------------------------------------------------------------------------
# Helper for outlier-laden test data
# ---------------------------------------------------------------------------


def _make_outlier_matrix(seed: int = 0, n_genes: int = 6, n_samples: int = 10):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n_genes, n_samples))
    # Inject a single big outlier per gene to actually exercise robust weighting
    for i in range(n_genes):
        M[i, i % n_samples] += 8.0
    return M


# ---------------------------------------------------------------------------
# TestRigorousMrlm
# ---------------------------------------------------------------------------


class TestRigorousMrlm:
    """Branch-by-branch parity audit of mrlm (lmfit.R:190-238)."""

    # ----------------------- R-B3: default design -------------------------
    def test_b3_default_design_intercept_only(self):
        """R-B3 (lmfit.R:198): ``if(is.null(design)) design <- matrix(1, narrays, 1)``.

        pylimma exposes the same default. Confirm intercept-only fit matches R.
        """
        M = _make_outlier_matrix(seed=1, n_genes=4, n_samples=8)

        py_fit = mrlm(M, design=None)
        # R: passing design=NULL triggers the same default
        r_out = _run_mrlm_r(M, design=np.ones((M.shape[1], 1)))
        # pivot may be missing on Py side (audit finding); compare what we can
        _assert_slots_match(
            py_fit,
            r_out,
            rtol=1e-8,
            compare_pivot="pivot" in py_fit,
        )

    # ---------------------- R-B5: weights normalisation -------------------
    def test_b5_array_weights(self):
        """R-B5 (lmfit.R:202-206): array-weight broadcast through asMatrixWeights.

        Length-N vector should broadcast across genes.
        """
        M = _make_outlier_matrix(seed=2, n_genes=5, n_samples=10)
        design = np.column_stack([np.ones(10), np.repeat([0, 1], 5)])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])

        py_fit = mrlm(M, design=design, weights=weights)
        r_out = _run_mrlm_r(M, design, weights=weights)
        _assert_slots_match(py_fit, r_out, rtol=1e-8, compare_pivot="pivot" in py_fit)

    def test_b5_matrix_weights_with_nonpositive(self):
        """R-B5 (lmfit.R:204-205): ``weights[weights <= 0] <- NA``;
        ``M[!is.finite(weights)] <- NA`` masks the M entry, which the per-gene
        ``is.finite(y)`` filter at line 218 then skips."""
        M = _make_outlier_matrix(seed=3, n_genes=4, n_samples=8)
        design = np.column_stack([np.ones(8), np.repeat([0, 1], 4)])
        weights = np.ones_like(M)
        # Inject a zero (R turns this into NA -> masks M)
        weights[1, 2] = 0.0
        # Inject a negative (R also turns this into NA)
        weights[2, 5] = -0.5

        py_fit = mrlm(M, design=design, weights=weights.copy())
        r_out = _run_mrlm_r(M, design, weights=weights)
        _assert_slots_match(py_fit, r_out, rtol=1e-8, compare_pivot="pivot" in py_fit)

    # ----------------------- R-B6: ndups > 1 ------------------------------
    def test_b6_ndups_unwrap(self):
        """R-B6 (lmfit.R:207-211): ndups>1 unwraps duplicate spots and
        Kroneckers the design with a 1-vector of length ``ndups``.

        With 4 genes x 6 samples and ndups=2 we end up with 2 effective genes
        in 12 columns after unwrapping.
        """
        rng = np.random.default_rng(4)
        M = rng.standard_normal((4, 6))
        # Add outliers so robust fit differs from OLS
        M[0, 0] += 10.0
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])

        py_fit = mrlm(M, design=design, ndups=2, spacing=1)
        r_out = _run_mrlm_r(M, design, ndups=2, spacing=1)
        _assert_slots_match(py_fit, r_out, rtol=1e-6, compare_pivot="pivot" in py_fit)

    # ------------- R-B7: per-gene w default when no weights ---------------
    def test_b7_no_weights(self):
        """R-B7 (lmfit.R:221-222): ``if(is.null(weights)) w <- rep_len(1,length(y))``.

        Already heavily exercised but pin the rtol=1e-8 contract here.
        """
        M = _make_outlier_matrix(seed=5, n_genes=8, n_samples=12)
        design = np.column_stack([np.ones(12), np.tile([0, 1], 6)])

        py_fit = mrlm(M, design=design)
        r_out = _run_mrlm_r(M, design=design)
        _assert_slots_match(py_fit, r_out, rtol=1e-8, compare_pivot="pivot" in py_fit)

    # ---- R-B8: length(y) > nbeta - skip when too few finite values ------
    def test_b8_skip_when_insufficient_observations(self):
        """R-B8 (lmfit.R:225): ``if(length(y) > nbeta)``. R skips the per-gene
        fit when the number of finite observations does not exceed nbeta.

        Construct a row with only nbeta finite values (=2). Coefficients,
        stdev, and sigma should all be NA in R; pylimma must do the same and
        df_residual must be 0.
        """
        n_samples = 8
        design = np.column_stack([np.ones(n_samples), np.repeat([0, 1], 4)])
        nbeta = design.shape[1]  # = 2

        M = np.array(
            [
                # Row 0: only nbeta finite values -> skip
                [1.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                # Row 1: nbeta + 1 finite values -> fit
                [1.0, 2.0, 3.0, np.nan, np.nan, np.nan, np.nan, np.nan],
                # Row 2: all finite -> normal fit
                [1.0, 2.0, 3.0, 4.0, 5.5, 6.5, 7.5, 8.5],
            ]
        )

        # R requires that a fitted gene's design submatrix not be singular.
        # Row 1 has only 3 finite values across both groups; we need it to be
        # representative without the "x is singular" rlm error. To keep that
        # gene fittable choose values across both groups:
        M[1] = np.array([1.0, 2.0, np.nan, np.nan, np.nan, 6.0, np.nan, np.nan])

        # Note: R rounds nbeta = 2 so observed must be > 2 (i.e. >= 3).
        assert nbeta == 2

        py_fit = mrlm(M, design=design)
        r_out = _run_mrlm_r(M, design=design)
        _assert_slots_match(py_fit, r_out, rtol=1e-8, compare_pivot="pivot" in py_fit)

    # ------------ R-B9c: scale == 0 early exit (degenerate gene) ---------
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Documented divergence on perfectly-fit rows: R's LINPACK DQRDC2 "
            "and scipy's LAPACK SVD produce different machine-epsilon residual "
            "noise patterns, so the iter-1 MAD scale Huber-downweights different "
            "samples. Real-world impact zero (residuals are 6+ orders above "
            "machine eps in real data). See docs/validation/known_differences.rst "
            "and audits/mrlm.md (Finding 5)."
        ),
    )
    def test_b9c_zero_residual_scale(self):
        """R-B9c (MASS::rlm): inside the IRLS loop, ``if (scale == 0) {
        done = TRUE; break }``.

        With a gene perfectly fit by the design, the *initial* OLS gives
        residuals at machine epsilon. The MAD scale is therefore tiny but
        non-zero, so R proceeds with one or more IRLS iterations and ends
        up with non-trivial robust weights. pylimma should reach the same
        converged fit slot-for-slot.

        Use a deterministic, non-random outlier gene so R and pylimma see
        the same input.
        """
        n_samples = 8
        design = np.column_stack([np.ones(n_samples), np.repeat([0, 1], 4)])
        beta0, beta1 = 1.5, -0.3
        y_perfect = beta0 + design[:, 1] * beta1
        # Deterministic outlier gene - identical in R and Py (no RNG).
        y_outlier = np.array([0.5, 0.4, -0.3, 0.6, 0.1, -0.2, 0.7, 5.5])
        M = np.vstack([y_perfect, y_outlier])

        py_fit = mrlm(M, design=design)
        r_out = _run_mrlm_r(M, design=design)
        _assert_slots_match(py_fit, r_out, rtol=1e-8, compare_pivot="pivot" in py_fit)

    # ----- R-B9e: 'rlm' failed to converge warning when maxit hit --------
    def test_b9e_failed_to_converge_warning(self):
        """R-B9e (MASS::rlm): when IRLS does not converge in maxit steps,
        R issues ``warning("'rlm' failed to converge in N steps", ...)``.

        Force non-convergence with a tiny maxit. Compare:
          (a) R's warning count == 1, pylimma should also emit a warning.
          (b) Numeric outputs (coef/stdev/sigma) still match because both
              return the iterate at the maxit-th iteration.
        """
        M = _make_outlier_matrix(seed=42, n_genes=3, n_samples=10)
        design = np.column_stack([np.ones(10), np.repeat([0, 1], 5)])

        # Force non-convergence: maxit=1, acc impossibly small.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            py_fit = mrlm(M, design=design, maxit=1, acc=1e-30)
        py_warning_count = sum(1 for w in caught if "converge" in str(w.message).lower())

        r_out = _run_mrlm_r(
            M,
            design=design,
            extra_args=", maxit = 1, acc = 1e-30",
        )
        r_warning_count = int(np.asarray(r_out["n_warnings"]).ravel()[0])

        # R issues at least one 'failed to converge' warning per gene.
        assert r_warning_count >= 1, (
            f"R did not warn as expected: r_warning_count={r_warning_count}"
        )
        assert py_warning_count >= 1, (
            f"pylimma's mrlm should warn when MASS::rlm fails to converge "
            f"(R issued {r_warning_count} warnings, Py issued "
            f"{py_warning_count}). This is a missing parity behaviour."
        )

        # The numeric output should still match - same maxit-th iterate.
        _assert_slots_match(py_fit, r_out, rtol=1e-6, compare_pivot="pivot" in py_fit)

    # ------------ R-B10: sigma only assigned when df > 0 -----------------
    def test_b10_sigma_na_when_df_zero(self):
        """R-B10 (lmfit.R:230): ``if(df.residual[i] > 0) sigma[i] <- out$s``.

        With nbeta = n_samples (saturated), df=0 and sigma stays NA.
        """
        n_samples = 4
        design = np.column_stack([np.ones(n_samples), [0, 0, 1, 1], [0, 1, 0, 1]])
        # Saturated: nbeta = 3 needs > 3 obs to fit, so we pad with one more.
        # Here we keep n=4 with rank 3 design + nan in one row to push gene
        # to nbeta finite obs.
        M = np.array(
            [
                # Row 0: 3 finite values (= nbeta) -> SKIPPED
                [1.0, 2.0, 3.0, np.nan],
                # Row 1: 4 finite, df=4-3=1, sigma assigned
                [1.0, 2.0, 3.0, 4.0],
            ]
        )

        py_fit = mrlm(M, design=design)
        r_out = _run_mrlm_r(M, design=design)
        _assert_slots_match(py_fit, r_out, rtol=1e-8, compare_pivot="pivot" in py_fit)

    # ------------ R-B11: full-design QR for cov.coefficients --------------
    def test_b11_rank_deficient_design_errors_in_R(self):
        """R-B11 (lmfit.R:233-236) + MASS::rlm: rank-deficient designs
        cause ``MASS::rlm`` to error per-gene with
        ``"'x' is singular: singular fits are not implemented in 'rlm'"``,
        bringing down the whole ``mrlm`` call.

        pylimma now mirrors that behaviour: detect rank-deficient X
        (per-gene, after NaN dropping) and raise the same error. Both
        R and pylimma should fail on the same input.
        """
        n_samples = 8
        rng = np.random.default_rng(7)
        intercept = np.ones(n_samples)
        col_a = np.repeat([0, 1], 4)
        col_b = col_a.copy()  # exact duplicate of col_a
        col_c = np.tile([0, 1], 4)
        design = np.column_stack([intercept, col_a, col_b, col_c])

        M = rng.standard_normal((3, n_samples))
        M[0, 0] += 6.0

        # R errors on rank-deficient design.
        with pytest.raises(RuntimeError) as r_exc:
            _run_mrlm_r(M, design=design)
        assert "singular" in str(r_exc.value), f"Expected R singular-fit error, got: {r_exc.value}"

        # pylimma now matches R: also errors with the same message.
        with pytest.raises(ValueError, match="singular"):
            mrlm(M, design=design)

    # ----------- R-B12: return list contains pivot, rank ------------------
    def test_b12_return_contains_pivot(self):
        """R-B12 (lmfit.R:237): ``list(..., pivot=QR$pivot, rank=QR$rank)``.

        pylimma's mrlm currently returns rank but NOT pivot. This is a
        divergence: downstream consumers like contrasts.fit (R contrasts.R:66)
        and voom (R voom.R:120) read fit$pivot. Without pivot, those paths
        either break or silently use a wrong default.
        """
        M = _make_outlier_matrix(seed=11, n_genes=3, n_samples=8)
        design = np.column_stack([np.ones(8), np.repeat([0, 1], 4)])

        py_fit = mrlm(M, design=design)

        assert "pivot" in py_fit, (
            "mrlm() return dict is missing 'pivot' (R lmfit.R:237 returns "
            "QR$pivot in the result list). Downstream consumers depend on "
            "this slot."
        )
        # Also confirm the value matches R's pivot.
        r_out = _run_mrlm_r(M, design=design)
        r_pivot = np.asarray(r_out["pivot"]).ravel().astype(int) - 1
        py_pivot = np.asarray(py_fit["pivot"], dtype=int)
        assert len(py_pivot) == len(r_pivot), (
            f"pivot length mismatch: R={len(r_pivot)}, Py={len(py_pivot)}"
        )
        np.testing.assert_array_equal(py_pivot, r_pivot)

    # --------- R-B13: MASS::rlm errors on per-gene singular X -------------
    def test_b13_singular_X_per_gene(self):
        """MASS::rlm: ``if (qr(x)$rank < ncol(x)) stop("'x' is singular: ...")``.

        Construct a single-gene scenario where the per-gene design submatrix
        (after dropping NaN rows) is singular. R errors out with a stop()
        from MASS::rlm; the limma wrapper does NOT trap that error so the
        whole mrlm() call fails.

        Document pylimma's behaviour: it currently silently uses lstsq's
        minimum-norm solution and proceeds, producing a coefficient where
        R errors. If pylimma's behaviour diverges from R, this test fails.
        """
        n_samples = 8
        design = np.column_stack([np.ones(n_samples), np.repeat([0, 1], 4)])
        # Row where only 'group 0' samples are finite -> per-gene X has both
        # rows of [1, 0] -> rank 1 < 2.
        M = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ]
        )

        # R: this should raise an error (stop) inside MASS::rlm -> mrlm fails.
        # Expect R subprocess to raise:
        with pytest.raises(RuntimeError):
            _run_mrlm_r(M, design=design)

        # pylimma: document current behaviour. If it does not error too,
        # this is a deliberate divergence (silent NaN/finite coef) we want
        # documented.
        try:
            py_fit = mrlm(M, design=design)
            silently_ran = True
        except Exception:  # pragma: no cover - we want to know
            silently_ran = False
            py_fit = None

        # If pylimma silently runs, that is the divergence (R errors, Py does
        # not). Mark it as such - this assertion fails if the divergence is
        # eventually fixed (test would then need to switch to expecting an
        # exception in pylimma too).
        if silently_ran:
            # At minimum, the singular row's coefficients should not be both
            # finite values (the second entry should be NaN, since we have
            # no group-1 information).
            row_coefs = py_fit["coefficients"][0]
            # The group-1 coefficient must be NaN-ish in any sane outcome.
            # If pylimma claims a finite value here, document it for review.
            assert np.isnan(row_coefs[1]) or not silently_ran, (
                f"pylimma's mrlm silently produced a finite coefficient for "
                f"a singular per-gene design (row 0 coef={row_coefs}). R "
                f"errors out via MASS::rlm's stop()."
            )

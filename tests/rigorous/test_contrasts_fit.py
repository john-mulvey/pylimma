"""
Rigorous per-branch parity tests for pylimma.contrasts.contrasts_fit.

Each test exercises a specific R branch of contrasts.fit() in R
limma's contrasts.R.

These tests were added by a rigorous single-function audit on
2026-04-29. They run a live R subprocess via helpers.run_r_comparison
so any divergence surfaces immediately. Tests that only assert a
warning / error (not a numerical output) document the trigger
condition from R source and exercise pylimma's behaviour directly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pylimma.contrasts import contrasts_fit, make_contrasts
from pylimma.lmfit import lm_fit

from ..helpers import (
    compare_arrays,
    limma_available,
    run_r_comparison,
)


pytestmark = pytest.mark.skipif(
    not limma_available(), reason="R/limma not available"
)


def _two_group_fit(seed=0, n_genes=10, n_samples=6):
    rng = np.random.default_rng(seed)
    expr = rng.standard_normal((n_genes, n_samples))
    design = np.column_stack(
        [np.ones(n_samples), np.array([0, 0, 0, 1, 1, 1], dtype=float)]
    )
    fit = lm_fit(expr, design)
    return expr, design, fit


def _three_group_fit(seed=0, n_genes=10, n_samples=9):
    rng = np.random.default_rng(seed)
    expr = rng.standard_normal((n_genes, n_samples))
    design = np.column_stack(
        [np.ones(n_samples),
         [0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1]]
    )
    fit = lm_fit(expr, design)
    return expr, design, fit


class TestRigorousContrastsFit:
    """One test per uncovered/partial branch of contrasts.fit()."""

    # ------------------------------------------------------------------
    # R-B6: if(!is.numeric(contrasts)) stop("contrasts must be numeric matrix")
    # pylimma uses np.asarray(..., dtype=np.float64) which silently
    # coerces booleans AND numeric strings -> floats.
    # ------------------------------------------------------------------
    def test_logical_contrast_rejected_as_in_r(self):
        """Exercises R-B6: contrasts.R:31 logical input rejected.

        R: contrasts.fit(fit, matrix(c(FALSE,TRUE), nrow=2)) raises
        'contrasts must be a numeric matrix'.
        pylimma silently casts via np.asarray(..., float64).
        """
        _, _, fit = _two_group_fit()
        contr_bool = np.array([[False], [True]])
        # Either pylimma raises (matching R) -- pass; or it silently
        # accepts -- this assertion fails and documents the divergence.
        with pytest.raises((ValueError, TypeError)):
            contrasts_fit(fit, contr_bool)

    def test_string_contrast_rejected_as_in_r(self):
        """Exercises R-B6: contrasts.R:31 character input rejected.

        R: contrasts.fit(fit, matrix(c('0','1'), nrow=2)) raises
        'contrasts must be a numeric matrix'.
        pylimma's np.asarray(..., float64) silently parses string
        digits to floats.
        """
        _, _, fit = _two_group_fit()
        contr_str = np.array([["0"], ["1"]])
        with pytest.raises((ValueError, TypeError)):
            contrasts_fit(fit, contr_str)

    # ------------------------------------------------------------------
    # R-B10/B11: rename "(Intercept)" -> "Intercept" in rn[1] and cn[1].
    # R-B12: warn when row names of contrasts don't match col names of
    # coefficients. pylimma silently accepts mismatched names.
    # ------------------------------------------------------------------
    def test_warns_on_row_col_name_mismatch(self):
        """Exercises R-B12: contrasts.R:40 row/col-name mismatch warning.

        The warning fires only when both rownames(contrasts) and
        colnames(fit$coefficients) are non-NULL, so the fit must be
        constructed with a named design (R: matrix dimnames; pylimma:
        DataFrame columns).
        """
        rng = np.random.default_rng(0)
        expr = rng.standard_normal((10, 6))
        design = pd.DataFrame(
            np.column_stack([np.ones(6), np.array([0, 0, 0, 1, 1, 1], dtype=float)]),
            columns=["(Intercept)", "groupB"],
        )
        fit = lm_fit(expr, design)
        contr = pd.DataFrame(
            [[0.0], [1.0]],
            index=["Intercept", "WrongName"],
            columns=["c1"],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            contrasts_fit(fit, contr)
        assert any(
            "row names" in str(w.message).lower()
            for w in caught
        ), (
            "pylimma did not emit the row/col-name mismatch warning. "
            "R's contrasts.R:40 emits "
            '"row names of contrasts don\'t match col names of '
            'coefficients" when rn != cn (after the (Intercept) rename).'
        )

    def test_no_warning_when_intercept_renamed_matches(self):
        """Exercises R-B10/B11: silent rename of '(Intercept)'.

        R: contrast row names ('Intercept', 'groupB') with coefficient
        col names ('(Intercept)', 'groupB') match silently because
        both rn[1] and cn[1] are renamed to 'Intercept' before the
        identity check at line 40. No warning is emitted.
        """
        rng = np.random.default_rng(0)
        expr_df = pd.DataFrame(
            rng.standard_normal((5, 6)),
            columns=[f"s{i}" for i in range(6)],
        )
        design = pd.DataFrame(
            np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]]),
            columns=["(Intercept)", "groupB"],
        )
        fit = lm_fit(expr_df, design)
        contr = pd.DataFrame(
            [[0.0], [1.0]],
            index=["Intercept", "groupB"],
            columns=["c1"],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            contrasts_fit(fit, contr)
        # Filter out h5py UserWarnings, etc. - we care about
        # pylimma-originated warnings only.
        relevant = [
            w for w in caught
            if "row names" in str(w.message).lower()
            or "col names" in str(w.message).lower()
        ]
        assert not relevant, (
            "pylimma emitted a name-mismatch warning even after "
            "the '(Intercept)' rename should make rn == cn. "
            "Caught: " + str([str(w.message) for w in relevant])
        )

    # ------------------------------------------------------------------
    # R-B16 vs R-B21: orthog test runs TWICE in R, with different
    # aggregator and threshold:
    #   B16: orthog <- sum(abs(off-diag)) < 1e-12  (sum, lax)
    #   B21: orthog <- all(abs(off-diag)  < 1e-14) (max, strict)
    # B21 OVERWRITES B16 just before the new-stdev branch. The flag
    # used to decide orthog vs non-orthog stdev is the strict 1e-14
    # max-abs test, NOT B16's sum-1e-12.
    #
    # pylimma's Py-B18 only computes the B16 form and reuses it for
    # the stdev branch decision; the B21 retest is missing. The two
    # aggregator-threshold pairs disagree when an off-diag correlation
    # lies in (1e-14, 1e-12) (or when the sum of multiple tiny
    # off-diags crosses 1e-12 while their max stays under 1e-14).
    #
    # We can't easily synthesise a design whose cormatrix lands in
    # that exact band, so the test below picks a near-orthogonal
    # design and asserts pylimma's stdev_unscaled matches R's exactly.
    # If the orthog flag diverges in either direction, the orthog and
    # non-orthog branches yield slightly different stdev values; that
    # divergence is what this test catches.
    # ------------------------------------------------------------------
    def test_near_orthogonal_design_matches_r_exactly(self):
        """Exercises R-B16 + R-B21 (two-pass orthog test).

        Near-orthogonal design with mild off-diagonal correlation;
        verifies that pylimma's stdev_unscaled and cov_coefficients
        match R's bit-for-bit (rtol=1e-10) regardless of which
        orthog-branch each implementation takes.
        """
        rng = np.random.default_rng(2)
        n = 8
        X = np.column_stack([np.ones(n), [-1, -1, -1, -1, 1, 1, 1, 1]])
        # Tiny perturbation to break exact orthogonality
        X[0, 1] = X[0, 1] + 1e-7
        X = X.astype(float)
        expr = rng.standard_normal((10, n))
        fit = lm_fit(expr, X)
        contr = np.array([[0.0], [1.0]])
        f2 = contrasts_fit(fit, contr)

        r_code = """
        library(limma)
        X <- as.matrix(read.csv('{tmpdir}/X.csv', row.names=1))
        expr <- as.matrix(read.csv('{tmpdir}/expr.csv', row.names=1))
        contr <- as.matrix(read.csv('{tmpdir}/contr.csv', row.names=1))
        fit <- lmFit(expr, X)
        f2 <- contrasts.fit(fit, contr)
        coef_r <- as.matrix(f2$coefficients)
        stdev_r <- as.matrix(f2$stdev.unscaled)
        cov_r <- as.matrix(f2$cov.coefficients)
        """
        r_out = run_r_comparison(
            py_data={"X": X, "expr": expr, "contr": contr},
            r_code_template=r_code,
            output_vars=["coef_r", "stdev_r", "cov_r"],
        )
        coef_match = compare_arrays(
            r_out["coef_r"].reshape(f2["coefficients"].shape),
            f2["coefficients"], rtol=1e-10,
        )
        assert coef_match["match"], coef_match
        stdev_match = compare_arrays(
            r_out["stdev_r"].reshape(f2["stdev_unscaled"].shape),
            f2["stdev_unscaled"], rtol=1e-10,
        )
        assert stdev_match["match"], stdev_match
        cov_match = compare_arrays(
            r_out["cov_r"].reshape(f2["cov_coefficients"].shape),
            f2["cov_coefficients"], rtol=1e-10,
        )
        assert cov_match["match"], cov_match

    # ------------------------------------------------------------------
    # R-B17: rank-deficient design - R uses fit$pivot to identify the
    # `r` estimable columns and reduces contrasts/coef/stdev
    # accordingly. Throws if any non-zero entry in the contrast appears
    # in a non-estimable row.
    #
    # pylimma uses NaN-on-cov_coef-diag detection instead (Py-B17).
    # The two heuristics agree when lm_fit zeroes the non-estimable
    # column AND fills its cov-diag with NaN; they could disagree if
    # those bookkeeping conventions change.
    # ------------------------------------------------------------------
    def test_rank_deficient_estimable_contrast_matches_r(self):
        """Exercises R-B17: rank-deficient pivot reduction with an
        estimable contrast (uses only the first two coefficients).
        """
        rng = np.random.default_rng(7)
        n = 8
        X = np.column_stack([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],  # col0 + col1
        ]).astype(float)
        expr = rng.standard_normal((6, n))
        fit = lm_fit(expr, X)
        contr = np.array([[1.0], [-1.0], [0.0]])
        f2 = contrasts_fit(fit, contr)

        r_code = """
        library(limma)
        X <- as.matrix(read.csv('{tmpdir}/X.csv', row.names=1))
        expr <- as.matrix(read.csv('{tmpdir}/expr.csv', row.names=1))
        contr <- as.matrix(read.csv('{tmpdir}/contr.csv', row.names=1))
        suppressWarnings({{
          fit <- lmFit(expr, X)
          f2 <- contrasts.fit(fit, contr)
        }})
        coef_r <- as.matrix(f2$coefficients)
        stdev_r <- as.matrix(f2$stdev.unscaled)
        cov_r <- as.matrix(f2$cov.coefficients)
        """
        r_out = run_r_comparison(
            py_data={"X": X, "expr": expr, "contr": contr},
            r_code_template=r_code,
            output_vars=["coef_r", "stdev_r", "cov_r"],
        )
        cm = compare_arrays(
            r_out["coef_r"].reshape(f2["coefficients"].shape),
            f2["coefficients"], rtol=1e-8,
        )
        assert cm["match"], cm
        sm = compare_arrays(
            r_out["stdev_r"].reshape(f2["stdev_unscaled"].shape),
            f2["stdev_unscaled"], rtol=1e-8,
        )
        assert sm["match"], sm
        cv = compare_arrays(
            r_out["cov_r"].reshape(f2["cov_coefficients"].shape),
            f2["cov_coefficients"], rtol=1e-8,
        )
        assert cv["match"], cv

    def test_rank_deficient_non_estimable_contrast_raises(self):
        """Exercises R-B17 stop branch: contrasts.R:68 raises
        'trying to take contrast of non-estimable coefficient' when
        any non-zero contrast entry sits on a non-estimable row.
        """
        rng = np.random.default_rng(8)
        n = 8
        X = np.column_stack([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]).astype(float)
        expr = rng.standard_normal((4, n))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = lm_fit(expr, X)
        # Contrast tries to use non-estimable col 3
        contr_bad = np.array([[0.0], [0.0], [1.0]])
        with pytest.raises(
            (ValueError, RuntimeError),
            match=r"non[- ]?estimable",
        ):
            contrasts_fit(fit, contr_bad)

    # ------------------------------------------------------------------
    # R-B14 (empty contrasts) - already partially covered by
    # TestFinding14. We add a slot-by-slot R parity check here to lock
    # in every slot the empty path produces.
    # ------------------------------------------------------------------
    def test_empty_contrasts_matches_r_slotwise(self):
        """Exercises R-B14: contrasts.R:44 'if(!ncol(contrasts)) return(fit[,0])'.

        Live-R slot-by-slot comparison of the empty path. R's
        fit[,0] subsetting collapses every coefficient-indexed slot
        (coefficients, stdev.unscaled, cov.coefficients, contrasts).
        """
        rng = np.random.default_rng(11)
        expr = rng.standard_normal((6, 8))
        X = np.column_stack(
            [np.ones(8), [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1]]
        ).astype(float)
        fit = lm_fit(expr, X)
        empty = np.zeros((3, 0))
        f2 = contrasts_fit(fit, empty)

        # Side-by-side R run. Use direct R subprocess + dim() output
        # rather than run_r_comparison because run_r_comparison can't
        # round-trip 0-column CSVs.
        from ..helpers import run_r_code
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmp:
            np.savetxt(os.path.join(tmp, "X.csv"), X, delimiter=",")
            np.savetxt(os.path.join(tmp, "expr.csv"), expr, delimiter=",")
            r_code = f"""
            suppressMessages(library(limma))
            X <- as.matrix(read.csv('{tmp}/X.csv', header=FALSE))
            expr <- as.matrix(read.csv('{tmp}/expr.csv', header=FALSE))
            fit <- lmFit(expr, X)
            empty <- matrix(0, nrow=3, ncol=0)
            f2 <- contrasts.fit(fit, empty)
            cat('coef_dim:', dim(f2$coefficients), '\\n')
            cat('stdev_dim:', dim(f2$stdev.unscaled), '\\n')
            cat('cov_dim:', dim(f2$cov.coefficients), '\\n')
            cat('contr_dim:', dim(f2$contrasts), '\\n')
            """
            output = run_r_code(r_code)
        # Parse R output dims
        r_dims = {}
        for line in output.split("\n"):
            for key in ("coef_dim", "stdev_dim", "cov_dim", "contr_dim"):
                if line.startswith(key + ":"):
                    parts = line.split(":")[1].split()
                    r_dims[key] = tuple(int(x) for x in parts)
        assert r_dims.get("coef_dim") == f2["coefficients"].shape, (
            f"R coef shape={r_dims.get('coef_dim')} vs "
            f"Py={f2['coefficients'].shape}"
        )
        assert r_dims.get("stdev_dim") == f2["stdev_unscaled"].shape, (
            f"R stdev shape={r_dims.get('stdev_dim')} vs "
            f"Py={f2['stdev_unscaled'].shape}"
        )
        py_cov_shape = np.asarray(f2["cov_coefficients"]).shape
        assert r_dims.get("cov_dim") == py_cov_shape, (
            f"R cov shape={r_dims.get('cov_dim')} vs Py={py_cov_shape}"
        )
        py_contr_shape = np.asarray(f2["contrasts"]).shape
        assert r_dims.get("contr_dim") == py_contr_shape, (
            f"R contr shape={r_dims.get('contr_dim')} vs Py={py_contr_shape}"
        )

    # ------------------------------------------------------------------
    # R-B18: ContrastsAllZero pruning. Already covered by
    # TestFinding15ContrastsAllZeroRParity for a simple 3-coef case.
    # We add a multi-contrast test where two coefficients are all-zero
    # in every contrast column -- exercises the pruning in conjunction
    # with the orthog re-test.
    # ------------------------------------------------------------------
    def test_all_zero_pruning_multiple_unused_coefs(self):
        """Exercises R-B18 with multiple all-zero contrast rows.

        4-coef design, two contrasts using only coefs 2 and 3.
        Coefs 0 and 1 should be pruned from cov / cormatrix BEFORE
        the orthog test (R) - or the orthog flag could differ from R.
        """
        rng = np.random.default_rng(13)
        n = 12
        X = np.column_stack([
            np.ones(n),
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0] * 6 + [1] * 3 + [0] * 3,
        ]).astype(float)
        expr = rng.standard_normal((8, n))
        fit = lm_fit(expr, X)
        # Two contrasts that ignore coefs 0 and 1
        contr = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        f2 = contrasts_fit(fit, contr)

        r_code = """
        library(limma)
        X <- as.matrix(read.csv('{tmpdir}/X.csv', row.names=1))
        expr <- as.matrix(read.csv('{tmpdir}/expr.csv', row.names=1))
        contr <- as.matrix(read.csv('{tmpdir}/contr.csv', row.names=1))
        fit <- lmFit(expr, X)
        f2 <- contrasts.fit(fit, contr)
        coef_r <- as.matrix(f2$coefficients)
        stdev_r <- as.matrix(f2$stdev.unscaled)
        cov_r <- as.matrix(f2$cov.coefficients)
        """
        r_out = run_r_comparison(
            py_data={"X": X, "expr": expr, "contr": contr},
            r_code_template=r_code,
            output_vars=["coef_r", "stdev_r", "cov_r"],
        )
        cm = compare_arrays(
            r_out["coef_r"].reshape(f2["coefficients"].shape),
            f2["coefficients"], rtol=1e-10,
        )
        sm = compare_arrays(
            r_out["stdev_r"].reshape(f2["stdev_unscaled"].shape),
            f2["stdev_unscaled"], rtol=1e-10,
        )
        cv = compare_arrays(
            r_out["cov_r"].reshape(f2["cov_coefficients"].shape),
            f2["cov_coefficients"], rtol=1e-10,
        )
        assert cm["match"], cm
        assert sm["match"], sm
        assert cv["match"], cv

    # ------------------------------------------------------------------
    # R-B19/R-B26: NACoef path. Inputs with NaN coefficients ->
    # set coef to 0 and stdev to 1e30 during transform, then
    # restore NaN where transformed stdev > 1e20.
    # ------------------------------------------------------------------
    def test_na_coef_path_matches_r(self):
        """Exercises R-B19 + R-B26: NA coefficient masking and restore.

        NaN entries in fit$coefficients (e.g. from probe-wise weighting
        with all-zero weights for some samples) must round-trip
        through contrasts.fit() unchanged where the contrast loads on
        the NA coefficient.
        """
        rng = np.random.default_rng(14)
        n = 6
        X = np.column_stack([np.ones(n), [0, 0, 0, 1, 1, 1]]).astype(float)
        expr = rng.standard_normal((5, n))
        # Hand-inject NaN into one coefficient row by zeroing all
        # weights for that gene's samples -> lmFit produces NaN coef.
        weights = np.ones((5, n))
        weights[2, :] = 0.0  # gene 2 has all-zero weights
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = lm_fit(expr, X, weights=weights)

        contr = np.array([[0.0], [1.0]])
        f2 = contrasts_fit(fit, contr)

        r_code = """
        library(limma)
        X <- as.matrix(read.csv('{tmpdir}/X.csv', row.names=1))
        expr <- as.matrix(read.csv('{tmpdir}/expr.csv', row.names=1))
        weights <- as.matrix(read.csv('{tmpdir}/weights.csv', row.names=1))
        contr <- as.matrix(read.csv('{tmpdir}/contr.csv', row.names=1))
        suppressWarnings({{
          fit <- lmFit(expr, X, weights=weights)
          f2 <- contrasts.fit(fit, contr)
        }})
        coef_r <- as.matrix(f2$coefficients)
        stdev_r <- as.matrix(f2$stdev.unscaled)
        """
        r_out = run_r_comparison(
            py_data={
                "X": X, "expr": expr,
                "weights": weights, "contr": contr,
            },
            r_code_template=r_code,
            output_vars=["coef_r", "stdev_r"],
        )
        # NaN pattern must agree
        py_coef = f2["coefficients"]
        r_coef = r_out["coef_r"].reshape(py_coef.shape)
        assert np.array_equal(np.isnan(py_coef), np.isnan(r_coef)), (
            "NaN pattern divergence: "
            f"R NaN at {np.where(np.isnan(r_coef))}, "
            f"Py NaN at {np.where(np.isnan(py_coef))}"
        )
        cm = compare_arrays(r_coef, py_coef, rtol=1e-10)
        assert cm["match"], cm
        py_stdev = f2["stdev_unscaled"]
        r_stdev = r_out["stdev_r"].reshape(py_stdev.shape)
        assert np.array_equal(
            np.isnan(py_stdev), np.isnan(r_stdev)
        ), "stdev NaN pattern divergence"

    # ------------------------------------------------------------------
    # R-B15 (cov.coefficients == NULL warning + diag construction).
    # Tests the fallback path when fit$cov.coefficients is missing.
    # ------------------------------------------------------------------
    def test_missing_cov_coefficients_warns_and_builds_diag(self):
        """Exercises R-B15: contrasts.R:48-53 missing cov path.

        R: warning("cov.coefficients not found in fit -- assuming
        coefficients are orthogonal"), constructs diag from
        colMeans(stdev.unscaled^2). pylimma must do the same.
        """
        rng = np.random.default_rng(15)
        n = 6
        X = np.column_stack([np.ones(n), [0, 0, 0, 1, 1, 1]]).astype(float)
        expr = rng.standard_normal((5, n))
        fit = lm_fit(expr, X)
        # Strip cov_coefficients to force the fallback branch
        fit_stripped = dict(fit)
        fit_stripped.pop("cov_coefficients", None)
        contr = np.array([[0.0], [1.0]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            f2 = contrasts_fit(fit_stripped, contr)
        relevant = [
            w for w in caught
            if "cov_coefficients" in str(w.message).lower()
            or "cov.coefficients" in str(w.message).lower()
            or "orthogonal" in str(w.message).lower()
        ]
        assert relevant, (
            "pylimma did not warn when cov_coefficients is absent. "
            "R's contrasts.R:49 emits 'cov.coefficients not found in "
            "fit - assuming coefficients are orthogonal'."
        )

        # Compare to R running the same fallback (via a fit with no
        # cov.coefficients - simplest is to manually set it NULL).
        r_code = """
        library(limma)
        X <- as.matrix(read.csv('{tmpdir}/X.csv', row.names=1))
        expr <- as.matrix(read.csv('{tmpdir}/expr.csv', row.names=1))
        contr <- as.matrix(read.csv('{tmpdir}/contr.csv', row.names=1))
        fit <- lmFit(expr, X)
        fit$cov.coefficients <- NULL
        suppressWarnings({{
          f2 <- contrasts.fit(fit, contr)
        }})
        coef_r <- as.matrix(f2$coefficients)
        stdev_r <- as.matrix(f2$stdev.unscaled)
        """
        r_out = run_r_comparison(
            py_data={"X": X, "expr": expr, "contr": contr},
            r_code_template=r_code,
            output_vars=["coef_r", "stdev_r"],
        )
        cm = compare_arrays(
            r_out["coef_r"].reshape(f2["coefficients"].shape),
            f2["coefficients"], rtol=1e-10,
        )
        assert cm["match"], cm
        sm = compare_arrays(
            r_out["stdev_r"].reshape(f2["stdev_unscaled"].shape),
            f2["stdev_unscaled"], rtol=1e-10,
        )
        assert sm["match"], sm

    # ------------------------------------------------------------------
    # R-B2: coefficients= path. R does fit[,coefficients] which is a
    # different code path (subsetting.R:107-185) including:
    #   - R-B2c: assign object$contrasts <- diag(ncol) with coef-name
    #     dimnames (only when contrasts was previously NULL).
    #   - R-B2d: subset all IJ/IX/JX slots (coef, stdev, t, p_value,
    #     lods, weights, genes).
    #   - R-B2e: subset cov.coefficients[jj,jj], var.prior[j].
    #   - R-B2f: regenerate F if F was previously set and j supplied.
    #
    # pylimma converts coefficients= to a contrast matrix and runs the
    # full contrast pipeline. The numeric output should still match R
    # but the resulting fit's contrasts slot has different content
    # (selected_names vs identity diag with all coef-name dimnames).
    # ------------------------------------------------------------------
    def test_coefficients_param_int_matches_r_fit_subset(self):
        """Exercises R-B2: coefficients=int path.

        R: contrasts.fit(fit, coefficients=2) returns fit[,2]. pylimma
        should produce numerically equivalent coefficients/stdev.
        """
        rng = np.random.default_rng(20)
        n = 9
        X = np.column_stack([
            np.ones(n),
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
        ]).astype(float)
        expr = rng.standard_normal((6, n))
        fit = lm_fit(expr, X)
        # Pylimma uses 0-based, R uses 1-based: ask for the second
        # coefficient (index 1 in pylimma, 2 in R).
        f2 = contrasts_fit(fit, coefficients=[1])

        r_code = """
        library(limma)
        X <- as.matrix(read.csv('{tmpdir}/X.csv', row.names=1))
        expr <- as.matrix(read.csv('{tmpdir}/expr.csv', row.names=1))
        fit <- lmFit(expr, X)
        f2 <- contrasts.fit(fit, coefficients=2)
        coef_r <- as.matrix(f2$coefficients)
        stdev_r <- as.matrix(f2$stdev.unscaled)
        cov_r <- as.matrix(f2$cov.coefficients)
        """
        r_out = run_r_comparison(
            py_data={"X": X, "expr": expr},
            r_code_template=r_code,
            output_vars=["coef_r", "stdev_r", "cov_r"],
        )
        cm = compare_arrays(
            r_out["coef_r"].reshape(f2["coefficients"].shape),
            f2["coefficients"], rtol=1e-10,
        )
        sm = compare_arrays(
            r_out["stdev_r"].reshape(f2["stdev_unscaled"].shape),
            f2["stdev_unscaled"], rtol=1e-10,
        )
        cv = compare_arrays(
            r_out["cov_r"].reshape(f2["cov_coefficients"].shape),
            f2["cov_coefficients"], rtol=1e-10,
        )
        assert cm["match"], cm
        assert sm["match"], sm
        assert cv["match"], cv

    # ------------------------------------------------------------------
    # R-B6 (numeric check) corollary: numpy float32 input. R is silent
    # on dtype, but the contract is "numeric matrix". Must produce same
    # answer as float64 input.
    # ------------------------------------------------------------------
    def test_float32_contrast_matches_float64(self):
        """Exercises Py-B9 dtype coercion (no R counterpart).

        pylimma forces dtype=float64. Float32 input should give the
        same answer as float64 - if not, dtype-dependent precision
        errors are leaking through.
        """
        _, _, fit = _two_group_fit(seed=21)
        contr_f64 = np.array([[0.0], [1.0]], dtype=np.float64)
        contr_f32 = np.array([[0.0], [1.0]], dtype=np.float32)
        f2_64 = contrasts_fit(fit, contr_f64)
        f2_32 = contrasts_fit(fit, contr_f32)
        np.testing.assert_allclose(
            f2_64["coefficients"], f2_32["coefficients"], rtol=0
        )

    # ------------------------------------------------------------------
    # R-B7: NA in contrasts. Already exercised by tests/test_contrasts
    # but never against live R. Verify both paths reject identically.
    # ------------------------------------------------------------------
    def test_nan_in_contrasts_rejected(self):
        """Exercises R-B7: contrasts.R:32 'NAs not allowed in contrasts'."""
        _, _, fit = _two_group_fit()
        contr_nan = np.array([[0.0], [np.nan]])
        with pytest.raises((ValueError, RuntimeError), match=r"[Nn][Aa]"):
            contrasts_fit(fit, contr_nan)

    # ------------------------------------------------------------------
    # R-B9: row count mismatch. Already covered, but confirm.
    # ------------------------------------------------------------------
    def test_wrong_nrow_rejected(self):
        """Exercises R-B9: contrasts.R:34 nrow(contrasts) != ncoef."""
        _, _, fit = _two_group_fit()
        # fit has 2 coefs - pass 3-row contrast
        contr_bad = np.array([[1.0], [0.0], [0.0]])
        with pytest.raises(ValueError, match=r"rows"):
            contrasts_fit(fit, contr_bad)

    # ------------------------------------------------------------------
    # R-B5: strip t/p.value/lods/F/F.p.value when called on an
    # already-eBayes'd fit. Already covered by test_contrasts.py
    # test_removes_test_statistics, but only checks t and p_value.
    # We add the F and lods slots.
    # ------------------------------------------------------------------
    def test_strips_all_test_statistics(self):
        """Exercises R-B5: contrasts.R:21-25 strips t/p.value/lods/F/F.p.value."""
        _, _, fit = _two_group_fit(seed=22)
        # Inject all five test-statistic slots
        fit["t"] = np.zeros_like(fit["coefficients"])
        fit["p_value"] = np.zeros_like(fit["coefficients"])
        fit["lods"] = np.zeros_like(fit["coefficients"])
        fit["F"] = np.zeros(fit["coefficients"].shape[0])
        fit["F_p_value"] = np.zeros(fit["coefficients"].shape[0])
        f2 = contrasts_fit(fit, np.array([[0.0], [1.0]]))
        for k in ("t", "p_value", "lods", "F", "F_p_value"):
            assert k not in f2, (
                f"{k} not stripped from fit after contrasts_fit"
            )

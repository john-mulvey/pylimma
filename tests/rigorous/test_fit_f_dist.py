"""
Rigorous per-branch parity tests for pylimma.squeeze_var.fit_f_dist.

Each test exercises a specific R branch of fitFDist() in R limma's
fitFDist.R.

These tests were added by a rigorous single-function audit. They are
intentionally tight (rtol=1e-8 where possible) and run a live R subprocess
via helpers.run_r_comparison so any regression surfaces immediately.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pylimma.squeeze_var import fit_f_dist

from ..helpers import (
    compare_arrays,
    limma_available,
    run_r_comparison,
)


pytestmark = pytest.mark.skipif(
    not limma_available(), reason="R/limma not available"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _r_fit_f_dist(x, df1, covariate=None, timeout=60):
    """Run R fitFDist via subprocess, returning {scale, df2}.

    Always passes x as a vector through CSV. df1 may be a scalar or a
    vector; encoded into the R call accordingly. covariate may be None or a
    vector.
    """
    py_data = {"x": np.asarray(x, dtype=float)}

    if np.isscalar(df1) or (np.asarray(df1).ndim == 0):
        df1_str = f"df1={float(df1)}"
        df1_csv = ""
    else:
        py_data["df1"] = np.asarray(df1, dtype=float)
        df1_str = "df1=df1_input[,1]"
        df1_csv = (
            "df1_input <- as.matrix(read.csv('{tmpdir}/df1.csv', row.names=1))\n"
        )

    if covariate is None:
        cov_str = "covariate=NULL"
        cov_csv = ""
    else:
        py_data["cov"] = np.asarray(covariate, dtype=float)
        cov_str = "covariate=cov_input[,1]"
        cov_csv = (
            "cov_input <- as.matrix(read.csv('{tmpdir}/cov.csv', row.names=1))\n"
        )

    code = f"""
    suppressMessages(library(limma))
    x <- as.matrix(read.csv('{{tmpdir}}/x.csv', row.names=1))[,1]
    {df1_csv}
    {cov_csv}
    fit <- fitFDist(x, {df1_str}, {cov_str})
    scale <- fit$scale
    df2 <- fit$df2
    """

    return run_r_comparison(
        py_data=py_data,
        r_code_template=code,
        output_vars=["scale", "df2"],
        timeout=timeout,
    )


def _atleast_1d(x):
    return np.atleast_1d(np.asarray(x, dtype=float))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRigorousFitFDist:
    """One test per uncovered/partial R branch of fitFDist."""

    # ------------------------------------------------------------------
    # R-B1 (fitFDist.R:9): n==0 -> return list(scale=NA, df2=NA)
    # ------------------------------------------------------------------
    def test_n_zero_returns_nan(self):
        """Exercises R-B1: n=0 returns NaN scale and NaN df2 (no R call needed)."""
        py = fit_f_dist(np.array([]), df1=5)
        assert np.isnan(py["scale"])
        assert np.isnan(py["df2"])

    # ------------------------------------------------------------------
    # R-B2 (fitFDist.R:10): n==1 -> return list(scale=x, df2=0)
    # ------------------------------------------------------------------
    def test_n_one_returns_x_and_zero(self):
        """Exercises R-B2: n=1 returns scale=x[0], df2=0."""
        py = fit_f_dist(np.array([0.42]), df1=5)
        assert py["scale"] == 0.42
        assert py["df2"] == 0.0
        # And against R directly
        r = _r_fit_f_dist(np.array([0.42]), df1=5)
        np.testing.assert_allclose(_atleast_1d(r["scale"]),
                                   _atleast_1d(py["scale"]), rtol=1e-15)
        np.testing.assert_allclose(_atleast_1d(r["df2"]),
                                   _atleast_1d(py["df2"]), atol=1e-15)

    # ------------------------------------------------------------------
    # R-B3a (fitFDist.R:14-19): scalar df1 invalid -> return NA/NA
    # In R, this is checked BEFORE expansion. In Python after broadcast.
    # ------------------------------------------------------------------
    def test_scalar_df1_zero_returns_nan(self):
        """Exercises R-B3a: scalar df1<=1e-15 returns NaN/NaN.

        R: line 14 `if(length(df1)==1L) { if(!ok) return(NaN/NaN) }`. R
        checks the SCALAR before expanding. pylimma broadcasts first then
        masks per-element, which can collapse to nok=0 instead of the
        explicit NaN return. Compare to R's behaviour directly.
        """
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        # df1=0 fails the `df1 > 1e-15` check
        r = _r_fit_f_dist(x, df1=0.0)
        py = fit_f_dist(x, df1=0.0)
        # R returns NaN scale, NaN df2
        np.testing.assert_array_equal(np.isnan(_atleast_1d(r["scale"])),
                                      np.isnan(_atleast_1d(py["scale"])))
        np.testing.assert_array_equal(np.isnan(_atleast_1d(r["df2"])),
                                      np.isnan(_atleast_1d(py["df2"])))

    def test_scalar_df1_negative_returns_nan(self):
        """Exercises R-B3a: scalar df1<0 returns NaN/NaN."""
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        r = _r_fit_f_dist(x, df1=-1.0)
        py = fit_f_dist(x, df1=-1.0)
        np.testing.assert_array_equal(np.isnan(_atleast_1d(r["scale"])),
                                      np.isnan(_atleast_1d(py["scale"])))
        np.testing.assert_array_equal(np.isnan(_atleast_1d(r["df2"])),
                                      np.isnan(_atleast_1d(py["df2"])))

    def test_scalar_df1_inf_returns_nan(self):
        """Exercises R-B3a: scalar df1=Inf returns NaN/NaN.

        R checks `is.finite(df1)` -- Inf fails. pylimma uses the same
        check after broadcast -- ok mask is all False; nok=0.

        Uses an inline R script to avoid CSV-passing the Inf token.
        """
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        # Use inline R because the CSV helper does not preserve "Inf".
        from ..helpers import run_r_code
        np.savetxt("/tmp/_x.csv", x)
        code = """
        suppressMessages(library(limma))
        x <- scan('/tmp/_x.csv', quiet=TRUE)
        fit <- fitFDist(x, df1=Inf, covariate=NULL)
        cat(is.na(fit$scale), is.na(fit$df2), sep=' ')
        """
        out = run_r_code(code).strip()
        r_scale_na, r_df2_na = [s == "TRUE" for s in out.split()]

        py = fit_f_dist(x, df1=np.inf)
        py_scale_na = np.isnan(py["scale"]) if np.isscalar(py["scale"]) else np.all(np.isnan(_atleast_1d(py["scale"])))
        py_df2_na = np.isnan(py["df2"])

        assert r_scale_na == py_scale_na, (
            f"R scale_NA={r_scale_na}, Py scale_NA={py_scale_na}"
        )
        assert r_df2_na == py_df2_na, (
            f"R df2_NA={r_df2_na}, Py df2_NA={py_df2_na}"
        )

    # ------------------------------------------------------------------
    # R-B3b (fitFDist.R:21): vector df1 length mismatch -> stop
    # ------------------------------------------------------------------
    def test_vector_df1_length_mismatch_raises(self):
        """Exercises R-B3b: vector df1 with len != n must error.

        R: `if(length(df1) != n) stop("x and df1 have different lengths")`.
        pylimma silently broadcasts in Py-B3 then may fail later during
        masking. Verify pylimma raises an error like R does.
        """
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        df1 = np.array([5.0, 6.0, 7.0])  # length 3, x has length 5

        # R must raise
        r_failed = False
        try:
            _r_fit_f_dist(x, df1=df1)
        except RuntimeError:
            r_failed = True
        assert r_failed, "expected R to error on length mismatch"

        # pylimma should also error
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            fit_f_dist(x, df1=df1)

    # ------------------------------------------------------------------
    # R-B4/R-B14 (fitFDist.R:25, 85-87): null covariate path; full math match
    # ------------------------------------------------------------------
    def test_null_covariate_full_math(self):
        """Exercises R-B14: null covariate path scale & df2 match R tightly.

        Uses constant df1, no missing values, no zero variances - the
        cleanest path. Tolerance 1e-8 because both implementations follow
        the same formulae.
        """
        rng = np.random.default_rng(101)
        n = 200
        # Generate well-behaved variances
        x = 0.3 * rng.chisquare(5, size=n) / 5
        py = fit_f_dist(x, df1=5)
        r = _r_fit_f_dist(x, df1=5)

        np.testing.assert_allclose(_atleast_1d(r["scale"]),
                                   _atleast_1d(py["scale"]), rtol=1e-8)
        np.testing.assert_allclose(_atleast_1d(r["df2"]),
                                   _atleast_1d(py["df2"]), rtol=1e-8)

    def test_null_covariate_vector_df1_full_match(self):
        """Exercises R-B14 with vector df1: math should match exactly.

        Vector df1 changes the e = log(x) + logmdigamma(df1/2) computation
        and the trigamma correction.
        """
        rng = np.random.default_rng(102)
        n = 100
        df1 = np.full(n, 5.0)  # constant but as vector
        df1[:30] = 4.0
        x = 0.3 * rng.chisquare(df1) / df1  # non-central F-like
        py = fit_f_dist(x, df1=df1)
        r = _r_fit_f_dist(x, df1=df1)

        np.testing.assert_allclose(_atleast_1d(r["scale"]),
                                   _atleast_1d(py["scale"]), rtol=1e-8)
        np.testing.assert_allclose(_atleast_1d(r["df2"]),
                                   _atleast_1d(py["df2"]), rtol=1e-8)

    # ------------------------------------------------------------------
    # R-B5 (fitFDist.R:28): covariate length mismatch -> stop
    # ------------------------------------------------------------------
    def test_covariate_length_mismatch(self):
        """Exercises R-B5: mismatched covariate length raises."""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        cov = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="same length"):
            fit_f_dist(x, df1=5, covariate=cov)

    # ------------------------------------------------------------------
    # R-B6 (fitFDist.R:29): NA in covariate -> stop
    # ------------------------------------------------------------------
    def test_covariate_na_raises(self):
        """Exercises R-B6: NA covariate raises."""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        cov = np.array([1.0, np.nan, 3.0, 4.0])
        with pytest.raises(ValueError, match="NA covariate"):
            fit_f_dist(x, df1=5, covariate=cov)

    # ------------------------------------------------------------------
    # R-B7a (fitFDist.R:30-35): some inf, some finite -> replace +/- Inf
    # ------------------------------------------------------------------
    def test_covariate_some_infinite(self):
        """Exercises R-B7a: covariate with mix of finite and infinite values.

        R replaces -Inf with min(finite)-1 and +Inf with max(finite)+1.
        Tests pylimma matches this expansion.
        """
        rng = np.random.default_rng(103)
        n = 50
        # Build covariate with a couple of infinities
        cov_finite_part = np.linspace(2, 10, n - 2)
        cov = np.concatenate([[float("-inf"), float("inf")], cov_finite_part])
        # x and df1 finite
        x = 0.3 * rng.chisquare(5, size=n) / 5

        py = fit_f_dist(x, df1=5, covariate=cov)
        r = _r_fit_f_dist(x, df1=5, covariate=cov)

        np.testing.assert_allclose(_atleast_1d(r["scale"]),
                                   _atleast_1d(py["scale"]), rtol=1e-5)
        np.testing.assert_allclose(_atleast_1d(r["df2"]),
                                   _atleast_1d(py["df2"]), rtol=1e-5)

    # ------------------------------------------------------------------
    # R-B7b (fitFDist.R:36-37): all infinite -> covariate <- sign(covariate)
    # ------------------------------------------------------------------
    def test_covariate_all_infinite(self):
        """Exercises R-B7b: all covariate values infinite -> sign-coded.

        R falls back to `covariate <- sign(covariate)`. With +/-Inf values
        this gives +1 or -1.
        """
        rng = np.random.default_rng(104)
        n = 30
        cov = np.where(rng.random(n) > 0.5, np.inf, -np.inf)
        x = 0.3 * rng.chisquare(5, size=n) / 5

        # Let R determine the result; just compare
        py = fit_f_dist(x, df1=5, covariate=cov)
        r = _r_fit_f_dist(x, df1=5, covariate=cov)

        np.testing.assert_allclose(_atleast_1d(r["scale"]),
                                   _atleast_1d(py["scale"]), rtol=1e-5)
        np.testing.assert_allclose(_atleast_1d(r["df2"]),
                                   _atleast_1d(py["df2"]), rtol=1e-5)

    # ------------------------------------------------------------------
    # R-B8 + R-B10 (fitFDist.R:43, 47-54): notallok subsetting
    # x with NaN/Inf/negative entries should be filtered out.
    # ------------------------------------------------------------------
    def test_notallok_subsetting_x(self):
        """Exercises R-B8/R-B10: invalid x entries are filtered out.

        Inserts NaN, +Inf and a negative value into x. Both R and pylimma
        should drop these from the moment estimation.
        """
        rng = np.random.default_rng(105)
        n = 100
        x = 0.3 * rng.chisquare(5, size=n) / 5
        # Inject bad entries
        x[0] = np.nan
        x[1] = np.inf
        x[2] = -0.5

        py = fit_f_dist(x, df1=5)
        r = _r_fit_f_dist(x, df1=5)

        np.testing.assert_allclose(_atleast_1d(r["scale"]),
                                   _atleast_1d(py["scale"]), rtol=1e-8)
        np.testing.assert_allclose(_atleast_1d(r["df2"]),
                                   _atleast_1d(py["df2"]), rtol=1e-8)

    # ------------------------------------------------------------------
    # R-B9 (fitFDist.R:45): nok==1 -> return list(scale=x[ok], df2=0)
    # ------------------------------------------------------------------
    def test_nok_one_returns_x_and_zero(self):
        """Exercises R-B9: when only one valid entry, returns scale=that, df2=0."""
        x = np.array([np.nan, np.inf, -1.0, 0.7, np.nan])
        py = fit_f_dist(x, df1=5)
        r = _r_fit_f_dist(x, df1=5)
        # R returns scalar 0.7, 0
        np.testing.assert_allclose(_atleast_1d(r["scale"]),
                                   _atleast_1d(py["scale"]), rtol=1e-15)
        np.testing.assert_allclose(_atleast_1d(r["df2"]),
                                   _atleast_1d(py["df2"]), atol=1e-15)

    # ------------------------------------------------------------------
    # R-B11 (fitFDist.R:57-67): covariate with insufficient unique values -> recurse
    # When unique(covariate) gives <2 splinedf, R recurses without covariate.
    # ------------------------------------------------------------------
    def test_covariate_one_unique_value_recurses(self):
        """Exercises R-B11: single unique covariate value -> recurse, scale broadcast."""
        rng = np.random.default_rng(106)
        n = 30
        x = 0.3 * rng.chisquare(5, size=n) / 5
        cov = np.full(n, 4.0)  # only one unique value

        py = fit_f_dist(x, df1=5, covariate=cov)
        r = _r_fit_f_dist(x, df1=5, covariate=cov)

        # In R-B11, scale is rep_len(scalar, n), so a length-n vector
        py_scale = _atleast_1d(py["scale"])
        r_scale = _atleast_1d(r["scale"])
        np.testing.assert_allclose(r_scale, py_scale, rtol=1e-8)
        np.testing.assert_allclose(_atleast_1d(r["df2"]),
                                   _atleast_1d(py["df2"]), rtol=1e-8)

    # ------------------------------------------------------------------
    # R-B12a (fitFDist.R:73-75): m==0 (>50% zero variances) -> warning + m<-1
    # ------------------------------------------------------------------
    def test_more_than_half_zero_warns(self):
        """Exercises R-B12a: median(x)==0 triggers eBayes-unreliable warning."""
        n = 30
        # >50% are zero
        x = np.concatenate([np.zeros(20), np.linspace(0.1, 1, 10)])

        with pytest.warns(UserWarning, match="More than half"):
            py = fit_f_dist(x, df1=5)
        # And R produces same numerical output
        # (R also warns; we just compare numbers)
        r = _r_fit_f_dist(x, df1=5)
        np.testing.assert_allclose(_atleast_1d(r["scale"]),
                                   _atleast_1d(py["scale"]), rtol=1e-6)
        np.testing.assert_allclose(_atleast_1d(r["df2"]),
                                   _atleast_1d(py["df2"]), rtol=1e-6)

    # ------------------------------------------------------------------
    # R-B12b (fitFDist.R:77): some zero variances but median > 0 -> warning
    # ------------------------------------------------------------------
    def test_zero_variance_warns(self):
        """Exercises R-B12b: a few zero variances among non-zero -> warning."""
        rng = np.random.default_rng(107)
        n = 50
        x = 0.3 * rng.chisquare(5, size=n) / 5
        x[0] = 0.0
        x[5] = 0.0

        with pytest.warns(UserWarning, match="Zero sample variances"):
            py = fit_f_dist(x, df1=5)
        r = _r_fit_f_dist(x, df1=5)
        np.testing.assert_allclose(_atleast_1d(r["scale"]),
                                   _atleast_1d(py["scale"]), rtol=1e-6)
        np.testing.assert_allclose(_atleast_1d(r["df2"]),
                                   _atleast_1d(py["df2"]), rtol=1e-6)

    # ------------------------------------------------------------------
    # R-B15 (fitFDist.R:90-102): covariate non-null -> spline fit
    # Constant covariate range with full data, large n.
    # ------------------------------------------------------------------
    def test_spline_trend_basic(self):
        """Exercises R-B15: spline fit for trend; scale becomes per-gene array."""
        rng = np.random.default_rng(108)
        n = 200
        cov = np.linspace(2, 10, n)
        # Variance trends with covariate
        x = (0.1 + 0.05 * cov) * rng.chisquare(5, size=n) / 5

        py = fit_f_dist(x, df1=5, covariate=cov)
        r = _r_fit_f_dist(x, df1=5, covariate=cov)

        # Slot equivalence (this is where many bugs hide)
        py_scale = _atleast_1d(py["scale"])
        r_scale = _atleast_1d(r["scale"])
        assert py_scale.shape == r_scale.shape, (
            f"scale shapes differ: R={r_scale.shape}, Py={py_scale.shape}"
        )
        # Tolerance: spline basis differs slightly between scipy BSpline
        # and R's splines::ns(); 1e-5 is the documented level.
        np.testing.assert_allclose(r_scale, py_scale, rtol=1e-5,
                                   err_msg="trend scale differs from R")
        np.testing.assert_allclose(_atleast_1d(r["df2"]),
                                   _atleast_1d(py["df2"]), rtol=1e-6)

    # ------------------------------------------------------------------
    # R-B15 with notallok=TRUE: spline fit + predict for non-ok entries
    # R uses predict(design, newx=covariate.notok) to extend emean to non-ok
    # entries. This is the most subtle covariate path. Critical: pylimma
    # _builds a new spline basis from scratch_ from covariate_notok rather
    # than using the original spline knots.
    # ------------------------------------------------------------------
    def test_spline_trend_with_notallok(self):
        """Exercises R-B15+R-B16: spline fit + predict for non-ok entries.

        Mix valid x with NaN/Inf x; R uses `predict(design, newx=...)` to
        extend the spline trend to the non-ok entries. pylimma rebuilds
        the basis from `covariate_notok` (line 417-419), which uses
        DIFFERENT boundary knots than the original fit. This is the
        critical divergence to discriminate.
        """
        rng = np.random.default_rng(109)
        n = 100
        cov = np.linspace(2, 10, n)
        x = (0.1 + 0.05 * cov) * rng.chisquare(5, size=n) / 5
        # Inject some NaN at varied covariate positions to force notallok
        bad_idx = [3, 27, 50, 71, 95]
        x[bad_idx] = np.nan

        py = fit_f_dist(x, df1=5, covariate=cov)
        r = _r_fit_f_dist(x, df1=5, covariate=cov)

        py_scale = _atleast_1d(py["scale"])
        r_scale = _atleast_1d(r["scale"])
        assert py_scale.shape == r_scale.shape

        # All-positions check, including the non-ok positions
        np.testing.assert_allclose(r_scale, py_scale, rtol=1e-5,
                                   err_msg="trend scale differs from R")
        np.testing.assert_allclose(_atleast_1d(r["df2"]),
                                   _atleast_1d(py["df2"]), rtol=1e-6)

    # ------------------------------------------------------------------
    # Critical sub-case of R-B16: bad x at boundary (extreme covariate values)
    # If pylimma rebuilds the spline basis from a smaller subset, the
    # boundary knots shift. Exercising this with a notallok at the edge
    # surfaces such bugs.
    # ------------------------------------------------------------------
    def test_spline_trend_notallok_at_boundary(self):
        """Exercises R-B16 with non-ok entries at the covariate boundary.

        The boundary knot determination differs if the basis is rebuilt
        from a smaller covariate range. R's predict() uses ORIGINAL knots
        from the fit; pylimma's _natural_spline_basis(covariate_notok, ...)
        builds NEW knots from covariate_notok's range. Whether this hits
        is data-dependent; this test stresses the case.
        """
        rng = np.random.default_rng(110)
        n = 80
        cov = np.linspace(2, 10, n)
        x = (0.1 + 0.05 * cov) * rng.chisquare(5, size=n) / 5
        # The OK subset's covariate has boundary [2, 10]. Make the
        # non-ok set at extreme boundaries [0, ...]:
        # actually the covariate only varies in [2, 10] so we put the
        # not-ok at indices 0 and n-1 to cover the edges.
        x[0] = np.nan
        x[-1] = np.nan

        py = fit_f_dist(x, df1=5, covariate=cov)
        r = _r_fit_f_dist(x, df1=5, covariate=cov)

        py_scale = _atleast_1d(py["scale"])
        r_scale = _atleast_1d(r["scale"])
        np.testing.assert_allclose(r_scale, py_scale, rtol=1e-5)

    # ------------------------------------------------------------------
    # R-B20 (fitFDist.R:106-108): evar > 0 -> df2 = 2 * trigammaInverse(evar)
    # The standard moment estimator path. Already covered by null-covariate
    # path; included here for explicit attribution.
    # ------------------------------------------------------------------
    def test_evar_positive_default_path(self):
        """Exercises R-B20: positive evar -> finite df2 via trigamma_inverse."""
        rng = np.random.default_rng(111)
        n = 150
        x = 0.5 * rng.chisquare(5, size=n) / 5  # high variability ensures evar>0

        py = fit_f_dist(x, df1=5)
        r = _r_fit_f_dist(x, df1=5)
        # Both should produce finite df2
        assert np.isfinite(py["df2"])
        assert np.isfinite(float(_atleast_1d(r["df2"])[0]))

        np.testing.assert_allclose(_atleast_1d(r["scale"]),
                                   _atleast_1d(py["scale"]), rtol=1e-8)
        np.testing.assert_allclose(_atleast_1d(r["df2"]),
                                   _atleast_1d(py["df2"]), rtol=1e-8)

    # ------------------------------------------------------------------
    # R-B21a (fitFDist.R:111-115): evar<=0, null covariate -> df2=Inf, s20=mean(x)
    # Constant variances trigger this.
    # ------------------------------------------------------------------
    def test_evar_nonpositive_null_cov(self):
        """Exercises R-B21a: evar<=0 with null covariate -> df2=Inf, scale=mean(x)."""
        n = 50
        x = np.full(n, 0.5)  # constant -> log-variance var ~ 0 -> evar < 0

        py = fit_f_dist(x, df1=5)
        r = _r_fit_f_dist(x, df1=5)

        # df2 should be Inf in both
        py_df2 = py["df2"]
        r_df2 = float(_atleast_1d(r["df2"])[0])
        assert np.isinf(py_df2), f"py_df2={py_df2}, expected Inf"
        assert np.isinf(r_df2), f"r_df2={r_df2}, expected Inf"

        # scale = mean(x) = 0.5
        np.testing.assert_allclose(_atleast_1d(r["scale"]),
                                   _atleast_1d(py["scale"]), rtol=1e-12)

    # ------------------------------------------------------------------
    # R-B21b (fitFDist.R:116-117): evar<=0 with covariate -> s20=exp(emean)
    # ------------------------------------------------------------------
    def test_evar_nonpositive_with_covariate(self):
        """Exercises R-B21b: evar<=0 with covariate -> df2=Inf, scale=exp(emean).

        Constant variance with varying covariate forces the spline fit
        but with effectively zero residual variance after de-trending.
        """
        n = 80
        rng = np.random.default_rng(112)
        cov = np.linspace(2, 10, n)
        # Variance is constant after removing the trend
        x = np.full(n, 0.5)

        # Suppress the eBayes warning that fires here ("More than half...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            py = fit_f_dist(x, df1=5, covariate=cov)
            r = _r_fit_f_dist(x, df1=5, covariate=cov)

        py_df2 = py["df2"]
        r_df2 = float(_atleast_1d(r["df2"])[0])
        # Both should be Inf or very large
        assert np.isinf(py_df2) or py_df2 > 1e6
        assert np.isinf(r_df2) or r_df2 > 1e6

        py_scale = _atleast_1d(py["scale"])
        r_scale = _atleast_1d(r["scale"])
        np.testing.assert_allclose(r_scale, py_scale, rtol=1e-5)

    # ------------------------------------------------------------------
    # Sanity: vector df1 with covariate (combined R-B14 & R-B15 path
    # ------------------------------------------------------------------
    def test_vector_df1_with_covariate(self):
        """Exercises R-B15 with vector df1: spline + per-gene df1 effects."""
        rng = np.random.default_rng(113)
        n = 150
        cov = np.linspace(2, 10, n)
        df1 = np.where(np.arange(n) < n // 2, 4.0, 6.0)
        x = (0.1 + 0.05 * cov) * rng.chisquare(df1) / df1

        py = fit_f_dist(x, df1=df1, covariate=cov)
        r = _r_fit_f_dist(x, df1=df1, covariate=cov)

        py_scale = _atleast_1d(py["scale"])
        r_scale = _atleast_1d(r["scale"])
        np.testing.assert_allclose(r_scale, py_scale, rtol=1e-5)
        np.testing.assert_allclose(_atleast_1d(r["df2"]),
                                   _atleast_1d(py["df2"]), rtol=1e-5)

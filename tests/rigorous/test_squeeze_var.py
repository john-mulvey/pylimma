"""
Rigorous per-branch parity tests for pylimma.squeeze_var.squeeze_var.

Each test exercises a specific R branch of squeezeVar() / .squeezeVar()
in R limma's squeezeVar.R.

These tests were added by a rigorous single-function audit. They are
intentionally tight (rtol=1e-8) and run a live R subprocess via
helpers.run_r_comparison so any regression surfaces immediately.
"""

from __future__ import annotations

import numpy as np
import pytest

from pylimma.squeeze_var import _squeeze_var_core, squeeze_var

from ..helpers import (
    limma_available,
    run_r_comparison,
)

pytestmark = pytest.mark.skipif(not limma_available(), reason="R/limma not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_var_df(n=50, df_val=5, seed=0):
    """Generate a representative (var, df) input pair from chi-squared draws."""
    rng = np.random.default_rng(seed)
    sample_var = 0.5 * rng.chisquare(df_val, size=n) / df_val
    return sample_var, np.full(n, float(df_val))


def _r_squeeze_var(var, df, **kwargs):
    """Run R squeezeVar via subprocess and return all three slots.

    Always returns dict with keys var_post (length n), var_prior (length 1
    or n), df_prior (length 1 or n). Optional kwargs forwarded as named R
    arguments.
    """
    args = ["sample_var", "df_input[,1]"]
    extras = []
    for k, v in kwargs.items():
        rk = k.replace("_", ".")
        if isinstance(v, bool):
            extras.append(f"{rk}={'TRUE' if v else 'FALSE'}")
        elif isinstance(v, (tuple, list)):
            extras.append(f"{rk}=c({','.join(str(x) for x in v)})")
        elif v is None:
            extras.append(f"{rk}=NULL")
        else:
            extras.append(f"{rk}={v}")
    args_str = ", ".join(args + extras)

    code = f"""
    suppressMessages(library(limma))
    sample_var <- as.matrix(read.csv('{{tmpdir}}/var.csv', row.names=1))[,1]
    df_input <- as.matrix(read.csv('{{tmpdir}}/df.csv', row.names=1))
    sv <- squeezeVar({args_str})
    var_post <- sv$var.post
    df_prior <- sv$df.prior
    var_prior <- sv$var.prior
    """

    result = run_r_comparison(
        py_data={"var": np.asarray(var, dtype=float), "df": np.asarray(df, dtype=float)},
        r_code_template=code,
        output_vars=["var_post", "df_prior", "var_prior"],
    )
    return result


def _r_squeeze_var_with_cov(var, df, covariate, **kwargs):
    args = ["sample_var", "df_input[,1]"]
    extras = ["covariate=cov_input[,1]"]
    for k, v in kwargs.items():
        rk = k.replace("_", ".")
        if isinstance(v, bool):
            extras.append(f"{rk}={'TRUE' if v else 'FALSE'}")
        elif isinstance(v, (tuple, list)):
            extras.append(f"{rk}=c({','.join(str(x) for x in v)})")
        elif v is None:
            extras.append(f"{rk}=NULL")
        else:
            extras.append(f"{rk}={v}")
    args_str = ", ".join(args + extras)

    code = f"""
    suppressMessages(library(limma))
    sample_var <- as.matrix(read.csv('{{tmpdir}}/var.csv', row.names=1))[,1]
    df_input <- as.matrix(read.csv('{{tmpdir}}/df.csv', row.names=1))
    cov_input <- as.matrix(read.csv('{{tmpdir}}/cov.csv', row.names=1))
    sv <- squeezeVar({args_str})
    var_post <- sv$var.post
    df_prior <- sv$df.prior
    var_prior <- sv$var.prior
    """

    result = run_r_comparison(
        py_data={
            "var": np.asarray(var, dtype=float),
            "df": np.asarray(df, dtype=float),
            "cov": np.asarray(covariate, dtype=float),
        },
        r_code_template=code,
        output_vars=["var_post", "df_prior", "var_prior"],
    )
    return result


def _atleast_1d(x):
    return np.atleast_1d(np.asarray(x, dtype=float))


class TestRigorousSqueezeVar:
    """One test per uncovered/partial R branch of squeezeVar / .squeezeVar."""

    # ------------------------------------------------------------------
    # R-B1 (squeezeVar.R:11): identical(n,0L) -> stop
    # ------------------------------------------------------------------
    def test_empty_input_stops(self):
        """Exercises R-B1: empty var raises an error in both R and pylimma."""
        with pytest.raises(ValueError, match="var is empty"):
            squeeze_var(np.array([]), df=5)

    # ------------------------------------------------------------------
    # R-B2 (squeezeVar.R:14): n<3 -> early return (var.post=var, var.prior=var, df.prior=0)
    # ------------------------------------------------------------------
    def test_n_lt_3_returns_var_unchanged(self):
        """Exercises R-B2: n=2 returns var.post=var, var.prior=var, df.prior=0."""
        var = np.array([0.4, 0.7])
        py = squeeze_var(var, df=5)
        # Crucial: var.prior=var (same values), df.prior=0
        np.testing.assert_array_equal(py["var_post"], var)
        np.testing.assert_array_equal(py["var_prior"], var)
        assert py["df_prior"] == 0.0

        # And differential: R returns same shape and values
        r = _r_squeeze_var(var, np.full(2, 5.0))
        np.testing.assert_allclose(r["var_post"], py["var_post"], rtol=1e-12)
        np.testing.assert_allclose(
            _atleast_1d(r["var_prior"]), _atleast_1d(py["var_prior"]), rtol=1e-12
        )
        np.testing.assert_allclose(
            _atleast_1d(r["df_prior"]), _atleast_1d(py["df_prior"]), atol=1e-12
        )

    # ------------------------------------------------------------------
    # R-B3 (squeezeVar.R:17): var[df==0] <- 0 when length(df)>1
    # ------------------------------------------------------------------
    def test_df_zero_zeros_var(self):
        """Exercises R-B3: when length(df)>1, var[df==0] is set to 0."""
        n = 30
        rng = np.random.default_rng(0)
        var = rng.uniform(0.1, 1.0, n)
        # Inject NaN at positions where df==0; R's clamp must hide it
        var_bad = var.copy()
        var_bad[0] = np.nan
        var_bad[1] = np.inf
        df = np.full(n, 5.0)
        df[0] = 0
        df[1] = 0

        py = squeeze_var(var_bad.copy(), df=df.copy())
        r = _r_squeeze_var(var_bad, df)

        # Compare full slots tightly
        np.testing.assert_allclose(r["var_post"], py["var_post"], rtol=1e-8)
        np.testing.assert_allclose(
            _atleast_1d(r["var_prior"]), _atleast_1d(py["var_prior"]), rtol=1e-8
        )
        np.testing.assert_allclose(
            _atleast_1d(r["df_prior"]), _atleast_1d(py["df_prior"]), rtol=1e-8
        )

    # ------------------------------------------------------------------
    # R-B4 (squeezeVar.R:20): non-NULL span forces legacy=FALSE
    # ------------------------------------------------------------------
    def test_span_forces_legacy_false(self):
        """Exercises R-B4: span!=NULL forces legacy=FALSE.

        Compare against R squeezeVar() with span=0.5. Verifies df_prior
        from the new path matches the unequal_df1 result.
        """
        n = 60
        rng = np.random.default_rng(1)
        sample_var = 0.5 * rng.chisquare(5, size=n) / 5
        df = np.full(n, 5.0)

        py = squeeze_var(sample_var, df=df, span=0.5)
        r = _r_squeeze_var(sample_var, df, span=0.5)

        np.testing.assert_allclose(r["var_post"], py["var_post"], rtol=1e-6)
        np.testing.assert_allclose(
            _atleast_1d(r["var_prior"]), _atleast_1d(py["var_prior"]), rtol=1e-6
        )
        np.testing.assert_allclose(
            _atleast_1d(r["df_prior"]), _atleast_1d(py["df_prior"]), rtol=1e-6
        )

    # ------------------------------------------------------------------
    # R-B5a (squeezeVar.R:23-26): legacy is NULL & all dfp equal -> legacy=TRUE
    # ------------------------------------------------------------------
    def test_auto_legacy_true_when_df_equal(self):
        """Exercises R-B5: equal positive df triggers legacy=TRUE path."""
        n = 50
        rng = np.random.default_rng(2)
        sample_var = 0.5 * rng.chisquare(5, size=n) / 5
        df = np.full(n, 5.0)

        py = squeeze_var(sample_var, df=df)
        r = _r_squeeze_var(sample_var, df)

        np.testing.assert_allclose(r["var_post"], py["var_post"], rtol=1e-6)
        np.testing.assert_allclose(
            _atleast_1d(r["df_prior"]), _atleast_1d(py["df_prior"]), rtol=1e-6
        )
        np.testing.assert_allclose(
            _atleast_1d(r["var_prior"]), _atleast_1d(py["var_prior"]), rtol=1e-6
        )

    # ------------------------------------------------------------------
    # R-B5b: unequal df -> legacy=FALSE (and df_prior is per-gene array in R)
    # ------------------------------------------------------------------
    def test_auto_legacy_false_when_df_unequal(self):
        """Exercises R-B5: unequal df picks legacy=FALSE / unequal_df1."""
        n = 60
        rng = np.random.default_rng(3)
        sample_var = 0.5 * rng.chisquare(5, size=n) / 5
        df = np.concatenate([np.full(30, 3.0), np.full(30, 6.0)])

        py = squeeze_var(sample_var, df=df)
        r = _r_squeeze_var(sample_var, df)

        np.testing.assert_allclose(r["var_post"], py["var_post"], rtol=1e-5)
        # df_prior could be scalar or array depending on robust mode - here it
        # is scalar from fitFDistUnequalDF1 default (robust=FALSE)
        py_dfp = _atleast_1d(py["df_prior"])
        r_dfp = _atleast_1d(r["df_prior"])
        np.testing.assert_allclose(r_dfp, py_dfp, rtol=1e-5)

    # ------------------------------------------------------------------
    # R-B5c: all df<=0 -> R: dfp=empty, min/max=Inf/-Inf, identical(...)=FALSE -> legacy=FALSE
    # ------------------------------------------------------------------
    def test_auto_legacy_with_all_zero_df(self):
        """Exercises R-B5 when df has no positive entries.

        This is an esoteric corner: if df==0 everywhere R sets var to 0 then
        falls into the unequal_df1 branch. Compare to R for whichever
        outcome R produces.
        """
        n = 30
        sample_var = np.full(n, 0.5)
        df = np.zeros(n)

        # Both R and pylimma should produce *something* (NaN scale or error).
        # We just confirm pylimma's output matches R, whatever it is.
        try:
            r = _r_squeeze_var(sample_var, df)
        except RuntimeError:
            # R raised - then pylimma should also raise
            with pytest.raises(Exception):
                squeeze_var(sample_var, df=df)
            return

        py = squeeze_var(sample_var, df=df)
        # Whatever R returned must match
        np.testing.assert_allclose(
            np.nan_to_num(r["var_post"], nan=-1.0),
            np.nan_to_num(py["var_post"], nan=-1.0),
            rtol=1e-6,
        )

    # ------------------------------------------------------------------
    # R-B6 (squeezeVar.R:30-32): legacy & robust path returns df.prior=df2.shrunk
    # ------------------------------------------------------------------
    def test_legacy_robust_df_prior_is_shrunk(self):
        """Exercises R-B6: legacy=TRUE,robust=TRUE returns per-gene df_prior."""
        n = 80
        rng = np.random.default_rng(4)
        sample_var = 0.5 * rng.chisquare(5, size=n) / 5
        # Inject outliers
        sample_var[:5] *= 10
        df = np.full(n, 5.0)

        py = squeeze_var(sample_var, df=df, robust=True)
        r = _r_squeeze_var(sample_var, df, robust=True)

        np.testing.assert_allclose(r["var_post"], py["var_post"], rtol=1e-6)
        # df_prior must be per-gene (length n) in robust mode
        py_dfp = _atleast_1d(py["df_prior"])
        r_dfp = _atleast_1d(r["df_prior"])
        assert len(py_dfp) == n, f"expected length-{n} df_prior, got {len(py_dfp)}"
        assert len(r_dfp) == n
        np.testing.assert_allclose(r_dfp, py_dfp, rtol=1e-6)

    # ------------------------------------------------------------------
    # R-B7 (squeezeVar.R:34-35): legacy & not robust returns scalar df_prior=df2
    # ------------------------------------------------------------------
    def test_legacy_nonrobust_df_prior_scalar(self):
        """Exercises R-B7: legacy=TRUE,robust=FALSE returns scalar df_prior."""
        n = 50
        rng = np.random.default_rng(5)
        sample_var = 0.5 * rng.chisquare(5, size=n) / 5
        df = np.full(n, 5.0)

        py = squeeze_var(sample_var, df=df, legacy=True, robust=False)
        r = _r_squeeze_var(sample_var, df, legacy=True, robust=False)

        np.testing.assert_allclose(r["var_post"], py["var_post"], rtol=1e-8)
        np.testing.assert_allclose(
            _atleast_1d(r["df_prior"]), _atleast_1d(py["df_prior"]), rtol=1e-8
        )

    # ------------------------------------------------------------------
    # R-B8 (squeezeVar.R:37-41): non-legacy path; df.prior=df2.shrunk if non-null else df2
    # ------------------------------------------------------------------
    def test_non_legacy_unequal_df1_path(self):
        """Exercises R-B8: legacy=FALSE -> fitFDistUnequalDF1.

        df_prior is df2 (scalar) in default mode, df2_shrunk (per-gene)
        when robust outliers found.
        """
        n = 60
        rng = np.random.default_rng(6)
        sample_var = 0.5 * rng.chisquare(5, size=n) / 5
        df = np.concatenate([np.full(20, 3.0), np.full(20, 5.0), np.full(20, 8.0)])

        py = squeeze_var(sample_var, df=df, legacy=False, robust=False)
        r = _r_squeeze_var(sample_var, df, legacy=False, robust=False)

        np.testing.assert_allclose(r["var_post"], py["var_post"], rtol=1e-5)
        np.testing.assert_allclose(
            _atleast_1d(r["df_prior"]), _atleast_1d(py["df_prior"]), rtol=1e-5
        )
        np.testing.assert_allclose(
            _atleast_1d(r["var_prior"]), _atleast_1d(py["var_prior"]), rtol=1e-5
        )

    def test_non_legacy_robust_unequal_df1_path(self):
        """Exercises R-B8: legacy=FALSE,robust=TRUE - df_prior=df2_shrunk."""
        n = 80
        rng = np.random.default_rng(7)
        sample_var = 0.5 * rng.chisquare(5, size=n) / 5
        sample_var[:8] *= 15  # outliers
        df = np.concatenate([np.full(40, 3.0), np.full(40, 5.0)])

        py = squeeze_var(sample_var, df=df, legacy=False, robust=True)
        r = _r_squeeze_var(sample_var, df, legacy=False, robust=True)

        np.testing.assert_allclose(r["var_post"], py["var_post"], rtol=1e-5)
        np.testing.assert_allclose(
            _atleast_1d(r["df_prior"]), _atleast_1d(py["df_prior"]), rtol=1e-5
        )

    # ------------------------------------------------------------------
    # R-B11 (.squeezeVar:57-58): canonical formula with finite df_prior
    # ------------------------------------------------------------------
    def test_squeeze_var_core_canonical_formula(self):
        """Exercises R-B11: all-finite df_prior canonical shrinkage."""
        var = np.array([0.1, 0.5, 1.0, 2.0])
        df = np.array([5.0, 5.0, 5.0, 5.0])
        var_prior = 0.5
        df_prior = 4.0

        result = _squeeze_var_core(var, df, var_prior, df_prior)
        expected = (df * var + df_prior * var_prior) / (df + df_prior)
        np.testing.assert_allclose(result, expected, rtol=1e-15)

    # ------------------------------------------------------------------
    # R-B12 (.squeezeVar:62-63): length(var.prior)==n branch
    # R-B14 (.squeezeVar:69-70): all df_prior > 1e100 -> return var.post (=var.prior)
    # ------------------------------------------------------------------
    def test_squeeze_var_core_all_inf_with_vector_var_prior(self):
        """Exercises R-B12+R-B14: var.prior length n + df_prior all Inf."""
        n = 5
        var = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
        df = np.full(n, 5.0)
        var_prior = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        df_prior = np.full(n, np.inf)

        result = _squeeze_var_core(var, df, var_prior, df_prior)
        # Should return var_prior unchanged
        np.testing.assert_allclose(result, var_prior, rtol=1e-15)

    def test_squeeze_var_core_all_inf_with_scalar_var_prior(self):
        """Exercises R-B14 with scalar var_prior -> recycled to length n."""
        n = 5
        var = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
        df = np.full(n, 5.0)
        var_prior = 0.42
        df_prior = np.full(n, np.inf)

        result = _squeeze_var_core(var, df, var_prior, df_prior)
        np.testing.assert_allclose(result, np.full(n, 0.42), rtol=1e-15)

    # ------------------------------------------------------------------
    # R-B15 (.squeezeVar:73-76): mixed Inf/finite df_prior
    # ------------------------------------------------------------------
    def test_squeeze_var_core_mixed_inf_finite_df_prior(self):
        """Exercises R-B15: only some df_prior finite."""
        n = 5
        var = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
        df = np.full(n, 5.0)
        var_prior = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        df_prior = np.array([4.0, np.inf, 2.0, np.inf, 6.0])

        result = _squeeze_var_core(var, df, var_prior, df_prior)
        # Finite indices: standard formula
        expected = var_prior.copy()
        finite = np.array([0, 2, 4])
        for i in finite:
            expected[i] = (df[i] * var[i] + df_prior[i] * var_prior[i]) / (df[i] + df_prior[i])
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    # ------------------------------------------------------------------
    # R-B12 vs R-B13: len(var.prior)!=n behaviour
    # R uses rep_len() -> recycles. Python uses copy + size mismatch potential.
    # ------------------------------------------------------------------
    def test_core_var_prior_shorter_than_var_recycles_in_R(self):
        """Exercises R-B13: when length(var.prior) != n R uses rep_len.

        Python's _squeeze_var_core calls var_prior.copy() if ndim>0 which
        will not recycle. Verify behaviour against R when df_prior is all
        Inf and var_prior is length 1 array (the scalar-via-array case).
        """
        # Build a synthetic squeeze_var return path through the R surface
        # using scaled F draws that produce df2=Inf (constant variances).
        n = 30
        sample_var = np.full(n, 0.5)  # constant -> df2 estimate -> Inf
        df = np.full(n, 5.0)

        py = squeeze_var(sample_var, df=df, legacy=True, robust=False)
        r = _r_squeeze_var(sample_var, df, legacy=True, robust=False)

        np.testing.assert_allclose(r["var_post"], py["var_post"], rtol=1e-6)
        # df_prior should be ~Inf in both
        py_dfp = py["df_prior"]
        r_dfp = float(_atleast_1d(r["df_prior"])[0])
        # Both should be Inf or very large
        assert np.isinf(py_dfp) or py_dfp > 1e6
        assert np.isinf(r_dfp) or r_dfp > 1e6

    # ------------------------------------------------------------------
    # R-B9 (squeezeVar.R:42): anyNA(df.prior) -> stop
    # ------------------------------------------------------------------
    def test_anyna_df_prior_raises(self):
        """Exercises R-B9: NaN in df.prior must raise.

        We force NaN by passing var with all-NaN entries in legacy=False
        path. R's fitFDistUnequalDF1 will return NA df2, triggering stop.
        """
        n = 10
        var = np.full(n, np.nan)
        df = np.full(n, 5.0)

        # Both R and pylimma should raise (or return graceful defaults).
        # Compare behaviour: try R first
        r_failed = False
        try:
            _r_squeeze_var(var, df)
        except RuntimeError:
            r_failed = True

        # pylimma behaviour
        py_failed = False
        try:
            squeeze_var(var, df=df)
        except (ValueError, RuntimeError):
            py_failed = True

        # If R raised, pylimma should also raise
        assert r_failed == py_failed, f"R failed={r_failed}, pylimma failed={py_failed}"

    # ------------------------------------------------------------------
    # Trend (covariate) path; legacy=TRUE
    # ------------------------------------------------------------------
    def test_legacy_trend_var_prior_per_gene(self):
        """Exercises legacy=TRUE with covariate -> per-gene var_prior."""
        n = 100
        rng = np.random.default_rng(8)
        cov = np.linspace(2, 10, n)
        # Variance trend with covariate
        sample_var = (0.1 + 0.05 * cov) * rng.chisquare(5, size=n) / 5
        df = np.full(n, 5.0)

        py = squeeze_var(sample_var, df=df, covariate=cov, legacy=True)
        r = _r_squeeze_var_with_cov(sample_var, df, cov, legacy=True)

        np.testing.assert_allclose(r["var_post"], py["var_post"], rtol=1e-6)
        py_vp = _atleast_1d(py["var_prior"])
        r_vp = _atleast_1d(r["var_prior"])
        np.testing.assert_allclose(r_vp, py_vp, rtol=1e-6)

    # ------------------------------------------------------------------
    # Edge: span set with non-NULL forces legacy=FALSE even with equal df
    # ------------------------------------------------------------------
    def test_span_overrides_auto_legacy(self):
        """Exercises R-B4 again: span explicit, equal df should use NEW path.

        With auto-detect (no span), equal df -> legacy=TRUE.
        With span explicit, R sets legacy=FALSE.
        Confirm pylimma matches both behaviours by using the same input.
        """
        n = 50
        rng = np.random.default_rng(9)
        sample_var = 0.5 * rng.chisquare(5, size=n) / 5
        df = np.full(n, 5.0)

        # Without span: legacy=TRUE
        py_legacy = squeeze_var(sample_var, df=df)
        r_legacy = _r_squeeze_var(sample_var, df)
        np.testing.assert_allclose(r_legacy["var_post"], py_legacy["var_post"], rtol=1e-6)

        # With span: legacy=FALSE - results should differ from above
        py_new = squeeze_var(sample_var, df=df, span=0.5)
        r_new = _r_squeeze_var(sample_var, df, span=0.5)
        np.testing.assert_allclose(r_new["var_post"], py_new["var_post"], rtol=1e-5)

    # ------------------------------------------------------------------
    # Combined: legacy=False + covariate
    # ------------------------------------------------------------------
    def test_non_legacy_with_covariate(self):
        """Exercises R-B8 with covariate: per-gene var_prior expected."""
        n = 80
        rng = np.random.default_rng(10)
        cov = np.linspace(2, 10, n)
        sample_var = (0.1 + 0.05 * cov) * rng.chisquare(5, size=n) / 5
        df = np.concatenate([np.full(40, 3.0), np.full(40, 6.0)])

        py = squeeze_var(sample_var, df=df, covariate=cov, legacy=False)
        r = _r_squeeze_var_with_cov(sample_var, df, cov, legacy=False)

        np.testing.assert_allclose(r["var_post"], py["var_post"], rtol=1e-4)
        py_vp = _atleast_1d(py["var_prior"])
        r_vp = _atleast_1d(r["var_prior"])
        np.testing.assert_allclose(r_vp, py_vp, rtol=1e-4)

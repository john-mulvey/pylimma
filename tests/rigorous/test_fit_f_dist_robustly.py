"""
Rigorous per-branch parity tests for pylimma.squeeze_var.fit_f_dist_robustly.

Each test exercises a branch of fitFDistRobustly() in R limma's
fitFDistRobustly.R against a live R subprocess, comparing every returned slot.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylimma.squeeze_var import fit_f_dist_robustly

from ..helpers import limma_available, run_r_code

pytestmark = pytest.mark.skipif(not limma_available(), reason="R/limma not available")

SLOTS = ("scale", "df2", "df2_shrunk", "tail_p_value", "prob_outlier", "df2_outlier")


def _r_fit(x, df1, covariate=None, winsor_tail_p=(0.05, 0.1)):
    """Every slot of R's fitFDistRobustly, keyed by pylimma's slot names."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pd.DataFrame({"v": np.atleast_1d(np.asarray(x, dtype=float))}).to_csv(
            tmp / "x.csv", index=False, na_rep="NA"
        )
        pd.DataFrame({"v": np.atleast_1d(np.asarray(df1, dtype=float))}).to_csv(
            tmp / "df1.csv", index=False, na_rep="NA"
        )
        pd.DataFrame({"v": np.atleast_1d(np.asarray(winsor_tail_p, dtype=float))}).to_csv(
            tmp / "wtp.csv", index=False
        )
        cov_code = "NULL"
        if covariate is not None:
            pd.DataFrame({"v": np.asarray(covariate, dtype=float)}).to_csv(
                tmp / "cov.csv", index=False, na_rep="NA"
            )
            cov_code = f"read.csv('{tmp}/cov.csv')$v"
        run_r_code(
            "suppressMessages(library(limma))\n"
            f"x <- read.csv('{tmp}/x.csv')$v\n"
            f"df1 <- read.csv('{tmp}/df1.csv')$v\n"
            f"fit <- fitFDistRobustly(x, df1, covariate = {cov_code}, "
            f"winsor.tail.p = read.csv('{tmp}/wtp.csv')$v)\n"
            f"for (nm in names(fit)) write.csv(data.frame(v = fit[[nm]]), "
            f"file.path('{tmp}', paste0(gsub('.', '_', nm, fixed = TRUE), '.csv')), row.names = FALSE)\n"
        )
        return {
            slot: pd.read_csv(tmp / f"{slot}.csv")["v"].to_numpy(dtype=float)
            for slot in SLOTS
            if (tmp / f"{slot}.csv").exists()
        }


def _r_error(call: str) -> str:
    return run_r_code(
        "suppressMessages(library(limma))\n"
        f"cat(tryCatch({{ {call}; '' }}, error = function(e) conditionMessage(e)))\n"
    ).strip()


def _assert_matches(py, r, rtol=1e-8):
    """Each R slot equal at `rtol`, then the same set of slots as R.

    prob_outlier is -expm1() of a small log probability, which magnifies
    ~1e-10 differences in the lowess trend; as a probability it is compared
    with an absolute tolerance of 1e-9 (arbitrary choice).
    """
    for slot, r_value in r.items():
        if py.get(slot) is None:
            continue
        py_value = np.atleast_1d(np.asarray(py[slot], dtype=float))
        if r_value.size > 1 and py_value.size == 1:
            py_value = np.full(r_value.size, py_value[0])
        if slot == "tail_p_value":
            # p-values on the log10 scale (project policy); exact zeros must agree
            assert np.array_equal(py_value == 0, r_value == 0), slot
            keep = r_value > 0
            np.testing.assert_allclose(
                np.log10(py_value[keep]), np.log10(r_value[keep]), rtol=1e-6, err_msg=slot
            )
        else:
            atol = 1e-9 if slot == "prob_outlier" else 1e-300
            np.testing.assert_allclose(py_value, r_value, rtol=rtol, atol=atol, err_msg=slot)
    assert sorted(k for k in SLOTS if py.get(k) is not None) == sorted(r), (sorted(py), sorted(r))


def _variances(n, df1, df2, seed, outliers=0):
    """Scaled-F variances s0^2 * F(df1, df2) with the first `outliers` inflated."""
    rng = np.random.default_rng(seed)
    x = (
        0.5 * rng.f(df1, df2, size=n)
        if np.isfinite(df2)
        else 0.5 * rng.chisquare(df1, size=n) / df1
    )
    x[:outliers] *= 20
    return x


class TestRigorousFitFDistRobustly:
    def test_finite_df2_with_outliers(self):
        """Exercises R-B16 (uniroot) and R-B17 (outlier df2) with R-B12 (trimmed mean)."""
        x = _variances(200, 4, 8, seed=1, outliers=5)
        _assert_matches(fit_f_dist_robustly(x, 4), _r_fit(x, 4))

    def test_infinite_df2_branch(self):
        """Exercises R-B14 (fitFDistRobustly.R:131-154): funvalInf <= 0, df2 = Inf,
        with heavily tied variances (rank ties in EmpiricalTailProb)."""
        x = np.full(100, 0.5)
        x[:6] = [5.0, 0.01, 4.0, 0.02, 3.0, 0.03]
        r = _r_fit(x, 4)
        assert np.isinf(r["df2"][0]) and "tail_p_value" in r
        _assert_matches(fit_f_dist_robustly(x, 4), r)

    def test_covariate_trend(self):
        """Exercises R-B13 (fitFDistRobustly.R:101): loessFit(z, covariate, span=0.4)."""
        rng = np.random.default_rng(3)
        cov = np.sort(rng.uniform(2, 12, 300))
        x = _variances(300, 4, 10, seed=3, outliers=4) * np.exp(-0.2 * cov)
        _assert_matches(fit_f_dist_robustly(x, 4, covariate=cov), _r_fit(x, 4, covariate=cov))

    def test_not_all_ok_with_tied_covariate(self):
        """Exercises R-B6 (fitFDistRobustly.R:26-47): NA x and zero df1 excluded,
        scale interpolated with approx(rule=2, ties=mean) at tied covariates."""
        rng = np.random.default_rng(4)
        cov = np.round(rng.uniform(2, 12, 200), 0)
        x = _variances(200, 4, 10, seed=4, outliers=3) * np.exp(-0.2 * cov)
        df1 = np.full(200, 4.0)
        x[[10, 50]] = np.nan
        df1[[20, 199]] = 0
        _assert_matches(fit_f_dist_robustly(x, df1, covariate=cov), _r_fit(x, df1, covariate=cov))

    def test_unequal_df1_transformed(self):
        """Exercises R-B11 (fitFDistRobustly.R:74-91): x for df1 < max(df1) is
        mapped through the F quantiles to df1 = max(df1)."""
        df1 = np.repeat([2.0, 4.0, 6.0], 70)
        x = np.concatenate([_variances(70, d, 10, seed=5 + int(d)) for d in (2, 4, 6)])
        x[:3] *= 20
        _assert_matches(fit_f_dist_robustly(x, df1), _r_fit(x, df1))

    def test_tied_variances(self):
        """Exercises R-B17's rank(Fstat) and order(LogTailP) with tied values."""
        x = np.round(_variances(200, 4, 8, seed=6, outliers=5), 1) + 0.05
        _assert_matches(fit_f_dist_robustly(x, 4), _r_fit(x, 4))

    def test_extreme_outlier_log_tail(self):
        """Exercises R-B17 (fitFDistRobustly.R:188, :222-231): pf(log.p=TRUE)
        for an F statistic whose tail probability underflows to 0 while its
        log stays finite, so df2.outlier comes from the finite log."""
        x = _variances(200, 4, 60, seed=7)
        x[0] = 1e30
        r = _r_fit(x, 4)
        assert r["tail_p_value"].min() == 0 and r["df2_outlier"][0] > 0
        # R's uniroot stops at tol 1e-8 on the df2/(1+df2) scale; with the root
        # near 1 (df2 = 32) R's df2 is only accurate to ~1e-8 relative
        _assert_matches(fit_f_dist_robustly(x, 4), r, rtol=1e-6)

    def test_scalar_winsor_tail_p(self):
        """Exercises R-B9 (fitFDistRobustly.R:66): rep_len(winsor.tail.p, 2)."""
        x = _variances(200, 4, 8, seed=8, outliers=5)
        _assert_matches(
            fit_f_dist_robustly(x, 4, winsor_tail_p=0.1), _r_fit(x, 4, winsor_tail_p=0.1)
        )

    def test_small_winsor_tail_p_returns_non_robust(self):
        """Exercises R-B10 (fitFDistRobustly.R:68-71): all(winsor.tail.p < 1/n)."""
        x = _variances(50, 4, 8, seed=9, outliers=2)
        wtp = (0.01, 0.01)
        _assert_matches(
            fit_f_dist_robustly(x, 4, winsor_tail_p=wtp), _r_fit(x, 4, winsor_tail_p=wtp)
        )

    def test_non_robust_df2_infinite_returns_non_robust(self):
        """Exercises R-B15 (fitFDistRobustly.R:167-170): NonRobust$df2 == Inf
        while the Winsorized variance exceeds the df2 = Inf moment (bimodal
        log-variances); R returns the non-robust fit with no tail p-values."""
        x = np.exp(np.tile([-0.7, 0.7], 50) + 0.01 * np.sin(np.arange(100)))
        r = _r_fit(x, 4)
        assert np.isinf(r["df2"][0]) and "tail_p_value" not in r
        _assert_matches(fit_f_dist_robustly(x, 4), r)

    @pytest.mark.parametrize("n", [1, 2])
    def test_too_few_values(self, n):
        """Exercises R-B1 (n < 2 -> NA) and R-B2 (n == 2 -> fitFDist) at :12-13."""
        x = np.array([0.4, 1.3])[:n]
        _assert_matches(fit_f_dist_robustly(x, 4), _r_fit(x, 4))

    @pytest.mark.parametrize(
        "kwargs, r_call, message",
        [
            (
                {"x": np.ones(5), "df1": np.ones(3)},
                "fitFDistRobustly(rep(1, 5), rep(1, 3))",
                "x and df1 are different lengths",
            ),
            (
                {"x": np.ones(5), "df1": 4, "covariate": np.ones(3)},
                "fitFDistRobustly(rep(1, 5), 4, covariate = rep(1, 3))",
                "x and covariate are different lengths",
            ),
            (
                {"x": np.ones(5), "df1": 4, "covariate": np.array([1, 2, np.inf, 4, 5])},
                "fitFDistRobustly(rep(1, 5), 4, covariate = c(1, 2, Inf, 4, 5))",
                "covariate contains NA or infinite values",
            ),
            (
                {"x": np.array([0.0, 0.0, 0.0, 1.0, 2.0]), "df1": 4},
                "fitFDistRobustly(c(0, 0, 0, 1, 2), 4)",
                "Variances are mostly <= 0",
            ),
        ],
    )
    def test_input_errors(self, kwargs, r_call, message):
        """Exercises R-B3, R-B4, R-B5 (fitFDistRobustly.R:16-21) and R-B7 (:51)."""
        assert _r_error(r_call) == message
        with pytest.raises(ValueError, match=message):
            fit_f_dist_robustly(**kwargs)

    def test_upper_trim_half_uses_median(self):
        """Exercises R-B12 (fitFDistRobustly.R:98): mean(z, trim=0.5) is the median."""
        x = _variances(200, 4, 8, seed=10, outliers=5)
        wtp = (0.05, 0.5)
        _assert_matches(
            fit_f_dist_robustly(x, 4, winsor_tail_p=wtp), _r_fit(x, 4, winsor_tail_p=wtp)
        )

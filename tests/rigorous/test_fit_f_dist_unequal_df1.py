"""
Rigorous per-branch parity tests for pylimma.squeeze_var.fit_f_dist_unequal_df1.

Each test exercises a branch of fitFDistUnequalDF1() in R limma's
fitFDistUnequalDF1.R against a live R subprocess, comparing every returned
slot. Fixture tests at rtol 1e-6 are in TestFitFDistUnequalDF1BranchParity
(tests/test_r_parity.py).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylimma.squeeze_var import fit_f_dist_unequal_df1

from ..helpers import limma_available, run_r_code

pytestmark = pytest.mark.skipif(not limma_available(), reason="R/limma not available")

SLOTS = ("scale", "df2", "df2_outlier", "df2_shrunk")


# limma 3.66.0 sets PriorWeights before `if(n.informative==2) prior.weights <- NULL`,
# so an existing zero weight leaves PriorWeights TRUE with NULL weights and a NaN
# scale. The patch recomputes the flag after the reset, which is what the reset
# intends (see docs/validation/known_differences.rst).
_PATCH = (
    "src <- deparse(limma::fitFDistUnequalDF1)\n"
    "at <- grep('prior.weights <- NULL', src, fixed = TRUE)\n"
    "stopifnot(length(at) == 1)\n"
    "src <- append(src, 'PriorWeights <- !is.null(prior.weights)', after = at)\n"
    "fitFDistUnequalDF1 <- eval(parse(text = src))\n"
    "environment(fitFDistUnequalDF1) <- asNamespace('limma')\n"
)


def _r_fit(x, df1, covariate=None, robust=False, prior_weights=None, patched=False):
    """Every slot of R's fitFDistUnequalDF1, keyed by pylimma's slot names.

    With patched=True, the minimally patched function described at _PATCH.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        args = {"x": x, "df1": df1, "covariate": covariate, "prior_weights": prior_weights}
        code = {}
        for name, value in args.items():
            if value is None:
                code[name] = "NULL"
                continue
            pd.DataFrame({"v": np.atleast_1d(np.asarray(value, dtype=float))}).to_csv(
                tmp / f"{name}.csv", index=False, na_rep="NA"
            )
            code[name] = f"read.csv('{tmp}/{name}.csv')$v"
        run_r_code(
            "suppressMessages(library(limma))\n"
            + (_PATCH if patched else "")
            + f"fit <- fitFDistUnequalDF1({code['x']}, {code['df1']}, covariate = {code['covariate']}, "
            f"robust = {'TRUE' if robust else 'FALSE'}, prior.weights = {code['prior_weights']})\n"
            f"for (nm in names(fit)) write.csv(data.frame(v = fit[[nm]]), "
            f"file.path('{tmp}', paste0(gsub('.', '_', nm, fixed = TRUE), '.csv')), row.names = FALSE)\n"
        )
        return {
            slot: pd.read_csv(tmp / f"{slot}.csv")["v"].to_numpy(dtype=float)
            for slot in SLOTS
            if (tmp / f"{slot}.csv").exists()
        }


def _assert_matches(py, r, rtol=1e-8):
    """Each R slot equal at `rtol` (NaN where R has NaN), then the same slots as R."""
    for slot, r_value in r.items():
        if py.get(slot) is None:
            continue
        py_value = np.atleast_1d(np.asarray(py[slot], dtype=float))
        if r_value.size > 1 and py_value.size == 1:
            py_value = np.full(r_value.size, py_value[0])
        np.testing.assert_allclose(py_value, r_value, rtol=rtol, atol=1e-300, err_msg=slot)
    assert sorted(k for k in SLOTS if py.get(k) is not None) == sorted(r), (sorted(py), sorted(r))


def _data(n, seed, outliers=0):
    """Variances with genewise df1 between 1 and 8, first `outliers` inflated."""
    rng = np.random.default_rng(seed)
    df1 = rng.integers(1, 9, size=n).astype(float)
    x = 0.5 * rng.f(df1, 10)
    x[:outliers] *= 30
    return x, df1


class TestRigorousFitFDistUnequalDF1:
    def test_non_robust(self):
        """Exercises R-B10 (weighted mean), R-B12/R-B13 (optimize) and R-B14."""
        x, df1 = _data(300, seed=1)
        _assert_matches(fit_f_dist_unequal_df1(x, df1), _r_fit(x, df1))

    def test_robust_with_outliers(self):
        """Exercises R-B15, R-B17 (refit with FDR weights), R-B20 and R-B21."""
        x, df1 = _data(300, seed=2, outliers=6)
        r = _r_fit(x, df1, robust=True)
        assert "df2_shrunk" in r
        _assert_matches(fit_f_dist_unequal_df1(x, df1, robust=True), r)

    def test_robust_covariate_trend(self):
        """Exercises R-B11 (loessFit trend, default span) with R-B17."""
        x, df1 = _data(400, seed=3, outliers=6)
        cov = np.random.default_rng(3).uniform(2, 12, 400)
        x = x * np.exp(-0.2 * cov)
        r = _r_fit(x, df1, covariate=cov, robust=True)
        _assert_matches(fit_f_dist_unequal_df1(x, df1, covariate=cov, robust=True), r)

    def test_robust_without_outliers(self):
        """Exercises R-B16 (fitFDistUnequalDF1.R:120): min(FDR) == 1 returns the
        non-robust estimates."""
        x, df1 = _data(300, seed=4)
        r = _r_fit(x, df1, robust=True)
        assert "df2_shrunk" not in r
        _assert_matches(fit_f_dist_unequal_df1(x, df1, robust=True), r)

    @pytest.mark.parametrize("zero_weight", ["na_x", "small_df1"])
    def test_two_informative_with_prior_weights(self, zero_weight):
        """Exercises R-B9 (fitFDistUnequalDF1.R:48-59) when prior weights were
        created by R-B6 (NA x) or R-B7 (df1 < 0.01) before n.informative == 2
        sets prior.weights to NULL.

        Deliberate divergence: limma 3.66.0 returns a NaN scale here (asserted
        below); pylimma follows the minimally patched R.
        """
        x = np.array([0.5, 1.5, 0.0, 0.0])
        df1 = np.array([4.0, 6.0, 4.0, 4.0])
        if zero_weight == "na_x":
            x[3] = np.nan
        else:
            df1[3] = 0.001
        assert np.isnan(_r_fit(x, df1)["scale"]).all()
        _assert_matches(fit_f_dist_unequal_df1(x, df1), _r_fit(x, df1, patched=True))

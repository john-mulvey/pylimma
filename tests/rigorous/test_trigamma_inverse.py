"""
Rigorous per-branch parity tests for pylimma.utils.trigamma_inverse.

Each test exercises a branch of trigammaInverse() in R limma's fitFDist.R
against a live R subprocess. Branches R-B2 to R-B7 are also covered by the
fixture test TestTrigammaInverseExtremesRParity in tests/test_r_parity.py.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pylimma.utils import trigamma_inverse

from ..helpers import compare_arrays, limma_available, run_r_code, run_r_comparison

pytestmark = pytest.mark.skipif(not limma_available(), reason="R/limma not available")


def _r_error(r_expr: str) -> str:
    """R's error message for evaluating ``trigammaInverse(<r_expr>)``, or "" if none."""
    out = run_r_code(
        "suppressMessages(library(limma))\n"
        f"cat(tryCatch({{ trigammaInverse({r_expr}); '' }}, "
        "error = function(e) conditionMessage(e)))\n"
    )
    return out.strip()


class TestRigorousTrigammaInverse:
    @pytest.mark.parametrize(
        "py_value, r_expr",
        [(True, "TRUE"), (np.array([True, False]), "c(TRUE, FALSE)")],
    )
    def test_logical_input_errors(self, py_value, r_expr):
        """Exercises R-B1 (fitFDist.R:7): `if(!is.numeric(x)) stop(...)`.

        R treats logical input as non-numeric and stops.
        """
        assert _r_error(r_expr) == "Non-numeric argument to mathematical function"
        with pytest.raises(TypeError, match="Non-numeric argument to mathematical function"):
            trigamma_inverse(py_value)

    def test_character_input_errors(self):
        """Exercises R-B1 (fitFDist.R:7) with character input."""
        assert _r_error("'a'") == "Non-numeric argument to mathematical function"
        with pytest.raises((TypeError, ValueError)):
            trigamma_inverse("a")

    def test_zero_returns_inf_without_warning(self):
        """Exercises R-B6 (fitFDist.R:32-38) at x = 0: `y[omit] <- 1/x[omit]`.

        R's 1/0 is Inf and raises no warning.
        """
        assert _r_error("0") == ""
        r = run_r_code(
            "suppressMessages(library(limma))\n"
            "w <- NULL\n"
            "y <- withCallingHandlers(trigammaInverse(0), warning = function(e) "
            "{ w <<- conditionMessage(e); invokeRestart('muffleWarning') })\n"
            "cat(y, is.null(w))\n"
        ).split()
        assert r == ["Inf", "TRUE"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert trigamma_inverse(0.0) == np.inf

    def test_newton_range_matches_r(self):
        """Exercises R-B7 (fitFDist.R:43-50) over the whole Newton range [1e-6, 1e7].

        R-B8 (fitFDist.R:51-53, iteration limit) is not reached: both
        implementations converge within 15 iterations on this grid, so no
        warning is expected from either.
        """
        x = 10 ** np.linspace(-6, 7, 2001)
        r = run_r_comparison(
            py_data={"x": x},
            r_code_template=(
                "suppressMessages(library(limma))\n"
                "x <- read.csv('{tmpdir}/x.csv', row.names = 1)[, 1]\n"
                "w <- NULL\n"
                "y <- withCallingHandlers(trigammaInverse(x), warning = function(e) "
                "{{ w <<- conditionMessage(e); invokeRestart('muffleWarning') }})\n"
                "warned <- !is.null(w)\n"
            ),
            output_vars=["y", "warned"],
        )
        assert not bool(np.ravel(r["warned"])[0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            py = trigamma_inverse(x)
        cmp = compare_arrays(np.ravel(r["y"]), py, rtol=1e-8)
        assert cmp["match"], cmp

    def test_integer_input_matches_r(self):
        """Exercises R-B1's numeric pass-through for integer input (is.numeric(1L) is TRUE)."""
        r = run_r_code(
            "suppressMessages(library(limma))\ncat(sprintf('%.17g', trigammaInverse(1:3)))\n"
        )
        np.testing.assert_allclose(
            trigamma_inverse(np.array([1, 2, 3])), [float(v) for v in r.split()], rtol=1e-8
        )

"""Tests for pylimma weights module (errors and interface; R's behaviour for
each case is recorded in tests/fixtures/generate_all_fixtures.R)."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylimma.weights import array_weights, modify_weights

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _fixture(name):
    return pd.read_csv(FIXTURES_DIR / f"R_{name}.csv").to_numpy()


@pytest.mark.parametrize("method", ["genebygene", "reml"])
def test_singular_information_matrix_raises(method):
    # R: solve() stops with "system is exactly singular" ("arrayWeights
    # scoring: a singular"); pylimma returned weights of 1
    y = _fixture("aws_singular_input")
    with pytest.raises(ValueError, match="information matrix of the array weights model is singular"):
        array_weights(y, design=np.eye(9)[:, :7], method=method, prior_n=0)


@pytest.mark.parametrize("weights", [None, np.ones((50, 6))], ids=["reml", "prwts"])
def test_reml_warns_at_the_iteration_limit(weights):
    y = _fixture("aws_maxiter_input")
    expected = _fixture("aws_maxiter2")[:, 0 if weights is None else 1]
    with pytest.warns(UserWarning, match="iteration limit reached"):
        got = array_weights(y, method="reml", weights=weights, maxiter=2)
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        array_weights(y, method="reml", weights=weights)


@pytest.mark.parametrize("method", ["genebygene", "reml"])
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"prior_n": np.inf}, "prior_n must be a finite number of 0 or more"),
        ({"prior_n": -1}, "prior_n must be a finite number of 0 or more"),
        ({"maxiter": 0}, "maxiter must be at least 1"),
        ({"maxiter": 2.5}, "maxiter must be a whole number"),
    ],
)
def test_invalid_prior_n_and_maxiter_raise(method, kwargs, message):
    # R: prior.n = Inf gives NaN weights or stops in lm.wfit; 1:0 iterates
    with pytest.raises(ValueError, match=message):
        array_weights(_fixture("aws_maxiter_input"), method=method, **kwargs)


def test_modify_weights_default_weights_are_one():
    np.testing.assert_array_equal(
        modify_weights(status=["a", "b"], values=["a"], multipliers=0), [[0.0], [1.0]]
    )
    with pytest.raises(TypeError, match='argument "values" is missing'):
        modify_weights(status=["a", "b"])

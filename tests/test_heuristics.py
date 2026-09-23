"""Test heurisitics."""

import pytest
import numpy as np
from riskslim.loss_functions.log_loss import log_loss_value_from_scores
from riskslim.heuristics import sequential_rounding, discrete_descent

@pytest.mark.parametrize('c0_weight', [1, .1])
def test_sequential_rounding(generated_normal_data, c0_weight):

    Z = generated_normal_data['Z'][0]
    rng = np.random.default_rng(0)
    weights = generated_normal_data['weights_true'][0] + (rng.random(Z.shape[-1]) * .5)
    weights_before_rounding = weights.copy()
    C_0 = np.ones_like(weights) * c0_weight

    get_L0_penalty = lambda weights: np.sum(
        C_0 * (weights != 0.0)
    )

    weights_rounded, best_objval, early_stop_flag = sequential_rounding(
        weights, Z, C_0, log_loss_value_from_scores, get_L0_penalty
    )

    np.testing.assert_array_equal(weights, weights_before_rounding)
    if not early_stop_flag:
        np.testing.assert_array_equal(weights_rounded, np.rint(weights_rounded))

    if c0_weight == 1:
        # Large penalty gives all zeros
        assert np.all(weights_rounded == 0)

    weights_rand = rng.random(12)
    objval_rand = log_loss_value_from_scores(Z.dot(weights_rand)) + get_L0_penalty(weights_rand)

    assert not early_stop_flag
    assert best_objval > 0 and best_objval < objval_rand


def test_discrete_descent(generated_normal_data):

    Z = generated_normal_data['Z'][0]
    weights = generated_normal_data['weights_true'][0]
    weights = np.ones(len(Z[0]))
    C_0 = np.ones_like(weights)

    weights_lb = np.ones_like(weights) * -5
    weights_ub = np.ones_like(weights) * 5

    get_L0_penalty = lambda weights: np.sum(
        C_0 * (weights != 0.0)
    )
    descent_dimensions = np.arange(len(weights)).astype(int)

    weights_discrete, base_loss, base_objval = discrete_descent(
        weights, Z, C_0, weights_ub, weights_lb, get_L0_penalty, log_loss_value_from_scores,
        descent_dimensions=descent_dimensions
    )

    assert base_loss < base_objval
    assert base_loss < log_loss_value_from_scores(Z.dot(weights))
    assert log_loss_value_from_scores(Z.dot(weights_discrete)).astype(np.float32).round(6) == \
        base_loss.astype(np.float32).round(6)
    assert len(weights) == len(weights_discrete)

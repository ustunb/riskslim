"""Test heurisitics."""

import pytest
import numpy as np
from riskslim.loss_functions.log_loss import log_loss_value_from_scores
from riskslim.heuristics import sequential_rounding, discrete_descent

@pytest.mark.parametrize('c0_weight', [1, .1])
def test_sequential_rounding(generated_normal_data, c0_weight):

    Z = generated_normal_data['Z'][0]
    rho = generated_normal_data['rho_true'][0] + (np.random.rand(Z.shape[-1]) * .5)
    C_0 = np.ones_like(rho) * c0_weight

    get_L0_penalty = lambda rho: np.sum(
        C_0 * (rho != 0.0)
    )

    rho_rounded, best_objval, early_stop_flag = sequential_rounding(
        rho, Z, C_0, log_loss_value_from_scores, get_L0_penalty
    )

    if c0_weight == 1:
        # Large penalty gives all zeros
        assert np.all(rho_rounded == 0)
    else:
        assert np.all(rho_rounded == rho)

    rho_rand = np.random.rand(12)
    objval_rand = log_loss_value_from_scores(Z.dot(rho_rand)) + get_L0_penalty(rho_rand)

    assert not early_stop_flag
    assert best_objval > 0 and best_objval < objval_rand


def test_discrete_descent(generated_normal_data):

    Z = generated_normal_data['Z'][0]
    rho = generated_normal_data['rho_true'][0]
    rho = np.ones(len(Z[0]))
    C_0 = np.ones_like(rho)

    rho_lb = np.ones_like(rho) * -5
    rho_ub = np.ones_like(rho) * 5

    get_L0_penalty = lambda rho: np.sum(
        C_0 * (rho != 0.0)
    )
    descent_dimensions = np.arange(len(rho)).astype(int)

    rho_discrete, base_loss, base_objval = discrete_descent(
        rho, Z, C_0, rho_ub, rho_lb, get_L0_penalty, log_loss_value_from_scores,
        descent_dimensions=descent_dimensions
    )

    assert base_loss < base_objval
    assert base_loss < log_loss_value_from_scores(Z.dot(rho))
    assert log_loss_value_from_scores(Z.dot(rho_discrete)).astype(np.float32).round(6) == \
        base_loss.astype(np.float32).round(6)
    assert len(rho) == len(rho_discrete)

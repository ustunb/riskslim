"""Test warmstart functions."""

import pytest
import numpy as np
from riskslim.mip import create_risk_slim
from riskslim.coefficient_set import CoefficientSet
from riskslim.loss_functions.log_loss import (
    log_loss_value, log_loss_value_and_slope, log_loss_value_from_scores
)
from riskslim.defaults import DEFAULT_CPA_SETTINGS
from riskslim.solution_pool import SolutionPool
from riskslim.warmstart import (
    run_standard_cpa, round_solution_pool, sequential_round_solution_pool,
    discrete_descent_solution_pool
)


@pytest.mark.parametrize("cpa_type", ["cvx", "ntree", None])
@pytest.mark.parametrize("maxes",
    [
        (0, 1000, 1000), (1000, 1000, 1000), (1000, 2, 1000), (1000, 1000, 0)
    ]
)
def test_run_standard_cpa(generated_normal_data, cpa_type, maxes):
    """Test cutting plane algorithm."""
    Z = generated_normal_data['Z'][0]
    variable_names = generated_normal_data['variable_names'].copy()

    vtypes = ['C'] * len(variable_names)

    coef_set = CoefficientSet(variable_names, vtype=vtypes)

    mip_settings = {
        "C_0": .1,
        "coef_set": coef_set,
        "include_auxillary_variable_for_L0_norm": True,
        "include_auxillary_variable_for_objval": True,
        "relax_integer_variables": False,
        "drop_variables": True,
    }

    mip, indices = create_risk_slim(coef_set, mip_settings)

    compute_loss_cut = lambda rho: log_loss_value_and_slope(Z, rho)

    settings = DEFAULT_CPA_SETTINGS.copy()
    settings["type"] = cpa_type
    settings['save_progress'] = True
    settings['max_iterations'] = maxes[0]
    settings['min_iterations_before_coefficient_gap_check'] = maxes[1]
    settings['max_runtime_per_iteration'] = maxes[2]

    stats, cuts, pool = run_standard_cpa(
        mip,
        indices,
        log_loss_value,
        compute_loss_cut,
        settings=settings,
        print_flag=True
    )

    assert len(stats['solution']) == len(coef_set)
    assert stats['loss_min'] < stats['loss_max']
    assert stats['objval_min'] < stats['objval_max']
    assert stats['upperbound'] > stats['lowerbound']
    assert len(cuts['coefs']) + 1 == len(cuts['lhs']) +1 == stats['n_iterations'] == len(pool)
    assert np.all(pool.objvals >= 0)


def test_round_solution_pool(generated_normal_data):

    Z = generated_normal_data['Z'][0].copy()
    rho = generated_normal_data['rho_true'][0].copy()

    # Add small amount of noise to move from ints to floats
    inds = np.where(rho != 0.)[0]
    rho[inds] = rho[inds] + (np.random.rand(len(inds))) * .1

    variable_names = generated_normal_data['variable_names'].copy()
    coef_set = CoefficientSet(variable_names)

    constraints = {
        "min_size": 0,
        "max_size": 10,
        "coef_set": coef_set,
    }

    objvals = log_loss_value(Z, rho)

    pool = SolutionPool({'objvals': objvals, 'solutions':rho})

    rounded_pool, total_runtime, total_rounded = round_solution_pool(pool, constraints)


    assert np.all(rounded_pool.solutions[0] == generated_normal_data['rho_true'][0])
    assert total_runtime > 0
    assert total_rounded > 0


def test_sequential_round_solution_pool(generated_normal_data):
    Z = generated_normal_data['Z'][0].copy()
    rho = generated_normal_data['rho_true'][0].copy()

    # Add small amount of noise
    inds = np.where(rho != 0.)[0]
    rho = rho + (np.random.rand(len(Z[0]))) * .1

    # Pool
    objvals = log_loss_value(Z, rho)
    pool = SolutionPool({'objvals': objvals, 'solutions':rho})

    # Args
    C_0 = np.zeros(len(Z[0])) + .1

    get_L0_penalty = lambda rho: np.sum(
        C_0 * (rho != 0.0)
    )

    compute_loss_from_scores = (
        lambda scores: log_loss_value_from_scores(scores)
    )

    rounded_pool, total_runtime, total_rounded = sequential_round_solution_pool(
        pool,
        Z,
        C_0,
        compute_loss_from_scores,
        get_L0_penalty
    )

    sol = rounded_pool.solutions[0]
    assert (sol -  generated_normal_data['rho_true'][0].copy()).mean() < .2
    assert total_runtime > 0
    assert total_rounded > 0


@pytest.mark.parametrize("non_integral", [True, False])
def test_discrete_descent_solution_pool(generated_normal_data, non_integral):

    Z = generated_normal_data['Z'][0].copy()
    rho = generated_normal_data['rho_true'][0].copy()
    rho[0] += 1
    # Add small amount of noise to move from ints to floats
    inds = np.where(rho != 0.)[0]

    if non_integral:
        rho[inds] = rho[inds] + (np.random.rand(len(inds)))

    variable_names = generated_normal_data['variable_names'].copy()
    coef_set = CoefficientSet(variable_names)

    constraints = {
        "min_size": 0,
        "max_size": 10,
        "coef_set": coef_set,
    }

    objvals = log_loss_value(Z, rho)

    pool = SolutionPool({'objvals': objvals, 'solutions':rho})

    C_0 = np.zeros(len(Z[0])) + .1e-16

    get_L0_penalty = lambda rho: np.sum(
        C_0 * (rho != 0.0)
    )

    compute_loss_from_scores = (
        lambda scores: log_loss_value_from_scores(scores)
    )

    polished_pool, total_runtime, total_polished = discrete_descent_solution_pool(
        pool,
        Z,
        C_0,
        constraints,
        get_L0_penalty,
        compute_loss_from_scores
    )

    if not non_integral:
        assert polished_pool.objvals[0] < pool.objvals[0]
        assert total_runtime > 0
        assert total_polished > 0
    else:
        assert len(polished_pool) == 0

"""Test RiskSlim fitting."""

import pytest
import numpy as np
from cplex import Cplex
from sklearn.datasets import load_breast_cancer
from riskslim.coefficient_set import CoefficientSet
from riskslim.utils import Stats
from riskslim.bounds import Bounds
from riskslim.classifier import RiskSLIMClassifier


@pytest.mark.parametrize('use_coef_set', [True, False])
def test_fit_builds_optimizer(generated_normal_data, use_coef_set):
    """Test RiskSLIMClassifier fit initalization."""
    X = generated_normal_data['X'][0]
    y = np.ravel(generated_normal_data['y'])
    variable_names = generated_normal_data['variable_names']

    coef_names = ['(Intercept)'] + variable_names
    coef_set = CoefficientSet(coef_names) if use_coef_set else None
    rs = RiskSLIMClassifier(
        coef_set=coef_set,
        max_size=10,
        variable_names=variable_names,
        outcome_name=generated_normal_data['outcome_name'],
        max_runtime=2,
        display_cplex_progress=False,
        round_flag=False,
        polish_flag=False,
    )

    rs.fit(X, y)

    # Checks
    assert isinstance(rs.optimizer.coef_set, CoefficientSet)
    assert isinstance(rs.optimizer.min_coef, np.ndarray)
    assert isinstance(rs.optimizer.max_coef, np.ndarray)

    assert isinstance(rs.optimizer.mip, Cplex)
    assert isinstance(rs.optimizer.mip_indices, dict)

    assert isinstance(rs.optimizer.bounds, Bounds)
    assert isinstance(rs.optimizer.stats, Stats)

    assert rs.optimizer.data.Z.shape == rs.optimizer.data.X.shape


@pytest.mark.parametrize('loss_computation', ['normal'])
def test_fit_sets_up_loss_functions(generated_normal_data, loss_computation):
    """Test setting up loss functions."""

    X = generated_normal_data['X'][0]
    y = np.ravel(generated_normal_data['y'])
    rho = np.insert(generated_normal_data['rho'], 0, 0.0)
    variable_names = generated_normal_data['variable_names']

    rs = RiskSLIMClassifier(
        max_size=10,
        variable_names=variable_names,
        outcome_name=generated_normal_data['outcome_name'],
        loss_computation=loss_computation,
        max_runtime=2,
        display_cplex_progress=False,
        round_flag=False,
        polish_flag=False,
    )

    rs.fit(X, y)

    rho = np.require(rho, requirements = ['F'])

    loss, slope = rs.optimizer.compute_loss_cut(rho)
    loss_real, slope_real = rs.optimizer.compute_loss_cut_real(rho)

    assert loss == loss_real
    assert np.all(slope == slope_real)


def test_time_limited_fit_matches_solver_objective():
    """Test the raw CPLEX objective for a time-limited fit."""
    data = load_breast_cancer()
    X = (data.data > np.median(data.data, axis=0)).astype(float)
    y = data.target
    c0_value = 1e-6
    rs = RiskSLIMClassifier(
        max_coef=5,
        max_size=5,
        c0_value=c0_value,
        variable_names=list(data.feature_names),
        outcome_name="diagnosis",
        max_runtime=2,
        cplex_randomseed=0,
        verbose=False,
    )

    rs.fit(X, y)

    rho = np.r_[rs.intercept_, rs.coef_]
    signed_scores = (1.0 - 2.0 * y) * (rho[0] + X @ rho[1:])
    mean_logistic_loss = float(np.mean(np.logaddexp(0.0, signed_scores)))
    expected_objective = mean_logistic_loss + c0_value * np.count_nonzero(rho[1:])
    raw_objective = rs.optimizer.solution.get_objective_value()

    np.testing.assert_allclose(raw_objective, expected_objective, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize('use_rounding', [True, False])
@pytest.mark.parametrize('polishing_after', [True, False])
def test_warmstart_fills_solution_pool(generated_normal_data, use_rounding, polishing_after):
    """Test RiskSLIMClassifier fitting."""
    X = generated_normal_data['X']
    y = np.ravel(generated_normal_data['y'])
    variable_names = generated_normal_data['variable_names']

    # Constraints
    coef_names = ['(Intercept)'] + variable_names
    ub = np.array([5.] * len(coef_names))
    lb = np.array([-5.] * len(coef_names))

    lb[0] = 0.
    ub[0] = 0.

    coef_set = CoefficientSet(variable_names=coef_names, lb=lb, ub=ub)

    # Test warm start
    if use_rounding or polishing_after:
        warmstart_settings = {
            'display_cplex_progress': False,
            'use_rounding': use_rounding,
            'use_sequential_rounding': use_rounding,
            'polishing_after': polishing_after
        }
    else:
        warmstart_settings = {}

    settings = {
        'initialization_flag': True,
        'max_runtime': 2,
        'display_cplex_progress': False,
        'round_flag': False,
        'polish_flag': False,
        'init_max_runtime': 2,
        'init_display_progress': False,
        **{'init_' + key: value for key, value in warmstart_settings.items()},
    }
    rs = RiskSLIMClassifier(
        coef_set=coef_set,
        max_size=5,
        variable_names=variable_names,
        outcome_name=generated_normal_data['outcome_name'],
        **settings,
    )

    rs.fit(X[0], y)

    assert len(rs.optimizer.pool) > 0

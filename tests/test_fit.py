"""Test RiskSlim fitting."""

import pytest
import numpy as np
from cplex import Cplex
from sklearn.datasets import load_breast_cancer
from riskslim.coefficient_set import CoefficientSet
from riskslim.utils import Stats
from riskslim.bounds import Bounds
from riskslim.classifier import RiskSLIMClassifier


@pytest.mark.parametrize('init_coef', [True, False])
def test_RiskSLIMClassifier_init(init_coef):
    """Test RiskSLIMClassifier initialization."""
    variable_names = ['variable_' + str(i) for i in range(10)]

    coef_set = CoefficientSet(variable_names) if init_coef else None

    max_size=10

    rs = RiskSLIMClassifier(coef_set=coef_set, max_size=max_size)

    assert rs.max_size == max_size
    assert rs.optimizer is None
    assert rs.coef_set is coef_set
    assert rs.variable_names is None
    assert rs.c0_value == 1e-6
    assert not rs.fitted


@pytest.mark.parametrize('use_coef_set', [True, False])
def test_RiskSLIMClassifier_init_fit(generated_normal_data, use_coef_set):
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
def test_RiskSLIMClassifier_init_loss(generated_normal_data, loss_computation):
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


def test_RiskSLIMClassifier_raw_objective_matches_returned_model():
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
def test_RiskSLIMClassifier_warmstart(generated_normal_data, use_rounding, polishing_after):
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


@pytest.mark.parametrize('max_size, expected_row', [
    (0, 'NO VARIABLES: SCORE IS 0'),
    (2, 'ADD POINTS FROM ROWS 1 to'),
])
def test_RiskSLIMClassifier_prints_score_table(max_size, expected_row):
    """Printing a fitted classifier shows its score table, including an intercept-only model."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(80, 3)).astype(float)
    y = (X[:, 0] + X[:, 1] + rng.random(80) > 1.2).astype(int)
    rs = RiskSLIMClassifier(max_size=max_size, verbose=False, max_runtime=10, cplex_randomseed=0)

    rs.fit(X, y)

    table = str(rs)
    assert table.startswith('+')
    assert expected_row in table
    assert repr(rs) == table

"""Test RiskSlim fitting."""

from warnings import warn
import pytest
import numpy as np
from scipy import sparse
from cplex import Cplex
from riskslim.coefficient_set import CoefficientSet
from riskslim.utils import Stats
from riskslim.bounds import Bounds
from riskslim.classifier import RiskSLIMClassifier


@pytest.mark.parametrize('init_coef', [True, False])
def test_RiskSLIMClassifier_init(init_coef):
    """Test RiskSLIMClassifier initialization."""
    variable_names = ['variable_' + str(i) for i in range(10)]

    coef_set = CoefficientSet(variable_names) if init_coef else None

    min_size=0
    max_size=10

    rs = RiskSLIMClassifier(coef_set=coef_set, min_size=min_size, max_size=max_size)

    assert rs.min_size == min_size
    assert rs.max_size == max_size
    assert rs.X is None
    assert rs.y is None

    type_check = lambda i, expected_type : isinstance(i, expected_type)

    if init_coef:
        expected_type = (float, int, np.ndarray)
        assert rs.variable_names == variable_names
    else:
        expected_type = (type(None), float)
        assert type_check(rs.variable_names, expected_type)

    assert type_check(rs.rho_min, expected_type)
    assert type_check(rs.rho_max, expected_type)
    assert type_check(rs.c0_value, expected_type)
    assert type_check(rs.L0_reg_ind, expected_type)
    assert type_check(rs.C_0, expected_type)
    assert type_check(rs.C_0_nnz, expected_type)


@pytest.mark.parametrize('use_coef_set', [True, False])
def test_RiskSLIMClassifier_init_fit(generated_normal_data, use_coef_set):
    """Test RiskSLIMClassifier fit initalization."""
    X = generated_normal_data['X'][0]
    y = generated_normal_data['y']
    variable_names = generated_normal_data['variable_names']

    coef_set = CoefficientSet(variable_names) if use_coef_set else None
    rs = RiskSLIMClassifier(coef_set=coef_set, min_size=0, max_size=10)

    # Load data into attribute
    #   this is normally done in .fit
    rs.X = X
    rs.y = y
    rs.variable_names = None
    rs.outcome_name = None
    rs.sample_weights = None

    rs.init_fit()
    rs.init_mip()

    # Checks
    assert isinstance(rs.coef_set, CoefficientSet)
    assert isinstance(rs.rho_min, np.ndarray)
    assert isinstance(rs.rho_max, np.ndarray)

    assert isinstance(rs.mip, Cplex)
    assert isinstance(rs.mip_indices, dict)

    assert isinstance(rs.bounds, Bounds)
    assert isinstance(rs.stats, Stats)

    assert (rs.Z.shape == rs.X.shape)


@pytest.mark.parametrize('loss_computation', ['fast', 'normal', 'weighted', 'lookup'])
def test_RiskSLIMClassifier_init_loss(generated_normal_data, loss_computation):
    """Test setting up loss functions."""

    X = generated_normal_data['X'][0]
    y = generated_normal_data['y']
    rho = generated_normal_data['rho']
    variable_names = generated_normal_data['variable_names']

    # Initalize
    if loss_computation == 'lookup':
        coef_set = CoefficientSet(variable_names, lb=-10, ub=10)
        coef_set.update_intercept_bounds(X=X, y=y, max_offset=0)
    else:
        coef_set = None

    settings = {'loss_computation': loss_computation}
    rs = RiskSLIMClassifier(coef_set=coef_set, min_size=0, max_size=10, settings=settings)

    # Load data into attribute
    #   this is normally done in .fit
    rs.X = X
    rs.y = y
    rs.variable_names = None
    rs.outcome_name = None
    rs.sample_weights = None

    if loss_computation == 'weighted':
        rs.sample_weights = np.random.rand(len(X))

    rs.init_fit()
    rs.init_mip()

    rs.coef_set = coef_set
    rs._init_loss_computation()

    Z = np.require(rs.Z, requirements = ['F'])
    rho = np.require(rho, requirements = ['F'])

    if loss_computation != 'lookup':
        pass
    else:
        assert rs.compute_loss(rho) == rs.compute_loss_real(rho)
        assert rs.compute_loss_from_scores(Z.dot(rho)) == rs.compute_loss_from_scores_real(Z.dot(rho))

    loss, slope = rs.compute_loss_cut(rho)
    loss_real, slope_real = rs.compute_loss_cut_real(rho)

    assert loss == loss_real
    assert np.all(slope == slope_real)


@pytest.mark.parametrize('use_rounding', [True, False])
@pytest.mark.parametrize('polishing_after', [True, False])
def test_RiskSLIMClassifier_warmstart(generated_normal_data, use_rounding, polishing_after):
    """Test RiskSLIMClassifier fitting."""
    X = generated_normal_data['X']
    y = generated_normal_data['y']
    variable_names = generated_normal_data['variable_names']

    # Constraints
    ub = np.array([5.] * len(variable_names))
    lb = np.array([-5.] * len(variable_names))

    lb[0] = 0.
    ub[0] = 0.

    coef_set = CoefficientSet(variable_names=variable_names, lb=lb, ub=ub)

    rs = RiskSLIMClassifier(coef_set=coef_set, min_size=0, max_size=5)

    # Load data into attribute
    #   this is normally done in .fit
    rs.X = X[0]
    rs.y = y
    rs.variable_names = None
    rs.outcome_name = None
    rs.sample_weights = None

    # Initalize fitting procedure
    rs.init_fit()
    rs.init_mip()

    # Test warm start
    if use_rounding or polishing_after:
        warmstart_settings = {
            'display_cplex_progress': False,
            'use_rounding': use_rounding,
            'use_sequential_rounding': use_rounding,
            'polishing_after': polishing_after
        }
    else:
        warmstart_settings = None

    rs.warmstart(warmstart_settings)

    assert rs.has_warmstart
    assert len(rs.pool) > 0


@pytest.mark.parametrize('polish_flag', [True, False])
def test_RiskSLIMClassifier_fit(generated_normal_data, polish_flag):
    """Test fitting RiskSLIMClassifier."""
    X = generated_normal_data['X'].copy()
    y = generated_normal_data['y'].copy()
    variable_names = generated_normal_data['variable_names'].copy()

    # Size of problem
    n_columns = 12
    n_iters = 25
    c0_value = 1e-2

    # Settings
    settings = {
        # LCPA Settings
        'max_runtime': 2,
        'max_tolerance': np.finfo('float').eps,
        'display_cplex_progress': False,
        'loss_computation': 'fast',
        'round_flag': False,
        'polish_flag': polish_flag,
        'chained_updates_flag': True,
        'add_cuts_at_heuristic_solutions': True,
        # Initialization
        'initialization_flag': True,
        'init_max_runtime': 60.0,
        'init_max_coefficient_gap': 5,
        'init_display_progress': False,
        # CPLEX Solver Parameters
        'cplex_randomseed': 0,
        'cplex_mipemphasis': 0,
    }

    solutions = np.zeros((n_iters, n_columns+1), dtype=np.int8)

    for ind in range(n_iters):

        # Initalize
        rs = RiskSLIMClassifier(max_size=10, max_coef=5, settings=settings, variable_names=variable_names)

        # Fit
        rs.fit(X[ind], y)
        assert rs.fitted

        # Get solutions
        solutions[ind] = np.insert(rs.coef_, 0, rs.intercept_)

    # Test accuracy between computed and persistent solution
    if polish_flag:
        # Persistent solution was computed with polish_flag = True
        thresh = 95
    else:
        thresh = 50

    solutions_persist = sparse.load_npz('riskslim/tests/solutions.npz').todense()
    percent_match = np.count_nonzero(solutions == solutions_persist)
    percent_match /= solutions.size
    percent_match *= 100

    if percent_match == 100.0:
        # Pass test
        pass
    elif percent_match >= thresh:
        # Wiggle room for machine precision, cplex updates, etc
        warn(f'Persistence array match is {round(percent_match, 3)}%.')
    elif percent_match < thresh:
        raise ValueError(f'Persistence array match is {round(percent_match, 3)}%.')

    # Test accessing properties
    assert isinstance(rs.coefficients, np.ndarray)
    assert len(rs.coefficients) == X[ind].shape[1]
    assert isinstance(rs.solution_info['solution'], np.ndarray)

    # Test scikit-learn methods
    y_pred = rs.predict(X[0])
    assert len(y_pred) == len(X[0])

    proba = rs.predict_proba(X)
    assert np.all((proba >= 0) & (proba <= 1))

    log_proba = rs.predict_log_proba(X)
    assert np.all(log_proba.max() <= 0)

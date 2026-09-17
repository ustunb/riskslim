"""Test MIP functions."""

import pytest
import numpy as np
from cplex import Cplex
from riskslim.coefficient_set import CoefficientSet
from riskslim.defaults import DEFAULT_CPLEX_SETTINGS
from riskslim.solution_pool import SolutionPool
from riskslim.loss_functions.log_loss import log_loss_value
from riskslim.mip import create_risk_slim, set_cplex_mip_parameters, add_mip_starts


@pytest.mark.parametrize("relax_integer_variables", [True, False])
def test_create_risk_slim(generated_normal_data, relax_integer_variables):
    """Testing setting up MIP in CPLEX."""

    variable_names = ['(Intercept)'] + generated_normal_data['variable_names'].copy()
    coef_set = CoefficientSet(variable_names)

    mip_settings = {
        "C_0": .1,
        "coef_set": coef_set,
        "include_auxillary_variable_for_L0_norm": True,
        "include_auxillary_variable_for_objval": True,
        "relax_integer_variables": relax_integer_variables,
        "drop_variables": True,
    }

    mip, indices = create_risk_slim(coef_set, mip_settings)

    assert isinstance(mip, Cplex)

    # Variables
    mip_vars = mip.variables.get_names()
    for i in range(len(variable_names)):
        assert 'rho_' + str(i) in mip_vars
        if i == 0:
            # Intercept doesn't have alpha
            assert 'alpha_' + str(i) not in mip_vars
        else:
            assert 'alpha_' + str(i) in mip_vars

    for other_var in ['loss', 'objval', 'L0_norm']:
        assert other_var in mip_vars

    # Constaints
    mip_constr = mip.linear_constraints.get_names()
    for i in range(1, len(variable_names)):
        assert 'L0_norm_lb_' + str(i) in mip_constr
        assert 'L0_norm_ub_' + str(i) in mip_constr
    assert 'objval_def' in mip_constr
    assert 'L0_norm_def' in mip_constr

    # Check indices
    assert indices['n_variables'] == len(mip_vars) == len(indices['names'])
    assert indices['n_constraints'] == len(mip_constr)
    assert len(indices['rho_names']) == len(indices['rho']) == len(variable_names)


@pytest.mark.parametrize("relax_integer_variables", [True, False])
def test_set_cplex_mip_parameters(generated_normal_data, relax_integer_variables):

    variable_names = ['(Intercept)'] + generated_normal_data['variable_names'].copy()

    if relax_integer_variables:
        vtypes = ['I'] * len(variable_names)
        vtypes[0] = 'C'
    else:
        vtypes = 'I'

    coef_set = CoefficientSet(variable_names, vtype=vtypes)

    mip_settings = {
        "C_0": .1,
        "coef_set": coef_set,
        "include_auxillary_variable_for_L0_norm": True,
        "include_auxillary_variable_for_objval": True,
        "relax_integer_variables": relax_integer_variables,
        "drop_variables": True,
    }

    mip, _ = create_risk_slim(coef_set, mip_settings)

    cplex_settings = DEFAULT_CPLEX_SETTINGS.copy()

    mip = set_cplex_mip_parameters(
        mip,
        cplex_settings,
        display_cplex_progress=False,
    )


def test_add_mip_starts(generated_normal_data):

    Z = np.insert(generated_normal_data['Z'][0], 0, 1.0, axis=1)
    variable_names = ['(Intercept)'] + generated_normal_data['variable_names'].copy()

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

    # Create CPLEX MIP
    mip, indices = create_risk_slim(coef_set, mip_settings)

    # Solutin pool
    solution = np.zeros_like(Z[0])
    objval = log_loss_value(Z, solution)
    pool = SolutionPool(len(Z[0]))
    pool.add(objvals=objval, solutions=solution)

    cpx = add_mip_starts(mip, indices, pool)
    assert cpx.MIP_starts.get_num() > 0

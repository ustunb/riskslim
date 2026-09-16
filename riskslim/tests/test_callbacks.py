"""Test callback classes."""

import pytest
import numpy as np
from riskslim.optimizer import RiskSLIMOptimizer
from riskslim.mip import create_risk_slim
from riskslim.solution_pool import FastSolutionPool
from riskslim.coefficient_set import CoefficientSet
from riskslim.heuristics import discrete_descent, sequential_rounding
from riskslim.callbacks import LossCallback, PolishAndRoundCallback
from riskslim.defaults import DEFAULT_LCPA_SETTINGS
from riskslim.data import ClassificationDataset

@pytest.mark.parametrize('cut_queue', [None, FastSolutionPool(12)])
@pytest.mark.parametrize('polish_queue', [None, FastSolutionPool(12)])
def test_losscallback(generated_normal_data, cut_queue, polish_queue):

    # Dataset
    X = generated_normal_data['X'][0]
    y = generated_normal_data['y']
    variable_names = generated_normal_data['variable_names']
    data = ClassificationDataset(X, y, variable_names=variable_names, outcome_name='outcome')

    # Create mip
    coef_set = CoefficientSet(data.variable_names)

    mip_settings =mip_settings = {
        "C_0": 1e-6,
        "coef_set": coef_set,
        "tight_formulation": DEFAULT_LCPA_SETTINGS["tight_formulation"],
        "drop_variables":DEFAULT_LCPA_SETTINGS["drop_variables"],
        "include_auxillary_variable_for_L0_norm": DEFAULT_LCPA_SETTINGS["include_auxillary_variable_for_L0_norm"],
        "include_auxillary_variable_for_objval": DEFAULT_LCPA_SETTINGS["include_auxillary_variable_for_objval"],
    }

    mip, indices = create_risk_slim(coef_set=coef_set, settings=mip_settings)

    # Create required attributes
    opt = RiskSLIMOptimizer(data, coef_set, 5)

    indices.update({"C_0_nnz": opt.C_0_nnz, "L0_reg_ind": opt.L0_reg_ind})

    loss_cb = mip.register_callback(LossCallback)

    loss_cb.initialize(
        indices=indices,
        stats=opt.stats,
        settings=DEFAULT_LCPA_SETTINGS,
        compute_loss_cut=opt.compute_loss_cut,
        get_alpha=opt.get_alpha,
        get_L0_penalty_from_alpha=opt.get_L0_penalty_from_alpha,
        initial_cuts=None,
        cut_queue=opt.cut_queue,
        polish_queue=opt.polish_queue,
        verbose=opt.verbose,
     )

    assert loss_cb.cut_queue is not None
    assert loss_cb.polish_queue is not None
    assert loss_cb.compute_loss_cut == opt.compute_loss_cut
    assert loss_cb.get_alpha ==  opt.get_alpha
    assert loss_cb.get_L0_penalty_from_alpha ==  opt.get_L0_penalty_from_alpha


def test_polish_and_round_callback(generated_normal_data):

    # Dataset
    X = generated_normal_data['X'][0]
    y = generated_normal_data['y']
    variable_names = generated_normal_data['variable_names']
    data = ClassificationDataset(X, y, variable_names=variable_names, outcome_name='outcome')

    # Create mip
    coef_set = CoefficientSet(data.variable_names)

    mip_settings =mip_settings = {
        "C_0": 1e-6,
        "coef_set": coef_set,
        "tight_formulation": DEFAULT_LCPA_SETTINGS["tight_formulation"],
        "drop_variables":DEFAULT_LCPA_SETTINGS["drop_variables"],
        "include_auxillary_variable_for_L0_norm": DEFAULT_LCPA_SETTINGS["include_auxillary_variable_for_L0_norm"],
        "include_auxillary_variable_for_objval": DEFAULT_LCPA_SETTINGS["include_auxillary_variable_for_objval"],
    }

    mip, indices = create_risk_slim(coef_set=coef_set, settings=mip_settings)

    # Create required attributes
    opt = RiskSLIMOptimizer(data, coef_set, 5)

    # Initialize solution queues
    opt.cut_queue = FastSolutionPool(12)
    opt.polish_queue = FastSolutionPool(12)

    polisher = lambda rho: discrete_descent(
        rho,
        opt.Z,
        opt.C_0,
        opt.rho_max,
        opt.rho_min,
        opt.get_L0_penalty,
        opt.compute_loss_from_scores,
        True,
    )

    rounder = lambda rho, cutoff: sequential_rounding(
        rho,
        opt.Z,
        opt.C_0,
        opt.compute_loss_from_scores_real,
        opt.get_L0_penalty,
        cutoff
    )

    polish_cb = mip.register_callback(PolishAndRoundCallback)

    polish_cb.initialize(
        indices=opt.mip_indices,
        control=opt.stats,
        settings=opt.settings,
        cut_queue=opt.cut_queue,
        polish_queue=opt.polish_queue,
        get_objval=opt.get_objval,
        get_L0_norm=opt.get_L0_norm,
        is_feasible=opt.is_feasible,
        polishing_handle=polisher,
        rounding_handle=rounder,
    )

    assert polish_cb.get_objval == opt.get_objval
    assert polish_cb.get_L0_norm == opt.get_L0_norm
    assert polish_cb.is_feasible == opt.is_feasible
    assert polish_cb.polishing_handle == polisher
    assert polish_cb.rounding_handle == rounder

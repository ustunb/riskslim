import os
import pprint

import pytest

import numpy as np
import pandas as pd
import riskslim

# Dataset Strategy
#
# variables:    binary, real,
# N+:           0, 1, >1
# N-:           0, 1, >1


# Testing Strategy
#
# loss_computation  normal, fast, lookup
# max_coefficient   0, 1, >1
# max_size      0, 1, >1
# max_offset        0, 1, Inf
# c0_value          eps, 1e-8, 0.01, C0_max
# sample_weights    no, yes
# w_pos             1.00, < 1.00, > 1.00
# initialization    on, off
# chained_updates   on, off
# polishing         on, off
# seq_rd            on, off

# data
data_name = "breastcancer"  # name of the data
data_dir = os.getcwd() + '/examples/data/'  # directory where datasets are stored
data_csv_file = data_dir + data_name + '_data.csv'  # csv file for the data
sample_weights_csv_file = None  # csv file of sample weights for the data (optional)

default_settings = {
    #
    'c0_value': 1e-6,
    'w_pos': 1.00,
    #
    # LCPA Settings
    'max_runtime': 300.0,                               # max runtime for LCPA
    'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
    'display_cplex_progress': True,                     # set to True to print CPLEX progress
    'loss_computation': 'normal',                       # how to compute the loss function ('normal','fast','lookup')
    'tight_formulation': True,                          # use a slightly formulation of surrogate MIP that provides a slightly improved formulation
    #
    # Other LCPA Heuristics
    'chained_updates_flag': True,                       # use chained updates
    'add_cuts_at_heuristic_solutions': True,            # add cuts at integer feasible solutions found using polishing/rounding
    #
    # LCPA Rounding Heuristic
    'round_flag': True,                                 # round continuous solutions with SeqRd
    'polish_rounded_solutions': True,                   # polish solutions rounded with SeqRd using DCD
    'rounding_tolerance': float('inf'),                 # only solutions with objective value < (1 + tol) are rounded
    'rounding_start_cuts': 0,                           # cuts needed to start using rounding heuristic
    'rounding_start_gap': float('inf'),                 # optimality gap needed to start using rounding heuristic
    'rounding_stop_cuts': 20000,                        # cuts needed to stop using rounding heuristic
    'rounding_stop_gap': 0.2,                           # optimality gap needed to stop using rounding heuristic
    #
    # LCPA Polishing Heuristic
    'polish_flag': True,                                # polish integer feasible solutions with DCD
    'polishing_tolerance': 0.1,                         # only solutions with objective value (1 + tol) are polished.
    'polishing_max_runtime': 10.0,                      # max time to run polishing each time
    'polishing_max_solutions': 5.0,                     # max # of solutions to polish each time
    'polishing_start_cuts': 0,                          # cuts needed to start using polishing heuristic
    'polishing_start_gap': float('inf'),                # min optimality gap needed to start using polishing heuristic
    'polishing_stop_cuts': float('inf'),                # cuts needed to stop using polishing heuristic
    'polishing_stop_gap': 5.0,                          # max optimality gap required to stop using polishing heuristic
    #
    # Initialization Procedure
    'initialization_flag': False,                       # use initialization procedure
    'init_display_progress': True,                      # show progress of initialization procedure
    'init_display_cplex_progress': False,               # show progress of CPLEX during intialization procedure
    #
    'init_max_runtime': 300.0,                          # max time to run CPA in initialization procedure
    'init_max_iterations': 10000,                       # max # of cuts needed to stop CPA
    'init_max_tolerance': 0.0001,                       # tolerance of solution to stop CPA
    'init_max_runtime_per_iteration': 300.0,            # max time per iteration of CPA
    'init_max_cplex_time_per_iteration': 10.0,          # max time per iteration to solve surrogate problem in CPA
    #
    'init_use_sequential_rounding': True,               # use SeqRd in initialization procedure
    'init_sequential_rounding_max_runtime': 30.0,       # max runtime for SeqRd in initialization procedure
    'init_sequential_rounding_max_solutions': 5,        # max solutions to round using SeqRd
    'init_polishing_after': True,                       # polish after rounding
    'init_polishing_max_runtime': 30.0,                 # max runtime for polishing
    'init_polishing_max_solutions': 5,                  # max solutions to polish
    #
    # CPLEX Solver Parameters
    'cplex_randomseed': 0,                              # random seed
    'cplex_mipemphasis': 0,                             # cplex MIP strategy
}


@pytest.mark.parametrize('max_coefficient', [5])
@pytest.mark.parametrize('max_size', [0, 1, 5])
@pytest.mark.parametrize('max_offset', [0, 50])
def test_risk_slim(max_coefficient, max_size, max_offset):

    # Load data
    df = pd.read_csv(data_csv_file)

    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    N, P = X.shape

    # Offset value
    coef_set = riskslim.CoefficientSet(
        variable_names=list(df.columns)[1:],
        lb=-max_coefficient, ub=max_coefficient
    )

    coef_set.update_intercept_bounds(
        X = X, y = y, max_offset=max_offset, max_size = max_size
    )

    # Create constraint dictionary
    trivial_max_size = P - np.sum(coef_set.C_0j == 0)
    max_size = min(max_size, trivial_max_size)

    # Train model using lattice_cpa
    rs = riskslim.RiskSLIMClassifier(
        coef_set=coef_set, min_size=0, max_size=max_size, settings=default_settings
    )
    rs.fit(X, y)

    # Model info contains key results
    pprint.pprint(rs.solution_info)


    assert rs.min_size == rs.bounds.min_size == 0
    assert rs.max_size == rs.bounds.max_size == max_size
    assert rs.coef_set == coef_set

    # Each column of X has a rho and alpha (except for intercept,
    #   which doesn't have an alpha). There are 3 additional parameters:
    #   loss, objval, L0_norm
    assert (len(X[0]) * 2) - 1 + 3 == rs.mip_indices['n_variables']

    assert True

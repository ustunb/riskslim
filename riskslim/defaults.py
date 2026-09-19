import numpy as np

OUTCOME_NAME = 'y'
INTERCEPT_NAME = '(Intercept)'

DEFAULT_LCPA_SETTINGS = {
    #  MIP Formulation
    'drop_variables': True,        #drop variables
    'tight_formulation': True,     #use a slightly tighter MIP formulation
    'include_auxillary_variable_for_objval': True,
    'include_auxillary_variable_for_L0_norm': True,
    #
    # LCPA Settings
    'max_runtime': 300.0,  # max runtime for LCPA
    'max_tolerance': 0.000001,  # tolerance to stop LCPA
    'display_cplex_progress': True,  # setting to True shows CPLEX progress
    'loss_computation': 'normal',  # type of loss computation to use ('normal','fast','lookup')
    'chained_updates_flag': True,  # use chained updates
    'initialization_flag': False,  # use initialization procedure
    'initial_bound_updates': True, # update bounds before solving
    'add_cuts_at_heuristic_solutions': True, #add cuts at integer feasible solutions found using polishing/rounding
    #
    #  LCPA Rounding Heuristic
    'round_flag': True,  # round continuous solutions with SeqRd
    'polish_rounded_solutions': True,  # polish solutions rounded with SeqRd using DCD
    'rounding_tolerance': float('inf'),  # only solutions with objective value < (1 + tol) are rounded
    'rounding_start_cuts': 0,  # cuts needed to start using rounding heuristic
    'rounding_start_gap': float('inf'),  # optimality gap needed to start using rounding heuristic
    'rounding_stop_cuts': 20000,  # cuts needed to stop using rounding heuristic
    'rounding_stop_gap': 0.2,  # optimality gap needed to stop using rounding heuristic
    #
    # LCPA Polishing Heuristic
    'polish_flag': True,  # polish integer feasible solutions with DCD
    'polishing_tolerance': 0.1, # only solutions with objective value (1 + polishing_ub_to_objval_relgap) are polished. setting to
    'polishing_max_runtime': 10.0,  # max time to run polishing each time
    'polishing_max_solutions': 5.0,  # max # of solutions to polish each time
    'polishing_start_cuts': 0,  # cuts needed to start using polishing heuristic
    'polishing_start_gap': float('inf'),  # min optimality gap needed to start using polishing heuristic
    'polishing_stop_cuts': float('inf'),  # cuts needed to stop using polishing heuristic
    'polishing_stop_gap': 5.0,  # max optimality gap required to stop using polishing heuristic
    #
    #  Internal Parameters
    'purge_loss_cuts': False,
    'purge_bound_cuts': False,
    }

DEFAULT_CPLEX_SETTINGS = {
    'randomseed': 0,  # random seed
    'mipemphasis': 0,  # cplex MIP strategy
    'mipgap': np.finfo('float').eps,  #
    'absmipgap': np.finfo('float').eps,  #
    'integrality_tolerance': 1e-6,  #
    'optimality_tolerance': 1e-9,  # distinguish sparsity penalty from zero reduced cost
    'repairtries': 20,  # number of tries to repair user provided solutions
    'poolsize': 100,  # number of feasible solutions to keep in solution pool
    'poolrelgap': float('nan'),  # discard if solutions
    'poolreplace': 2,  # solution pool
    'n_cores': 1,  # number of cores to use in B & B (must be 1)
    'nodefilesize': (120 * 1024) / 1,  # node file size
    }

DEFAULT_CPA_SETTINGS = {
    #
    'type': 'cvx',
    'display_progress': True,  # print progress of initialization procedure
    'display_cplex_progress': False,  # print of CPLEX during intialization procedure
    'save_progress': False,  # print progress of initialization procedure
    'update_bounds': True,
    #
    'max_runtime': 300.0,  # max time to run CPA in initialization procedure
    'max_runtime_per_iteration': 15.0,  # max time per iteration of CPA
    #
    'max_coefficient_gap': 0.49, # stopping tolerance for CPA (based on gap between consecutive solutions)
    'min_iterations_before_coefficient_gap_check': 250,
    #
    'max_iterations': 10000,  # max # of cuts needed to stop CPA
    'max_tolerance': 0.0001,  # stopping tolerance for CPA (based on optimality gap)
    }

DEFAULT_INITIALIZATION_SETTINGS = {
    'type': 'cvx',
    'use_rounding': True,  # use SeqRd in initialization procedure
    'rounding_max_runtime': 30.0,  # max runtime for Rs in initialization procedure
    'rounding_max_solutions': 5,  # max solutions to round using Rd
    #
    'use_sequential_rounding': True,  # use SeqRd in initialization procedure
    'sequential_rounding_max_runtime': 30.0,  # max runtime for SeqRd in initialization procedure
    'sequential_rounding_max_solutions': 5,  # max solutions to round using SeqRd
    #
    'polishing_after': True,  # polish after rounding
    'polishing_max_runtime': 30.0,  # max runtime for polishing
    'polishing_max_solutions': 5  # max solutions to polish
    }

# Initialization Settings includes CPA Settings
DEFAULT_INITIALIZATION_SETTINGS.update(DEFAULT_CPA_SETTINGS)

# LCPA Settings includes Initialization and CPLEX settings
DEFAULT_LCPA_SETTINGS.update({'init_%s' % k: v for k,v in DEFAULT_INITIALIZATION_SETTINGS.items()})
DEFAULT_LCPA_SETTINGS.update({'cplex_%s' % k: v for k,v in DEFAULT_CPLEX_SETTINGS.items()})

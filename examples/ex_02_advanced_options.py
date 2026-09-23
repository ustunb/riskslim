"""
02. Advanced Options
====================

Advanced settings for riskslim.
"""

###################################################################################################

from pathlib import Path

import numpy as np

from riskslim import RiskSLIMClassifier
from riskslim.data import BinaryClassificationDataset

###################################################################################################
# Load Data
# ---------
#
# The example data in this tutorial is used to predict if a breast cancer tumor is beign or
# malignant.
#

# Load Data (outcome in the first column)
data_name = "breastcancer"
data = BinaryClassificationDataset.read_csv(
    Path(__file__).resolve().parents[1] / "data" / f"{data_name}_data.csv"
)

# Unpack data
X = data.X
y = data.y
variable_names = list(data.names.X)
outcome_name = data.names.y

###################################################################################################
# Settings
# --------
#
# RiskSLIM settings and brief descriptions are provided below.
#

settings = {

    ## LCPA Settings
    # max runtime for LCPA
    "max_runtime": 300.0,
    # tolerance to stop LCPA (set to 0 to return provably optimal solution)
    "max_tolerance": np.finfo("float").eps,
    # set to True to print CPLEX progress
    "display_cplex_progress": False,

    ## Other LCPA Heuristics
    # use chained updates
    "chained_updates_flag": True,
    # add cuts at integer feasible solutions found using polishing/rounding
    "add_cuts_at_heuristic_solutions": True,

    # LCPA Rounding Heuristic
    # round continuous solutions with SeqRd
    "round_flag": True,
    # polish solutions rounded with SeqRd using DCD
    "polish_rounded_solutions": True,
    # only solutions with objective value < (1 + tol) are rounded
    "rounding_tolerance": float("inf"),
    # cuts needed to start using rounding heuristic
    "rounding_start_cuts": 0,
    # optimality gap needed to start using rounding heuristic
    "rounding_start_gap": float("inf"),
    # cuts needed to stop using rounding heuristic
    "rounding_stop_cuts": 20000,
    # optimality gap needed to stop using rounding heuristic
    "rounding_stop_gap": 0.2,

    # LCPA Polishing Heuristic
    # polish integer feasible solutions with DCD
    "polish_flag": True,
    # only solutions with objective value (1 + tol) are polished.
    "polishing_tolerance": 0.1,
    # max time to run polishing each time
    "polishing_max_runtime": 10.0,
    # max # of solutions to polish each time
    "polishing_max_solutions": 5.0,
    # cuts needed to start using polishing heuristic
    "polishing_start_cuts": 0,
    # min optimality gap needed to start using polishing heuristic
    "polishing_start_gap": float("inf"),
    # cuts needed to stop using polishing heuristic
    "polishing_stop_cuts": float("inf"),
    # max optimality gap required to stop using polishing heuristic
    "polishing_stop_gap": 0.0,

    ## Initialization Procedure
    # use initialization procedure
    "initialization_flag": True,
    # show progress of initialization procedure
    "init_display_progress": False,
    # show progress of CPLEX during intialization procedure
    "init_display_cplex_progress": False,

    # max time to run CPA in initialization procedure
    "init_max_runtime": 300.0,
    # max # of cuts needed to stop CPA
    "init_max_iterations": 10000,
    # tolerance of solution to stop CPA
    "init_max_tolerance": 0.0001,
    # max time per iteration of CPA
    "init_max_runtime_per_iteration": 300.0,

    # use Rd in initialization procedure
    "init_use_rounding": True,
    # max runtime for Rd in initialization procedure
    "init_rounding_max_runtime": 30.0,
    # max solutions to round using Rd
    "init_rounding_max_solutions": 5,

    # use SeqRd in initialization procedure
    "init_use_sequential_rounding": True,
    # max runtime for SeqRd in initialization procedure
    "init_sequential_rounding_max_runtime": 10.0,
    # max solutions to round using SeqRd
    "init_sequential_rounding_max_solutions": 5,

    # polish after rounding
    "init_polishing_after": True,
    # max runtime for polishing
    "init_polishing_max_runtime": 30.0,
    # max solutions to polish
    "init_polishing_max_solutions": 5,

    ## CPLEX Solver Parameters
    # random seed
    "cplex_randomseed": 0,
    # cplex MIP strategy
    "cplex_mipemphasis": 0,
}

###################################################################################################
# Problem Parameters
# ------------------
#
# These parameters determine the sparisty and constraints of the model. The bounds on magnitiude of
# coefficients and number of non-zero coefficents are set below. Pass these parameters to a
# ``RiskSLIM`` object during initialization.
#

# Value of largest/smallest coefficient
max_coefficient = 5

# Maximum model size (number of non-zero coefficients; default set as float(inf))
max_size = 5

# L0-penalty parameter
#   c0_value > 0
#   larger values -> sparser models
#   small values (1e-6) give model with max_L0 terms
c0_value = 1e-6

###################################################################################################
# Initialize & Cross-Validate
# ---------------------------
#
# The ``RiskSLIMClassifier`` is first initalized using the bound on coefficients (``max_coef``),
# the bound on the number of non-zero coefficients (``max_size``), and the settings defined in the
# previous cell. Fitting the model object is performed using ``.fit(X, y)``, where ``X`` is a
# 2d-array of features and ``y`` is an array of class labels. The ``.fit_cv`` method then
# cross-validates the model using scikit-learn's ``cv`` options.
#

rs = RiskSLIMClassifier(max_coef = max_coefficient, max_size = max_size,
                        variable_names = variable_names, c0_value = c0_value,
                        verbose = False, outcome_name = outcome_name,
                        **settings)

rs.fit(X, y)

rs.fit_cv(
    X,
    y,
    cv=5,
    scoring="roc_auc"
)

print(rs)

###################################################################################################
# Results
# -------
#
# The solver's solution and a dictionary of solution info are stored in the ``.optimizer.solution``
# and ``.solution_info_`` attributes, respectively. The fitted coefficients are in ``.coef_`` and
# ``.intercept_``, and ``print(rs)`` shows the score table. Reports may be generated using the
# ``.report()`` method.
#

rs.report(model_type="risk_score", data=data)

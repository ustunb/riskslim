"""
01. Quickstart
==============

A minimial example for learning risk scores.
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
data_file = Path(__file__).resolve().parents[1] / "data" / f"{data_name}_data.csv"
data = BinaryClassificationDataset.read_csv(data_file)

# Unpack data
X = data.X
y = data.y
variable_names = list(data.names.X)
outcome_name = data.names.y

###################################################################################################
# Settings
# --------
#
# RiskSLIM settings and brief descriptions are provided below. See the next example for
# a full set of options.
#

# Major settings
settings = {
    # LCPA Settings
    # -------------
    # max runtime for LCPA
    "max_runtime": 30.0,
    # tolerance to stop LCPA (set to 0 to return provably optimal solution)
    "max_tolerance": np.finfo("float").eps,

    # LCPA Improvements
    # -----------------
    # round continuous solutions with SeqRd
    "round_flag": True,
    # polish integer feasible solutions with DCD
    "polish_flag": True,
    # use chained updates
    "chained_updates_flag": True,
    # add cuts at integer feasible solutions found using polishing/rounding
    "add_cuts_at_heuristic_solutions": True,

    # Initialization
    # --------------
    # use initialization procedure
    "initialization_flag": True,
    # max time to run CPA in initialization procedure
    "init_max_runtime": 120.0,
    "init_max_coefficient_gap": 0.49,

    # CPLEX Solver Parameters
    # -----------------------
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
# ``RiskSLIMClassifier`` object during initialization.
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
# Initialize & Fit
# ----------------
#
# The ``RiskSLIMClassifier`` is first initalized using the bound on the magnitude of coefficients
# (``max_coef``), the maximum number of non-zero coefficients (``max_size``), and the settings
# defined in the previous cells. Fitting the model object is performed using ``.fit(X, y)``,
# where ``X`` is a 2d-array of features and ``y`` is an array of class labels.
#

rs = RiskSLIMClassifier(max_coef = max_coefficient, max_size = max_size,
                        variable_names = variable_names, c0_value = c0_value,
                        verbose = False, outcome_name = outcome_name,
                        **settings)

rs.fit(X, y)

###################################################################################################
# Results
# -------
#
# The fitted optimizer (with the CPLEX model) and a dictionary of solver statistics are stored in
# the ``.optimizer_`` and ``.solution_info_`` attributes, respectively. The optimized coefficients
# are in ``.coef_`` and ``.intercept_``, and printing the classifier shows the risk score.
# Reports may be generated using the ``.report()`` method.
#

print(rs)


###################################################################################################
# Interactive Reports
# -------------------
#
# ``.report()`` returns an interactive report of the model, with a summary table and ROC and
# calibration plots. Save it as an html file with ``.save``; notebooks display it inline.
#

# Create interactive html report
report = rs.report(model_type = "risk_score", data = data)
report.save("example_report.html")

# Display report
report  # noqa: B018 (sphinx-gallery shows the last expression)


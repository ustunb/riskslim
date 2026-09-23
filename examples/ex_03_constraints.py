"""
03. Constraints
===============

Adding constraints to the MIP.
"""

###################################################################################################

from pathlib import Path

from plotly.io import show
from sklearn import clone

from riskslim import RiskSLIMClassifier, load_data_from_csv

###################################################################################################
# Load Data
# ---------
#
# The example data in this tutorial is used to predict if a breast cancer tumor is beign or
# malignant.
#

# Load Data
data_name = "breastcancer"
data = load_data_from_csv(dataset_csv_file = Path(f'data/{data_name}_data.csv'))

# Unpack data
X = data["X"]
y = data["y"]
variable_names = data["variable_names"]
outcome_name = data["outcome_name"]

# Procedures and improvement settings
settings = {}
settings['drop_variables'] = False
settings['initialization_flag'] = True
settings['round_flag'] = False
settings['polish_flag'] = True
settings['chained_updates_flag'] = False


###################################################################################################
# Base Estimator
# --------------
#
# A base estimator is defined below and used to generate solutions with and without constraints.
#

rs_base = RiskSLIMClassifier(max_coef = 5, max_size = 5,
                             variable_names = data["variable_names"],
                             verbose = False, settings = settings)

# Fit
rs = clone(rs_base)
rs.fit(X, y)

rs.scores  # noqa: B018 (sphinx-gallery shows the last expression)


###################################################################################################
# Constraints
# -----------
#
# Next, an estimator with constraints is created using the ``.add_constraint`` method. The added
# constraint ensure that either ClumpThickness or MarginalAdhesion will be zero. This is done by
# setting the var_type parameter to "alpha", which must be either zero or one for each parameter
# in the constraint. Constraints should be placed on either "alpha" or "rho", obeying:
#
# -rho_min_i * alpha_i < coefficient_i < rho_max_i * alpha_i
#

rs_constrained = clone(rs_base)

rs_constrained.add_constraint(
    # Variable names
    ['ClumpThickness', 'MarginalAdhesion'],
    # Variable type ("rho" or "alpha")
    "alpha",
    # Constraint coefficients
    [1., 1.],
    # Right hand side
    1.,
    # Sense or (in)equality
    "L",
    # Name of constraint
    "either_or"
)

rs_constrained.fit(X, y)

rs_constrained.scores  # noqa: B018 (sphinx-gallery shows the last expression)

###################################################################################################
#
# Coefficients (rho) values may instead be used to directly constrain problem. The coefficient of
# the ClumpThickness variable is constrained below to be a minimum of 2.
#

rs_constrained.add_constraint(
    # Variable names
    ['ClumpThickness'],
    # Variable type ("rho" or "alpha")
    "rho",
    # Constraint coefficients
    [1.],
    # Right hand side
    2.,
    # Sense or (in)equality
    "G",
    # Name of constraint
    "min_rho_thickness"
)

rs_constrained.fit(X, y)

rs_constrained.scores  # noqa: B018 (sphinx-gallery shows the last expression)

# sphinx_gallery_start_ignore
show(rs_constrained.create_report(only_table=True, show=False))
# sphinx_gallery_end_ignore

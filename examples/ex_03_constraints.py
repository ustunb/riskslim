"""
03. Constraints
===============

Adding constraints to the MIP.
"""

###################################################################################################

from pathlib import Path

from sklearn import clone

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
X, y = data.X, data.y
variable_names = list(data.names.X)
outcome_name = data.names.y

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
                             variable_names = variable_names, outcome_name = outcome_name,
                             verbose = False, **settings)

# Fit
rs = clone(rs_base)
rs.fit(X, y)

print(rs)


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

print(rs_constrained)

###################################################################################################
#
# Coefficients (rho) values may instead be used to directly constrain problem. The coefficient of
# the ClumpThickness variable is constrained below to be a minimum of 2.
#

rs_constrained.add_constraint(
    # Variable names
    ['ClumpThickness'],
    # Variable type ("rho" or "alpha"); "rho" is the MIP's name for the coefficient variables
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

print(rs_constrained)

# sphinx_gallery_start_ignore
rs_constrained.report(model_type="risk_score", data=data)  # noqa: B018 (sphinx-gallery shows the last expression)
# sphinx_gallery_end_ignore

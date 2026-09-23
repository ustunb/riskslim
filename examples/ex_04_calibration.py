"""
04. RiskSLIM Calibration
========================

Calibrate RiskSLIMClassifier probability estimates.
"""

###################################################################################################

from pathlib import Path

import numpy as np
from plotly.io import show
from plotly.subplots import make_subplots
from sklearn import clone

from riskslim import RiskSLIMClassifier, load_data_from_csv

###################################################################################################
# Calibration
# -----------
#
# Calibration curves plot the percents or fractions of the true positive class per probability bin
# on the y-axis and the mean predicted prbability (or risk) per bin on the x-axis. Optimal
# calibration results in predicted risk (or probability of class 1) equal to observed risk.
#
# ``RiskSLIMClassifier``'s probability estimates or predicted risk may be calibrated using sklearn's
# ``CalibratedClassifierCV``. The calibration object fits a linear regressor on cross-validated
# probability estimates of ``RiskSLIMClassifier`` to predict true labels.
#

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

###################################################################################################
# Initialize
# ----------
#

settings = {
    "max_runtime": 30.0,
    "max_tolerance": np.finfo("float").eps,
    "round_flag": True,
    "polish_flag": True,
    "chained_updates_flag": True,
    "add_cuts_at_heuristic_solutions": True,
    "initialization_flag": True,
    "init_max_runtime": 120.0,
    "init_max_coefficient_gap": 0.49,
    "cplex_randomseed": 0,
    "cplex_mipemphasis": 0,
}

rs_base = RiskSLIMClassifier(max_coef = 5, max_size = 10,
                             variable_names = variable_names, verbose = False,
                             outcome_name = outcome_name, **settings)

###################################################################################################
# Fit
# ---
#
# After ``.fit`` of ``.fit_cv`` is called, ``.recalibrate`` may be used to fit a post-hoc
# calibrator.
# After calibrating, the ``.predict`` and ``.predict_proba`` use the output from the calibrator
# (``.calibrated_estimator``) trained on all of the data passed. If ``.fit_cv`` is used, calibrator
# estimators (``.cv_calibrated_estimators_``) are trained on the train sets and the plots will show
# performance on the test set.the plots will contain on the test sets. Plots will show performance
# on each fold in light gray, while the calibrator trained on all of the data will be shown in
# black. Options for calibration methods include "isotonic" and "sigmoid" (Platt).
#

# Calibrated model
rs_cal = clone(rs_base)
rs_cal.fitcv(X, y, k=5)
rs_cal.calibrate(X, y, method="isotonic")

# Un calibrated model
rs = clone(rs_base)
rs.fitcv(X, y, k=5)

###################################################################################################
# Plot Results
# ------------
#

fig_nocal = rs.create_report()
fig_cal = rs_cal.create_report()

# Isoloate calibration subplot of the
#   figure create from create_report
xaxis, yaxis = fig_cal.get_subplot(3, 1)
xaxis.domain = None
yaxis.domain = None

# Create a new figure
fig = make_subplots(1, 2, subplot_titles=("Un-Calibrated", "Calibrated"))

col = 1
for data in [fig_nocal.data, fig_cal.data]:
    for sp in data:
        if not type(sp).__name__ == 'Scattergl':
            # Skip tables
            continue
        if sp.x.max() <= 1:
            # Skip ROC
            continue
        fig.add_trace(sp, col=col, row=1)
    col += 1

# Transfer axis and layout settings
fig.update_xaxes(xaxis)
fig.update_yaxes(yaxis)

fig.update_layout({
    "width": 1000,
    "height": 500,
    "showlegend": False,
    "title": "Calibration"
})

fig.show()

# sphinx_gallery_start_ignore
show(fig)
# sphinx_gallery_end_ignore
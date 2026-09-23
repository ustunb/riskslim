"""
04. RiskSLIM Calibration
========================

Calibrate RiskSLIMClassifier probability estimates.
"""

###################################################################################################

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from riskslim import RiskSLIMClassifier
from riskslim.data import BinaryClassificationDataset

###################################################################################################
# Calibration
# -----------
#
# Calibration curves plot the percents or fractions of the true positive class per probability bin
# on the y-axis and the mean predicted prbability (or risk) per bin on the x-axis. Optimal
# calibration results in predicted risk (or probability of class 1) equal to observed risk.
#
# ``RiskSLIMClassifier``'s probability estimates or predicted risk may be calibrated using sklearn's
# ``CalibratedClassifierCV``. The calibration object fits a sigmoid (Platt) or isotonic regressor on
# cross-validated probability estimates of ``RiskSLIMClassifier`` to predict true labels.
#

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

rs = RiskSLIMClassifier(max_coef = 5, max_size = 10,
                        variable_names = variable_names, verbose = False,
                        outcome_name = outcome_name, **settings)

###################################################################################################
# Fit
# ---
#
# After ``.fit`` or ``.fit_cv`` is called, ``.recalibrate`` may be used to fit a post-hoc
# calibrator.
# After calibrating, ``.predict`` and ``.predict_proba`` use the output from the calibrator
# (``.calibrated_estimator_``) trained on all of the data passed. If ``.fit_cv`` is used, calibrator
# estimators (``.cv_calibrated_estimators_``) are also trained on each fold's train set, so they
# can be evaluated on the fold's test set. Options for calibration methods include "isotonic" and
# "sigmoid" (Platt).
#

rs.fit(X, y)
rs.fit_cv(X, y, cv=5)
print(rs)

# Un-calibrated risk on each test fold, from each fold's model
test_folds = rs.cv_results_["indices"]["test"]
risk_nocal = np.empty(len(y))
for fold_model, test in zip(rs.cv_results_["estimator"], test_folds, strict=True):
    risk_nocal[test] = fold_model.predict_proba(X[test])[:, 1]

# Calibrated model
rs.recalibrate(X, y, method="isotonic")

# Calibrated risk on each test fold, from each fold's calibrator
risk_cal = np.empty(len(y))
for fold_calibrator, test in zip(rs.cv_calibrated_estimators_, test_folds, strict=True):
    risk_cal[test] = fold_calibrator.predict_proba(X[test])[:, 1]

###################################################################################################
# Compare Calibration
# -------------------
#
# Observed vs. predicted risk on the test folds, in five bins of predicted risk, for the
# un-calibrated and calibrated models. The report (``rs.report(data=data)``) shows the calibration
# of the risk score itself, i.e. the un-calibrated model.
#

rows = {}
for name, risk in [("Un-Calibrated", risk_nocal), ("Calibrated", risk_cal)]:
    observed, predicted = calibration_curve(y, risk, n_bins=5)
    rows[name] = pd.DataFrame({"predicted risk": predicted, "observed risk": observed})
    rows[name].loc["mean |error|"] = [np.nan, np.abs(predicted - observed).mean()]

calibration = pd.concat(rows, axis=1).round(3)
print(calibration)

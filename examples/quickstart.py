from pathlib import Path

from riskslim import CoefficientSet, RiskSLIMClassifier
from riskslim.data import BinaryClassificationDataset, NumericBinarizer

# import data (683 x 9, integer features, outcome in the first column) with 5 CV folds
data_file = Path(__file__).resolve().parents[1] / "data" / "breastcancer_data.csv"
data = BinaryClassificationDataset.read_csv(data_file, n_folds=(5,))

# binarize: each item is a feature at 5 or above (features are 1-10), named "<feature>_geq_5"
data.processor.update({name: NumericBinarizer(thresholds=[5]) for name in data.names.X})
X, y = data.X, data.y
items = list(data.names.X)

# fit a risk score with at most 5 items and coefficients in -5,...,5
clf = RiskSLIMClassifier(
    max_size=5,
    max_coef=5,
    variable_names=items,
    outcome_name=data.names.y,
    cplex_randomseed=0,
    verbose=False,
)
clf.fit(X, y, max_runtime=30.0)

# print the score table
print(clf)

# predict and score
print("train accuracy:", clf.score(X, y))

# cross-validate on the dataset's 5 folds, so the report shows a 5-CV sample next to training
clf.fit_cv(X, y, data=data, max_runtime=30.0)

# save an HTML report of the model
clf.report(model_type="risk_score", data=data).save("riskslim_report.html")

# fit a checklist on the same items with coefficients in {0, 1}, i.e. "predict the outcome if at
# least M of these items are checked"
coef_set = CoefficientSet(
    ["(Intercept)"] + items,
    lb=[-5] + [0] * len(items),  # the intercept keeps the default bounds
    ub=[5] + [1] * len(items),
    print_flag=False,
)
checklist = RiskSLIMClassifier(
    max_size=5,
    coef_set=coef_set,
    variable_names=items,
    outcome_name=data.names.y,
    cplex_randomseed=0,
    verbose=False,
)
checklist.fit(X, y, max_runtime=30.0)
print(checklist)
print("checklist train accuracy:", checklist.score(X, y))
checklist.fit_cv(X, y, data=data, max_runtime=30.0)
checklist.report(model_type="checklist", data=data).save("checklist_report.html")

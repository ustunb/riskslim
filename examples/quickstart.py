from pathlib import Path

import pandas as pd

from riskslim import CoefficientSet, RiskSLIMClassifier

# import data (683 x 9, integer features, outcome in the first column)
data_file = Path(__file__).resolve().parents[1] / "data" / "breastcancer_data.csv"
df = pd.read_csv(data_file)
y, X = df.iloc[:, 0], df.iloc[:, 1:]

# binarize: each item is a feature at 5 or above (features are 1-10)
X_items = (X >= 5).astype(int).add_suffix(" >= 5")
items = list(X_items.columns)

# fit a risk score with at most 5 items and coefficients in -5,...,5
clf = RiskSLIMClassifier(
    max_size=5,
    max_coef=5,
    variable_names=items,
    outcome_name=df.columns[0],
    cplex_randomseed=0,
    verbose=False,
)
clf.fit(X_items.values, y.values, max_runtime=30.0)

# print the score table
print(clf)

# predict and score
print("train accuracy:", clf.score(X_items.values, y.values))

# save an HTML report of the model
clf.report(model_type="risk_score").save("riskslim_report.html")

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
    outcome_name=df.columns[0],
    cplex_randomseed=0,
    verbose=False,
)
checklist.fit(X_items.values, y.values, max_runtime=30.0)
print(checklist)
print("checklist train accuracy:", checklist.score(X_items.values, y.values))
checklist.report(model_type="checklist").save("checklist_report.html")

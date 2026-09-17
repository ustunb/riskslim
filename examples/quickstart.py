from pathlib import Path

import pandas as pd

from riskslim import RiskSLIMClassifier

# import data (683 x 9, integer features, outcome in the first column)
data_file = Path(__file__).resolve().parents[1] / "data" / "breastcancer_data.csv"
df = pd.read_csv(data_file)
y, X = df.iloc[:, 0], df.iloc[:, 1:]

# fit a risk score with at most 5 variables and coefficients in -5,...,5
clf = RiskSLIMClassifier(
    max_size=5,
    max_coef=5,
    variable_names=list(X.columns),
    outcome_name=df.columns[0],
    cplex_randomseed=0,
    verbose=False,
)
clf.fit(X.values, y.values, max_runtime=30.0)

# print the score table
print(clf)

# predict and score
print("train accuracy:", clf.score(X.values, y.values))

# save an HTML report of the model
clf.reporter.create_report("riskslim_report.html")

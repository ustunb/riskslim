"""
00. Quickstart
"""
import pandas as pd

from riskslim import RiskSLIMClassifier
from riskslim.utils import open_file

# Load Data
df = pd.read_csv("https://raw.githubusercontent.com/ustunb/riskslim/master/examples/data/mushroom_data.csv")
y, X = df.iloc[:, 0].values, df.iloc[:, 1:].values


# fit model
clf = RiskSLIMClassifier(max_coef = 5, max_size = 5, variable_names = df.columns[1:].tolist(),
                         outcome_name = df.columns[0])
clf.fit(X, y)

# Create Report
report_file = clf.report().save('report.html')
open_file(report_file)
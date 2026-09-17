"""RiskSLIM quickstart — deterministic, no network, CE-sized."""

from pathlib import Path

import pandas as pd

from riskslim import RiskSLIMClassifier

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "breastcancer_data.csv"


def main():
    frame = pd.read_csv(DATA_FILE)  # 683 x 9 integer features, outcome in column 0
    X = frame.iloc[:, 1:].to_numpy()
    y = frame.iloc[:, 0].to_numpy()
    clf = RiskSLIMClassifier(
        max_coef=5,
        max_size=5,
        variable_names=list(frame.columns[1:]),
        outcome_name=frame.columns[0],
        verbose=False,
        cplex_randomseed=0,
    )
    clf.fit(X, y, max_runtime=30.0)
    print(clf)  # score table
    print("train accuracy:", clf.score(X, y))
    clf.reporter.create_report("riskslim_report.html")


if __name__ == "__main__":
    main()

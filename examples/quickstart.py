"""RiskSLIM quickstart — deterministic, no network, CE-sized."""

import numpy as np
from sklearn.datasets import load_breast_cancer  # sklearn is already a runtime dependency

from riskslim import RiskSLIMClassifier


def main():
    X, y = load_breast_cancer(return_X_y=True)  # 569 x 30, bundled with sklearn
    X = (X > np.median(X, axis=0)).astype(int)  # binarize at medians -> 30 binary features
    names = [f"{n} > median" for n in load_breast_cancer().feature_names]
    clf = RiskSLIMClassifier(
        max_coef=5,
        max_size=5,
        variable_names=names,
        outcome_name="malignant",
        verbose=False,
        cplex_randomseed=0,
        initialization_flag=True,
        init_max_runtime=5.0,
        init_use_rounding=False,
        init_use_sequential_rounding=False,
        init_polishing_after=False,
    )
    clf.fit(X, y, max_runtime=30.0)  # y in {0,1} must work (defect m)
    print(clf)  # score table
    print("train accuracy:", clf.score(X, y))
    clf.reporter.create_report("riskslim_report.html")


if __name__ == "__main__":
    main()

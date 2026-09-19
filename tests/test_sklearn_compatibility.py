"""RiskSLIMClassifier is a scikit-learn estimator."""

import numpy as np
from sklearn.base import clone
from sklearn.utils.estimator_checks import parametrize_with_checks

from riskslim import RiskSLIMClassifier


@parametrize_with_checks([RiskSLIMClassifier(verbose=False, max_runtime=10, cplex_randomseed=0)])
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)


def test_clone_keeps_settings_and_refits_identically():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(80, 4)).astype(float)
    y = np.where(X[:, 0] + X[:, 1] + rng.random(80) > 1.2, "yes", "no")
    original = RiskSLIMClassifier(max_coef=3, verbose=False, max_runtime=10, cplex_randomseed=0)

    cloned = clone(original)

    assert cloned.get_params()["max_runtime"] == 10
    assert cloned.get_params()["cplex_randomseed"] == 0
    np.testing.assert_array_equal(original.fit(X, y).predict(X), cloned.fit(X, y).predict(X))
    assert set(original.predict(X)) <= {"yes", "no"}

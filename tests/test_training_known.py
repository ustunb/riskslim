# Test strategy: training against saved known results
#
# Dimensions:
#   known dataset: breastcancer, mammo — distinct real inputs and saved results
#
# These saved known results are regression references, not independently exact oracles.

import numpy as np
import pytest

from riskslim import RiskSLIMClassifier


def test_training_matches_known_reference(known_dataset_name, training_test_cases):
    reference = training_test_cases["known_reference_results"][known_dataset_name]
    X = reference["X"]
    y = reference["y"]
    expected = reference["expected"]
    clf = RiskSLIMClassifier(
        max_coef=5,
        max_size=5,
        c0_value=1e-6,
        variable_names=list(reference["variable_names"]),
        outcome_name=reference["outcome_name"],
        verbose=False,
        cplex_randomseed=0,
        max_tolerance=1e-6,
        max_runtime=300,
    )
    clf.fit(X, y)

    solution_info = clf.optimizer.solution_info
    assert abs(solution_info["objective_value"] - expected["objective_value"]) <= 1e-4
    assert clf.optimizer.stats.cplex_status in {
        "integer optimal solution",
        "integer optimal, tolerance",
    }
    rho = np.insert(clf.coef_, 0, clf.intercept_)
    assert np.count_nonzero(clf.coef_) <= 5
    assert np.all(np.abs(rho[1:]) <= 5)
    assert np.all(rho == np.rint(rho))
    assert np.mean(clf.predict(X) == np.asarray(expected["predictions"])) >= 0.99


@pytest.fixture(params=["breastcancer", "mammo"])
def known_dataset_name(request):
    return request.param

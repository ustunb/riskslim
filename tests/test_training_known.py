# Test strategy: training against saved known results
#
# Dimensions:
#   known dataset: breastcancer, mammo — distinct real inputs and saved results
#   planted-data seed: 0..4 — retained temporarily until exact synthetic-oracle tests replace it
#
# These saved known results are regression references, not independently exact oracles.

import numpy as np
import pytest

from riskslim import RiskSLIMClassifier
from riskslim.data import ClassificationDataset
from riskslim.loss_functions.log_loss import log_loss_value
from utils import generate_random_normal


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


@pytest.mark.parametrize("seed", range(5))
def test_synthetic_planted(seed):
    data, rho_true = generate_random_normal(200, 12, 4, seed)
    y = np.ravel(data["y"])
    dataset = ClassificationDataset(
        data["X"], y, variable_names=data["variable_names"], outcome_name=data["outcome_name"]
    )
    rho_planted = np.insert(rho_true, 0, 0)
    planted_objective = log_loss_value(dataset.Z, rho_planted) + 1e-6 * 4

    clf = RiskSLIMClassifier(
        max_coef=5,
        max_size=4,
        c0_value=1e-6,
        variable_names=data["variable_names"],
        outcome_name=data["outcome_name"],
        verbose=False,
        max_runtime=60,
    )
    clf.fit(data["X"], y)
    assert clf.optimizer.solution_info["objective_value"] <= planted_objective + 1e-6

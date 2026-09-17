import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from riskslim import RiskSLIMClassifier
from riskslim.data import ClassificationDataset
from riskslim.loss_functions.log_loss import log_loss_value
from utils import generate_random_normal


GOLDEN_DIR = Path(__file__).parent / "golden"


def test_golden_dataset(dataset_name):
    fixture = json.loads((GOLDEN_DIR / f"{dataset_name}_k5.json").read_text())
    frame = pd.read_csv(Path(__file__).parents[1] / "data" / f"{dataset_name}_data.csv")
    X = frame.iloc[:, 1:].to_numpy()
    y = frame.iloc[:, 0].to_numpy()
    clf = RiskSLIMClassifier(
        max_coef=5,
        max_size=5,
        c0_value=1e-6,
        variable_names=list(frame.columns[1:]),
        outcome_name=frame.columns[0],
        verbose=False,
        cplex_randomseed=0,
        max_tolerance=1e-6,
        max_runtime=300,
    )
    clf.fit(X, y)

    solution_info = clf.optimizer.solution_info
    assert abs(solution_info["objective_value"] - fixture["objective_value"]) <= 1e-4
    assert clf.optimizer.stats.cplex_status in {
        "integer optimal solution",
        "integer optimal, tolerance",
    }
    rho = np.insert(clf.coef_, 0, clf.intercept_)
    assert np.count_nonzero(clf.coef_) <= 5
    assert np.all(np.abs(rho[1:]) <= 5)
    assert np.all(rho == np.rint(rho))
    assert np.mean(clf.predict(X) == np.asarray(fixture["predictions"])) >= 0.99


@pytest.fixture(params=["breastcancer", "mammo"])
def dataset_name(request):
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

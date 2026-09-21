"""Tests for riskslim.report.data (build_report_data).

Test strategy
-------------
Fixture: the shared report sample in conftest (8 training rows, 4 test rows, three binary
features, fixed coefficient vectors); expected values are literals worked out by hand, not
recomputed. No solver is called.

Dimensions:
  model type:   risk_score (points 2, 1, -1), checklist (+1 items only), checklist (with a -1 item)
                -- inferred from the coefficients; model_type="risk_score" overrides a checklist
  label coding: {0, 1}, {-1, 1}    -- both must map to the same positive class
  samples:      train only, train + test (the test sample has no row with score 1)

Rejection paths owned here (one invalid mutation of a valid call each): rho and variable_names of
different lengths, variable_names without "(Intercept)" first, non-finite rho, unknown model_type,
samples without 'train', X column count != number of variable names, y row count != X row count,
labels outside {0, 1} / {-1, 1}, and a sample with a single class.
n/a: non-binary features for the checklist -- the score range rule is shared with the risk score.
"""

import json

import numpy as np
import pytest
from conftest import (
    CHECKLIST_RHO,
    CONSTRAINTS,
    NAMES,
    RISK_SCORE_RHO,
    TRAINING,
    X_TEST,
    X_TRAIN,
    Y_TEST,
    Y_TRAIN,
)

from riskslim.report import build_report_data

VALID_CALL = {"rho": RISK_SCORE_RHO, "variable_names": NAMES, "outcome_name": "y",
              "samples": {"train": (X_TRAIN, Y_TRAIN), "test": (X_TEST, Y_TEST)}}


def build(rho, samples=None, **kwargs):
    samples = samples or {"train": (X_TRAIN, Y_TRAIN), "test": (X_TEST, Y_TEST)}
    return build_report_data(rho, NAMES, "y", samples, **kwargs)


def test_risk_score_model_has_score_range_and_score_to_risk():
    model = build(RISK_SCORE_RHO)["model"]

    assert model["type"] == "risk_score"
    assert [(item["name"], item["points"]) for item in model["items"]] == [
        ("a", 2), ("b", 1), ("c", -1)]
    assert model["score_range"] == [-1, 3]
    assert model["score_to_risk"]["scores"] == [-1, 0, 1, 2, 3]
    assert model["score_to_risk"]["risk"] == pytest.approx(
        [0.0474258732, 0.1192029220, 0.2689414214, 0.5, 0.7310585786])
    assert model["checklist_m"] is None


def test_non_binary_feature_score_range_spans_points_times_value_range():
    X = np.column_stack([np.arange(1, 9), X_TRAIN[:, 1:]])  # a takes values 1..8

    model = build(RISK_SCORE_RHO, samples={"train": (X, Y_TRAIN)})["model"]

    assert model["items"][0] == {"name": "a", "points": 2, "binary": False,
                                 "value_range": [1, 8]}
    assert model["score_range"] == [1, 17]
    assert model["score_to_risk"]["scores"] == list(range(1, 18))


@pytest.mark.parametrize("rho, expected_m, expected_rule", [
    pytest.param(CHECKLIST_RHO, 2, "Predict y if at least 2 of 2 items are checked",
                 id="positive-items"),
    pytest.param([0, 1, 1, -1], 1,
                 "Predict y if the number of checked (+) items minus the number of checked "
                 "(−) items is at least 1", id="with-negative-item"),
    pytest.param([-3, 1, 1, 0], 4,
                 "Never predict y (no set of checked items reaches the threshold)",
                 id="threshold-out-of-reach"),
])
def test_checklist_m_is_smallest_net_count_with_positive_prediction(rho, expected_m,
                                                                    expected_rule):
    model = build(rho)["model"]

    assert model["type"] == "checklist"
    assert model["checklist_m"] == expected_m
    assert model["rule"] == expected_rule


def test_risk_score_type_can_be_forced_for_unit_coefficients():
    assert build(CHECKLIST_RHO, model_type="risk_score")["model"]["type"] == "risk_score"


@pytest.mark.parametrize("negative_label", [0, -1], ids=["labels-01", "labels-pm1"])
def test_calibration_bins_count_rows_per_score(negative_label):
    y_train = np.where(Y_TRAIN == 1, 1, negative_label)
    y_test = np.where(Y_TEST == 1, 1, negative_label)

    data = build(RISK_SCORE_RHO, samples={"train": (X_TRAIN, y_train), "test": (X_TEST, y_test)})
    train, test = data["calibration"]["train"], data["calibration"]["test"]

    assert train["scores"] == [-1, 0, 1, 2, 3]
    assert train["n"] == [1, 2, 2, 2, 1]
    assert train["observed"] == [0.0, 0.5, 0.0, 1.0, 1.0]
    assert train["error"] == pytest.approx(0.3269805367)
    # score 1 has no test rows: absent, not NaN
    assert test["scores"] == [-1, 0, 2, 3]
    assert test["n"] == [1, 1, 1, 1]
    assert test["observed"] == [0.0, 0.0, 1.0, 1.0]
    assert test["error"] == pytest.approx(0.2338925541)


def test_roc_has_a_point_per_score_threshold_from_origin_to_corner():
    roc = build(RISK_SCORE_RHO)["roc"]

    assert roc["train"]["thresholds"] == [None, 3, 2, 1, 0, -1]
    assert roc["train"]["fpr"] == [0.0, 0.0, 0.0, 0.5, 0.75, 1.0]
    assert roc["train"]["tpr"] == [0.0, 0.25, 0.75, 0.75, 1.0, 1.0]
    assert roc["train"]["auc"] == 0.84375
    assert roc["test"]["auc"] == 1.0


def test_summary_has_four_blocks_of_formatted_values():
    data = build(RISK_SCORE_RHO, training=TRAINING, constraints=CONSTRAINTS)

    assert data["samples"] == ["train", "test"]
    assert {block["key"]: block["rows"] for block in data["summary"]} == {
        "dataset": [["n", "8", "4"], ["outcome rate", "50.0%", "50.0%"]],
        "constraints": [["model size", "3 (max 3)"], ["point range", "-5 to 5"]],
        "training": [["objective value", "0.5000"], ["optimality gap", "n/a"],
                     ["run time", "1.2 s"]],
        "performance": [["AUC", "0.844", "1.000"], ["calibration error", "32.7%", "23.4%"],
                        ["log loss", "0.579", "0.295"]],
    }


@pytest.mark.parametrize("rho", [RISK_SCORE_RHO, CHECKLIST_RHO], ids=["risk-score", "checklist"])
def test_data_is_strict_json(rho):
    data = build(rho, training=TRAINING, constraints=CONSTRAINTS)

    assert json.loads(json.dumps(data, allow_nan=False)) == data
    assert data["schema_version"] == 1
    assert [c["component"] for row in data["layout"] for c in row["components"]] == [
        "ModelCard", "SummaryTable", "RocPlot", "CalibrationPlot"]


@pytest.mark.parametrize("invalid, match", [
    pytest.param({"rho": RISK_SCORE_RHO[:3]}, "same length", id="rho-shorter-than-names"),
    pytest.param({"variable_names": ["bias", "a", "b", "c"]}, r"'\(Intercept\)' first",
                 id="no-intercept-name"),
    pytest.param({"rho": [float("nan"), 2, 1, -1]}, "rho must be finite", id="non-finite-rho"),
    pytest.param({"model_type": "decision_tree"}, "model_type must be one of",
                 id="unknown-model-type"),
    pytest.param({"samples": {"test": (X_TEST, Y_TEST)}}, "'train' entry", id="missing-train"),
    pytest.param({"samples": {"train": (X_TRAIN[:, :2], Y_TRAIN)}},
                 "variable_names lists 3 features", id="column-count"),
    pytest.param({"samples": {"train": (X_TRAIN, Y_TRAIN[:-1])}},
                 "X has 8 rows but y has 7", id="row-count"),
    pytest.param({"samples": {"train": (X_TRAIN, np.where(Y_TRAIN == 1, 1, 2))}},
                 r"y must be in \{0, 1\} or \{-1, 1\}", id="unsupported-labels"),
    pytest.param({"samples": {"train": (X_TRAIN, Y_TRAIN), "test": (X_TEST, np.zeros(4))}},
                 "'test' has a single class", id="single-class"),
])
def test_invalid_inputs_are_rejected(invalid, match):
    with pytest.raises(Exception, match=match):
        build_report_data(**{**VALID_CALL, **invalid})

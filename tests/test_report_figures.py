"""Tests for riskslim.report.figures (Plotly figure dicts for ROC and calibration).

Test strategy
-------------
Input: the two-sample (train + test) risk-score data dict from test_report_data's fixture.

Dimensions:
  figure:  roc, calibration  -- each gets one trace per sample in sample order and a top-left
                                metrics box (AUC / calibration error)
Checks that need no browser: `plotly.graph_objects.Figure(fig)` rejects misspelled or invalid
properties. Rendering itself is covered by the opt-in browser test in test_report_render.py.
"""

import plotly.graph_objects as go
import pytest

from riskslim.report import build_report_data
from riskslim.report.figures import build_figures
from test_report_data import NAMES, RISK_SCORE_RHO, X_TEST, X_TRAIN, Y_TEST, Y_TRAIN


@pytest.fixture(scope="module")
def figures():
    data = build_report_data(RISK_SCORE_RHO, NAMES, "y",
                             {"train": (X_TRAIN, Y_TRAIN), "test": (X_TEST, Y_TEST)})
    return build_figures(data)


@pytest.mark.parametrize("key, metrics_text", [
    ("roc", ["AUC", "train</span> 0.844", "test</span> 1.000"]),
    ("calibration", ["CAL", "train</span> 32.7%", "test</span> 23.4%"]),
])
def test_figure_has_a_trace_per_sample_and_top_left_metrics_box(figures, key, metrics_text):
    figure = figures[key]

    go.Figure(figure)  # raises on invalid Plotly properties
    assert [trace["name"] for trace in figure["data"]] == ["train", "test"]
    (box,) = figure["layout"]["annotations"]
    assert (box["xref"], box["yref"], box["xanchor"], box["yanchor"]) == (
        "paper", "paper", "left", "top")
    assert box["x"] <= 0.05 and box["y"] >= 0.95
    assert all(text in box["text"] for text in metrics_text), box["text"]


def test_calibration_bubbles_are_labelled_with_scores_and_sized_by_n(figures):
    train, test = figures["calibration"]["data"]

    assert train["text"] == ["-1", "0", "1", "2", "3"]
    assert test["text"] == ["-1", "0", "2", "3"]
    # train n = [1, 2, 2, 2, 1]; test n = [1, 1, 1, 1]
    small, large = train["marker"]["size"][0], train["marker"]["size"][1]
    assert small < large
    assert train["marker"]["size"] == [small, large, large, large, small]
    assert test["marker"]["size"] == [small] * 4


def test_roc_points_carry_score_thresholds(figures):
    train = figures["roc"]["data"][0]

    assert train["customdata"] == ["none", "score ≥ 3", "score ≥ 2", "score ≥ 1",
                                   "score ≥ 0", "score ≥ -1"]

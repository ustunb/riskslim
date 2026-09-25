"""Tests for riskslim.report.model_report (ModelReport: data block, Plotly figures, HTML page).

Test strategy
-------------
Fixture: the shared report sample in tests/utils.py (8 training rows, 4 test rows, three binary
features, fixed coefficient vectors). One small classifier (max size 3, points -5 to 5) is fit
once on the training rows; each report is built from a copy of it carrying a fixed coefficient
vector and fixed solver statistics, with the test rows passed as X_test / y_test, so expected
values are literals worked out by hand, not recomputed.

One end-to-end test skips the fixture: it reads data/breastcancer_data.csv into a
BinaryClassificationDataset, fits a small model and its 5 fold models (fit, then fit_cv), and
checks only that clf.report(data=...) builds a page holding every component (model card,
summary table, ROC, calibration) and both samples (Training, 5-CV). It asserts presence, not values or appearance.

Dimensions:
  model type:   risk_score (points 2, 1, -1), checklist (+1 items only), checklist (with a -1 item)
                -- inferred from the coefficients; model_type="risk_score" overrides a checklist,
                and unit coefficients on a non-binary item infer a risk score
  label coding: {0, 1}, {-1, 1}    -- both must map to the same positive class (one fit each)
  samples:      Training only, Training + Test (the test sample has no row with score 1)
  tails:        no score outside 1%..99% (one cell and one point per score), a high tail of
                many scores, a low tail holding every score -- a tail of one score is not a
                separate value: it is printed and plotted like any other, which the first case
                already covers. The strip and the calibration points collapse from one rule, so
                the strip carries the three cases and the points one pooling case. A discrete
                strip over max_scores_printed narrows its printed risk range (one case).
  score type:   discrete (binary items, integer points: every other case), continuous (a
                non-binary item, fractional points: one case -- risk bins for the strip, the
                calibration points and the ECE, empty bins dropped per sample, plain circles)
  figure:       roc, calibration   -- one trace per sample, plotted straight from the data block's
                                      roc / calibration sections, each named for its legend entry
                                      (sample, n and outcome rate, AUC / ECE)
  settings:     the defaults (every other test), and one report that reorders and drops cards,
                reorders the samples (each keeps its colour) and lowers high_risk_threshold
  label text:   contains "</script>" (also in mixed case) and "<!--"  -- must not end the JSON
                                      data block early, nor appear unescaped anywhere on the page
                                      (a plain label is the same path with nothing to escape, so
                                      it is not a separate case)

Rejection paths owned here (one invalid mutation of a valid call each): non-finite weights, unknown
model_type, model_type="checklist" with a non-binary item, X_test column count != the fitted
feature count, y_test row count != X_test row count, y_test labels outside the classes seen in
fit, a sample with a single class, and components / samples that are empty, repeat an entry or
name an unknown one, risk thresholds out of order or outside 0..1, and a max_scores_printed
below 2.

Checks that need no browser: `plotly.graph_objects.Figure(fig)` rejects misspelled or invalid
properties, and one test asserts the Plotly template's and the sample colours are the ones
styles.css writes as `--rs-*` tokens (the page and the plots hold the palette in their own idiom, so the test is what
keeps them from drifting). The browser test is opt-in (`pytest -m browser`): one page per saved
report (each model type, plus `wide_strip` -- a continuous model with a non-binary item) is
built and saved once, then opened in headless Chromium at 1280 px and 375 px, requiring
no console or page errors, 2 rendered Plotly charts, one model row per item, a score-to-risk
strip that fits the card at that width on one row, and a Model card whose readout and outlined
strip cell follow a checked item and every value typed into a non-binary item (on wide_strip, by
the shipped bin edges, including a bin with no training rows); screenshots and the HTML go to a tmp_path or --report-dir=DIR
(write it with "=": with a space, pytest reads an existing DIR as a test path and misses the
config).
"""

import copy
import html
import json
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pytest
from scipy.special import expit, logit
from utils import (
    CHECKLIST_WEIGHTS,
    NAMES,
    RISK_SCORE_WEIGHTS,
    TRAINING,
    X_TEST,
    X_TRAIN,
    Y_TEST,
    Y_TRAIN,
)

from riskslim import RiskSLIMClassifier
from riskslim.data import BinaryClassificationDataset
from riskslim.report import ModelReport
from riskslim.report.model_report import ASSETS, SAMPLE_COLORS, TEMPLATE_NAME

DATA_KEYS = {"schema_version", "title", "outcome_name", "samples", "model", "summary", "roc",
             "calibration", "settings", "figures", "narrow"}
CDN_URLS = [
    "https://cdn.jsdelivr.net/npm/plotly.js-basic-dist-min@4.1.1/plotly-basic.min.js",
    "https://cdn.jsdelivr.net/npm/@picocss/pico@2.1.1/css/pico.min.css",
]
DATA_BLOCK = re.compile(r'<script type="application/json" id="report-data">(.*?)</script>', re.S)
HOSTILE_NAMES = ["a</script><script>alert(1)</script>", "b<!-- c", "c</ScRiPt><script>alert(2)</script>"]
BREASTCANCER_FILE = Path(__file__).parents[1] / "data" / "breastcancer_data.csv"
COMPONENTS = ['class="rs-model-table"', 'class="rs-summary-table"', 'data-figure="roc"',
              'data-figure="calibration"']
# How far the score-to-risk strip runs past its own box, past the card holding it, and past the
# viewport, in px, and how many rows it takes. The three overflows are 0 at every width: the strip
# neither scrolls sideways nor pushes the page wider than the window, and it is one row.
STRIP_FIT = """() => {
  const strip = document.querySelector(".rs-score-grid");
  const card = strip.closest(".rs-card");
  const past = (edge, limit) => Math.max(0, Math.ceil(edge - limit));
  return {
    strip: past(strip.scrollWidth, strip.clientWidth),
    card: past(strip.getBoundingClientRect().right, card.getBoundingClientRect().right),
    page: past(document.documentElement.scrollWidth, document.documentElement.clientWidth),
    rows: new Set([...strip.children].map((cell) => cell.getBoundingClientRect().top)).size,
  };
}"""


def fit_classifier(X=X_TRAIN, y=Y_TRAIN):
    """A small classifier fit on the training rows: max size 3, points -5 to 5."""
    classifier = RiskSLIMClassifier(max_size=3, max_coef=5, variable_names=NAMES[1:],
                                    outcome_name="y", verbose=False, max_runtime=5,
                                    cplex_randomseed=0)
    return classifier.fit(X, y)


@pytest.fixture(scope="module")
def fitted():
    """The one classifier most tests report on, fit once."""
    return fit_classifier()


def make_report(classifier, weights, test=(X_TEST, Y_TEST), **kwargs):
    """A report on a copy of ``classifier`` with coefficients ``weights`` (intercept first) and the
    shared solver statistics, so expected values do not depend on what the solver found."""
    model = copy.copy(classifier)
    model.intercept_, model.coef_ = float(weights[0]), np.asarray(weights[1:], dtype=float)
    model.solution_info_ = dict(TRAINING)
    X_test, y_test = test
    return ModelReport(model, X_test=X_test, y_test=y_test, **kwargs)


@pytest.fixture(scope="module")
def report(fitted):
    """The default risk-score report."""
    return make_report(fitted, RISK_SCORE_WEIGHTS)


def test_risk_score_model_has_items_and_score_range(fitted):
    model = make_report(fitted, RISK_SCORE_WEIGHTS).data["model"]

    assert model["type"] == "risk_score"
    assert [(item["name"], item["points"]) for item in model["items"]] == [
        ("a", 2), ("b", 1), ("c", -1)]
    assert model["score_range"] == [-1, 3]
    assert model["checklist_m"] is None


@pytest.fixture(scope="module")
def fitted_wide():
    """A classifier whose first feature takes values 1..8: a continuous score type."""
    return fit_classifier(np.column_stack([np.arange(1, 9), X_TRAIN[:, 1:]]))


def test_non_binary_feature_score_range_spans_points_times_value_range(fitted_wide):
    model = make_report(fitted_wide, RISK_SCORE_WEIGHTS, test=(None, None)).data["model"]

    assert model["items"][0] == {"name": "a", "points": 2, "binary": False,
                                 "value_range": [1, 8], "name_label": "a (1–8)",
                                 "points_label": "2 × value"}
    assert model["score_range"] == [1, 17]


@pytest.mark.parametrize("weights, expected", [
    pytest.param(RISK_SCORE_WEIGHTS,
                 [("-1", "4.7%"), ("0", "11.9%"), ("1", "26.9%"), ("2", "50.0%"), ("3", "73.1%")],
                 id="narrow-range-keeps-one-cell-per-score"),
    pytest.param([0, 5, 4, 3], [("0", "50.0%"), ("3", "95.3%"), ("4", "98.2%"),
                                ("5 to 12", "> 99.0%")],
                 id="wide-range-collapses-the-high-tail"),
    pytest.param([-9, 2, 1, -1], [("-1 to 3", "< 1.0%")],
                 id="every-risk-below-one-percent-collapses-to-one-cell"),
])
def test_score_to_risk_collapses_the_tails_of_the_strip(fitted, weights, expected):
    model = make_report(fitted, weights).data["model"]

    assert [(cell["score"], cell["risk"]) for cell in model["score_to_risk"]] == expected
    assert not any(cell["positive"] for cell in model["score_to_risk"])


def test_a_discrete_strip_over_max_scores_printed_narrows_its_printed_risks(fitted):
    report = make_report(fitted, [-5, 5, 4, 2], low_risk_threshold=0.001,
                         high_risk_threshold=0.999, max_scores_printed=5)
    model, settings = report.data["model"], report.data["settings"]

    # scores 0, 2, 4, 5, 6, 7, 9, 11, all inside 0.1%..99.9%: eight cells before folding
    assert (model["score_type"], model["n_scores"], model["n_scores_printed"]) == ("discrete", 8, 5)
    # the most extreme score folds into its tail, one at a time (11, 0, 9, 2, 7), and each tail
    # reads the risk of the nearest score still printed on its own
    assert [(cell["score"], cell["risk"]) for cell in model["score_to_risk"]] == [
        ("0 to 2", "< 26.9%"), ("4", "26.9%"), ("5", "50.0%"), ("6", "73.1%"),
        ("7 to 11", "> 73.1%")]
    assert (settings["low_risk_threshold"], settings["high_risk_threshold"]) == (0.001, 0.999)
    assert settings["min_printed_risk"] == pytest.approx(expit(4 - 5))
    assert settings["max_printed_risk"] == pytest.approx(expit(6 - 5))
    # the calibration points (the training rows hold every score) and the Model card collapse
    # the same tails
    assert report.data["figures"]["calibration"]["data"][-1]["text"] == ["≤2", "4", "5", "6", "7+"]
    assert model["cell_by_score"]["9"]["cell"] == 4


@pytest.mark.parametrize("weights, expected_m, expected_rule", [
    pytest.param(CHECKLIST_WEIGHTS, 2, "Predict y if at least 2 of 2 items are checked",
                 id="positive-items"),
    pytest.param([0, 1, 1, -1], 1,
                 "Predict y if the number of checked (+) items minus the number of checked "
                 "(−) items is at least 1", id="with-negative-item"),
    pytest.param([-3, 1, 1, 0], 4,
                 "Never predict y (no set of checked items reaches the threshold)",
                 id="threshold-out-of-reach"),
])
def test_checklist_m_is_smallest_net_count_with_positive_prediction(fitted, weights, expected_m,
                                                                    expected_rule):
    model = make_report(fitted, weights).data["model"]

    assert model["type"] == "checklist"
    assert model["checklist_m"] == expected_m
    assert model["rule"] == expected_rule


@pytest.mark.parametrize("wide, model_type", [(False, "risk_score"), (True, None)],
                         ids=["forced", "inferred-for-a-non-binary-item"])
def test_unit_coefficients_make_a_risk_score_when_forced_or_an_item_is_not_binary(
        fitted, fitted_wide, wide, model_type):
    report = make_report(fitted_wide if wide else fitted, CHECKLIST_WEIGHTS, model_type=model_type)

    assert report.data["model"]["type"] == "risk_score"


@pytest.mark.parametrize("negative_label", [0, -1], ids=["labels-01", "labels-pm1"])
def test_calibration_bins_count_rows_per_score(negative_label):
    y_train = np.where(Y_TRAIN == 1, 1, negative_label)
    y_test = np.where(Y_TEST == 1, 1, negative_label)

    data = make_report(fit_classifier(X_TRAIN, y_train), RISK_SCORE_WEIGHTS, test=(X_TEST, y_test)).data
    train, test = data["calibration"]["Training"], data["calibration"]["Test"]

    assert train["scores"] == [-1, 0, 1, 2, 3]
    assert train["n"] == [1, 2, 2, 2, 1]
    assert train["observed"] == [0.0, 0.5, 0.0, 1.0, 1.0]
    assert train["ece"] == pytest.approx(0.3269805367)
    # score 1 has no test rows: absent, not NaN
    assert test["scores"] == [-1, 0, 2, 3]
    assert test["n"] == [1, 1, 1, 1]
    assert test["observed"] == [0.0, 0.0, 1.0, 1.0]
    assert test["ece"] == pytest.approx(0.2338925541)


def test_roc_has_a_point_per_score_threshold_from_origin_to_corner(report):
    roc = report.data["roc"]

    assert roc["Training"]["thresholds"] == [None, 3, 2, 1, 0, -1]
    assert roc["Training"]["fpr"] == [0.0, 0.0, 0.0, 0.5, 0.75, 1.0]
    assert roc["Training"]["tpr"] == [0.0, 0.25, 0.75, 0.75, 1.0, 1.0]
    assert roc["Training"]["auc"] == 0.84375
    assert roc["Test"]["auc"] == 1.0


def test_summary_is_one_flat_table_of_formatted_values(report):
    data = report.data

    assert data["samples"] == ["Training", "Test"]
    # one header row, naming the samples; no block subheaders and no "value" header
    assert data["summary"]["columns"] == ["", "Training", "Test"]
    # a row that is not per-sample carries one value, whatever the number of samples
    assert [(row["label"], row["values"]) for row in data["summary"]["rows"]] == [
        ("N", ["8", "4"]),
        ("Outcome Rate", ["50.0%", "50.0%"]),
        ("Model Size", ["3 (max 3)"]),
        ("Point Range", ["-5 to 5"]),
        ("Objective Value", ["0.5000"]),
        ("Optimality Gap", ["n/a"]),
        ("Run Time", ["1.2 s"]),
        ("AUC", ["0.844", "1.000"]),
        ("ECE", ["32.7%", "23.4%"]),
        ("Log Loss", ["0.579", "0.295"]),
    ]


@pytest.mark.parametrize("weights", [RISK_SCORE_WEIGHTS, CHECKLIST_WEIGHTS], ids=["risk-score", "checklist"])
def test_data_is_strict_json(fitted, weights):
    data = make_report(fitted, weights).data

    assert json.loads(json.dumps(data, allow_nan=False)) == data
    assert data["schema_version"] == 2


@pytest.mark.parametrize("invalid, match", [
    pytest.param({"weights": [float("nan"), 2, 1, -1]}, "weights must be finite", id="non-finite-weights"),
    pytest.param({"model_type": "decision_tree"}, "model_type must be one of",
                 id="unknown-model-type"),
    pytest.param({"fixture": "fitted_wide", "weights": CHECKLIST_WEIGHTS, "model_type": "checklist"},
                 r"model_type='checklist' needs binary items .*\['a'\]", id="non-binary-checklist-item"),
    pytest.param({"test": (X_TEST[:, :2], Y_TEST)}, "expecting 3 features", id="column-count"),
    pytest.param({"test": (X_TEST, Y_TEST[:-1])}, "inconsistent numbers of samples",
                 id="row-count"),
    pytest.param({"test": (X_TEST, np.where(Y_TEST == 1, 1, 2))},
                 "labels outside the classes seen in fit", id="unsupported-labels"),
    pytest.param({"test": (X_TEST, np.zeros(4))}, "'Test' has a single class",
                 id="single-class"),
    pytest.param({"components": []}, "components must name at least one", id="no-components"),
    pytest.param({"components": ["roc", "roc"]}, "components must not repeat", id="repeated-component"),
    pytest.param({"components": ["legend"]}, "components must be drawn from", id="unknown-component"),
    pytest.param({"samples": []}, "samples must name at least one", id="no-samples"),
    pytest.param({"samples": ["test", "test"]}, "samples must not repeat", id="repeated-sample"),
    # a key, not the display name; and "cv" without fit_cv is unavailable the same way
    pytest.param({"samples": ["Training"]}, "samples must be drawn from", id="unknown-sample"),
    pytest.param({"low_risk_threshold": 0.5, "high_risk_threshold": 0.5}, "risk thresholds must",
                 id="thresholds-out-of-order"),
    pytest.param({"high_risk_threshold": 1.5}, "risk thresholds must", id="threshold-above-one"),
    pytest.param({"max_scores_printed": 1}, "max_scores_printed must be an integer of at least 2",
                 id="max-scores-printed-below-two"),
])
def test_invalid_inputs_are_rejected(request, invalid, match):
    call = {"fixture": "fitted", "weights": RISK_SCORE_WEIGHTS, **invalid}
    classifier = request.getfixturevalue(call.pop("fixture"))
    with pytest.raises(ValueError, match=match):
        make_report(classifier, **call)


def test_html_holds_one_data_block_that_round_trips(fitted):
    hostile_data = BinaryClassificationDataset(X=X_TRAIN, y=Y_TRAIN, X_names=HOSTILE_NAMES,
                                               y_name="y", n_folds=())
    hostile_report = make_report(fitted, RISK_SCORE_WEIGHTS, data=hostile_data)

    page = hostile_report.html
    blocks = DATA_BLOCK.findall(page)

    assert len(blocks) == 1
    # escaped everywhere: in the data block and in the markup Jinja writes
    assert not [name for name in HOSTILE_NAMES if name in page]
    assert json.loads(blocks[0]) == hostile_report.data
    assert set(hostile_report.data) == DATA_KEYS
    assert set(hostile_report.data["figures"]) == {"roc", "calibration"}
    assert all(url in hostile_report.html for url in CDN_URLS)


def test_report_of_a_classifier_fit_on_a_dataset_holds_every_component():
    dataset = BinaryClassificationDataset.read_csv(BREASTCANCER_FILE)
    X, y = dataset.X, dataset.y
    classifier = RiskSLIMClassifier(max_size=3, max_coef=5, verbose=False, max_runtime=5,
                                    cplex_randomseed=0)
    classifier.fit(X, y).fit_cv(X, y)

    page = classifier.report(data=dataset).html
    (block,) = DATA_BLOCK.findall(page)
    data = json.loads(block)

    assert data["samples"] == ["Training", "5-CV"]
    for component in COMPONENTS:
        assert component in page
    assert data["model"]["items"]
    assert data["summary"]["columns"] == ["", *data["samples"]]
    assert data["summary"]["rows"]  # the rows themselves are pinned by the flat-table test
    for key in ("roc", "calibration"):
        # a trace is named for its legend entry: the sample, then its n, outcome rate and metric;
        # the traces run last sample first, so the first is drawn on top
        assert [trace["meta"] for trace in data["figures"][key]["data"]] == data["samples"][::-1]
        assert all(trace["name"].startswith(f"{trace['meta']}<br>")
                   for trace in data["figures"][key]["data"])


def test_settings_choose_the_cards_the_samples_and_where_the_tails_collapse(fitted):
    report = make_report(fitted, RISK_SCORE_WEIGHTS, components=["calibration", "model"],
                         samples=["test", "training"], high_risk_threshold=0.45)
    data = report.data

    assert data["settings"] == {"components": ["calibration", "model"],
                                "samples": ["test", "training"],
                                "low_risk_threshold": 0.01, "high_risk_threshold": 0.45,
                                "min_printed_risk": 0.01, "max_printed_risk": 0.45,
                                "max_scores_printed": 12}
    assert re.findall(r"<h3>(.*?)</h3>", report.html) == ["Calibration", "Model"]
    assert data["samples"] == list(data["roc"]) == list(data["calibration"]) == \
        ["Test", "Training"]
    assert data["summary"]["columns"] == ["", "Test", "Training"]
    train, test = data["figures"]["calibration"]["data"]  # and no ROC figure: its card is not shown
    assert set(data["figures"]) == {"calibration"}
    # a sample keeps its colour wherever it is listed
    assert [test["marker"]["color"], train["marker"]["color"]] == [SAMPLE_COLORS["test"],
                                                                   SAMPLE_COLORS["training"]]
    # scores 2 and 3 are above 45%: one strip cell, and one point labelled by its lowest score
    assert [(cell["score"], cell["risk"]) for cell in data["model"]["score_to_risk"][-2:]] == [
        ("1", "26.9%"), ("2 to 3", "> 45.0%")]
    assert train["text"] == ["-1", "0", "1", "2+"]


def test_save_and_notebook_display_carry_the_same_html(report, tmp_path):
    path = report.save(tmp_path / "report.html")
    iframe = report._repr_html_()
    (srcdoc,) = re.findall(r'<iframe srcdoc="([^"]*)"', iframe)

    assert path.read_text(encoding="utf-8") == report.html
    assert html.unescape(srcdoc) == report.html


@pytest.mark.parametrize("key, coordinates, legend_names", [
    ("roc", ("fpr", "tpr"), ["Training<br>(n = 8 p = 50.0%)<br>AUC = 0.844",
                             "Test<br>(n = 4 p = 50.0%)<br>AUC = 1.000"]),
    ("calibration", ("predicted", "observed"), ["Training<br>(n = 8 p = 50.0%)<br>ECE = 32.7%",
                                                "Test<br>(n = 4 p = 50.0%)<br>ECE = 23.4%"]),
])
def test_figure_plots_each_sample_section_and_names_its_trace_for_the_legend(report, key,
                                                                            coordinates,
                                                                            legend_names):
    figure = report.data["figures"][key]
    x_field, y_field = coordinates

    go.Figure(figure)  # raises on invalid Plotly properties
    # the sample's n, outcome rate and metric are the legend entry, not a box on the panel
    assert "annotations" not in figure["layout"]
    # drawn last sample first, so Training is on top; the legend lists them in page order
    assert figure["layout"]["template"]["layout"]["legend"]["traceorder"] == "reversed"
    assert [trace["name"] for trace in figure["data"]] == legend_names[::-1]
    for trace, name in zip(figure["data"], ["Test", "Training"]):
        assert trace["x"] == report.data[key][name][x_field]
        assert trace["y"] == report.data[key][name][y_field]
        # the hover names the sample on its first line, inside the box
        assert trace["hovertemplate"].startswith(f"<b>{name}</b><br>")


def test_calibration_circles_are_one_size_with_the_score_inside_joined_by_a_line(report):
    test, train = report.data["figures"]["calibration"]["data"]

    # each sample's circles, joined in risk order, with the score printed in the middle
    assert train["mode"] == test["mode"] == "lines+markers+text"
    assert train["textposition"] == "middle center"
    assert train["text"] == ["-1", "0", "1", "2", "3"]
    assert test["text"] == ["-1", "0", "2", "3"]
    assert train["x"] == sorted(train["x"])
    # one size, whatever n (train n = [1, 2, 2, 2, 1]): n is in the hover instead
    assert isinstance(train["marker"]["size"], (int, float))
    assert train["marker"]["size"] == test["marker"]["size"]


def test_calibration_points_pool_the_rows_of_a_collapsed_tail(fitted):
    report = make_report(fitted, [0, 5, 4, 3], test=(None, None))
    (train,) = report.data["figures"]["calibration"]["data"]
    section = report.data["calibration"]["Training"]

    # 8 rows, 8 distinct scores; the five above 99% risk are one point, the other three their own
    assert section["scores"] == [0, 3, 4, 5, 7, 8, 9, 12]
    # the plot labels the tail by its lowest score; the hover shows the range, as the strip does
    assert train["text"] == ["0", "3", "4", "5+"]
    scores, n, local_error = zip(*train["customdata"])  # hover: scores, n and calibration error
    assert list(scores) == ["0", "3", "4", "5 to 12"]
    assert list(n) == [1, 1, 1, 5]
    assert list(local_error) == pytest.approx(
        [0.5, 0.9525741268, 0.9820137900, 0.1983862418])
    assert train["x"][-1] == pytest.approx(0.9983862418)  # the pooled rows' mean predicted risk
    assert train["y"] == [0.0, 0.0, 0.0, pytest.approx(0.8)]  # 4 of the 5 pooled rows are positive
    # pooling moves no reported number: ECE and local_error stay on the distinct scores
    assert len(section["local_error"]) == 8
    assert section["ece"] == pytest.approx(0.4302482509)


def test_a_continuous_model_bins_its_rows_by_predicted_risk(fitted_wide):
    # scores 2a + 0.5b - c: training 2.5, 4, 5.5, 8.5, 10, 11, 13.5, 15 (risks 62%, 88%, then
    # six above 97%), test 2.5, 0, 2, -1; intercept -2
    report = make_report(fitted_wide, [-2, 2, 0.5, -1])
    model, calibration = report.data["model"], report.data["calibration"]
    test, train = report.data["figures"]["calibration"]["data"]

    assert (model["score_type"], model["n_scores"], model["n_scores_printed"]) == (
        "continuous", None, 3)
    # 12 risk bins (max_scores_printed); the strip keeps the three holding training rows, each
    # labelled by its rows' scores and reading their mean predicted risk
    assert [(cell["score"], cell["risk"]) for cell in model["score_to_risk"]] == [
        ("2.5", "62.2%"), ("4", "88.1%"), ("5.5 to 15.0", "99.5%")]
    # the Model card's lookup: each bin's score edge and strip cell, None where no row falls
    assert model["score_bins"]["cells"] == [None] * 7 + [0, None, None, 1, 2]
    assert model["score_bins"]["edges"] == pytest.approx(logit(np.arange(1, 12) / 12) + 2)
    # each sample keeps its own non-empty bins, so Test has points in bins the strip drops
    assert calibration["Training"]["score_ranges"] == [[2.5, 2.5], [4, 4], [5.5, 15]]
    assert calibration["Test"]["score_ranges"] == [[-1, -1], [0, 0], [2, 2], [2.5, 2.5]]
    # ECE over the bins: the six top rows (2 of them positive) are one bin
    assert calibration["Training"]["ece"] == pytest.approx(
        (1 - expit(0.5) + 1 - expit(2)
         + 6 * abs(2 / 6 - np.mean(expit([3.5, 6.5, 8, 9, 11.5, 13])))) / 8)
    # plain circles, with the bin's score range in the hover
    assert train["mode"] == "lines+markers" and "text" not in train
    assert [scores for scores, _, _ in train["customdata"]] == ["2.5", "4", "5.5 to 15.0"]


def test_roc_points_carry_score_thresholds(report):
    train = report.data["figures"]["roc"]["data"][-1]

    assert train["customdata"] == ["None", "Score ≥ 3", "Score ≥ 2", "Score ≥ 1",
                                   "Score ≥ 0", "Score ≥ -1"]


def test_plot_template_holds_the_same_palette_as_the_stylesheet(report):
    tokens = {name: value.strip() for name, value in
              re.findall(r"(--[\w-]+):\s*([^;]+);", (ASSETS / "styles.css").read_text())}
    layout = pio.templates[TEMPLATE_NAME].layout
    (diagonal,) = report.data["figures"]["roc"]["layout"]["shapes"]

    assert layout.font.family == tokens["--pico-font-family"]
    assert layout.paper_bgcolor == layout.plot_bgcolor == tokens["--rs-bg"]
    assert layout.font.color == layout.xaxis.title.font.color == \
        layout.yaxis.title.font.color == tokens["--rs-ink"]
    assert layout.xaxis.gridcolor == layout.yaxis.gridcolor == tokens["--rs-grid"]
    # the panel frame, the ticks and the diagonal are the grid colour, lighter than the data
    assert layout.xaxis.linecolor == layout.yaxis.linecolor == tokens["--rs-grid"]
    assert layout.xaxis.tickcolor == layout.yaxis.tickcolor == tokens["--rs-grid"]
    assert diagonal["line"]["color"] == tokens["--rs-grid"]
    assert layout.xaxis.tickfont.color == layout.yaxis.tickfont.color == tokens["--rs-axis-text"]
    # Validation has no house colour: it borrows the muted grey
    assert SAMPLE_COLORS == {"training": tokens["--rs-training"], "test": tokens["--rs-test"],
                             "cv": tokens["--rs-cv"], "validation": tokens["--rs-muted"]}


@pytest.fixture(scope="module")
def report_dir(request, tmp_path_factory):
    directory = request.config.getoption("--report-dir")
    if directory is None:
        return tmp_path_factory.mktemp("report")
    Path(directory).mkdir(parents=True, exist_ok=True)
    return Path(directory)


@pytest.fixture(scope="module", params=["risk_score", "checklist", "wide_strip"])
def saved_report(request, fitted, fitted_wide, report_dir):
    """One saved report page per model type, built and written once for all viewport widths.

    ``wide_strip`` is a continuous model (an item taking values 1..8): its strip cells are risk
    bins labelled by score ranges ("3 to 6"), and its Model card finds a score's bin by the
    shipped score edges; some bins hold no training row.
    """
    pages = {"risk_score": (fitted, RISK_SCORE_WEIGHTS, (X_TEST, Y_TEST)),
             "checklist": (fitted, CHECKLIST_WEIGHTS, (X_TEST, Y_TEST)),
             "wide_strip": (fitted_wide, [-9, 2, 1, -1], (None, None))}
    classifier, weights, test = pages[request.param]
    return make_report(classifier, weights, test=test).save(
        report_dir / f"{request.param}_report.html")


@pytest.fixture(scope="module")
def chromium():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.fail("Playwright is not installed. Run: "
                    "uv sync --group browser && uv run playwright install chromium",
                    pytrace=False)
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception as error:
            pytest.fail(f"Chromium failed to launch ({error}). Run: "
                        "uv run playwright install chromium", pytrace=False)
        yield browser
        browser.close()


@pytest.mark.browser
@pytest.mark.parametrize("width", [1280, 375])
def test_report_renders_in_browser_without_errors(chromium, saved_report, width):
    (block,) = DATA_BLOCK.findall(saved_report.read_text(encoding="utf-8"))
    data = json.loads(block)
    model = data["model"]
    items = model["items"]
    page = chromium.new_page(viewport={"width": width, "height": 900})
    errors = []
    page.on("console", lambda message: message.type == "error" and errors.append(message.text))
    page.on("pageerror", lambda error: errors.append(str(error)))

    try:
        page.goto(saved_report.as_uri(), wait_until="networkidle")
        try:
            page.wait_for_function("document.querySelectorAll('.js-plotly-plot').length === 2",
                                   timeout=15_000)
        except Exception:
            pass  # the assertions below report what did render
        page.screenshot(path=saved_report.with_name(f"{saved_report.stem}_{width}px.png"),
                        full_page=True)

        assert errors == []
        assert page.locator(".rs-model-table .rs-item-row").count() == len(items)
        assert page.locator(".js-plotly-plot").count() == 2
        # the strip fits the card at this width instead of running off the side of it, on one row
        fit = page.evaluate(STRIP_FIT)
        assert (fit["strip"], fit["card"], fit["page"]) == (0, 0, 0)
        assert fit["rows"] == 1

        def assert_readout_follows(score):
            """The readout and the outlined strip cell are those of ``score``: a discrete score's
            cell in cell_by_score; a continuous score's bin by the shipped score edges, whose
            cell may be None (a bin without training rows: risk "—", nothing outlined)."""
            bins = model["score_bins"]
            if bins is None:
                current = model["cell_by_score"][f"{score:g}"]
                cell, label = current["cell"], current["label"]
            else:
                cell = bins["cells"][sum(score >= edge for edge in bins["edges"])]
                label = f"{score:.{bins['score_digits']}f}"
            assert page.locator("#rs-score").text_content() == label
            assert page.locator("#rs-risk").text_content() == (
                "—" if cell is None else model["score_to_risk"][cell]["risk"])
            outlined = page.locator(".rs-current")
            assert outlined.count() == (cell is not None)
            assert cell is None or outlined.get_attribute("data-cell") == str(cell)

        # every box starts unchecked and every number at its item's smallest value
        score = sum(item["points"] * item["value_range"][0] for item in items if not item["binary"])
        assert_readout_follows(score)
        # checking an item adds its points to the score
        checkbox = page.locator(".rs-model-table input[type=checkbox]").first
        checkbox.check()
        score += float(checkbox.get_attribute("data-points"))
        assert_readout_follows(score)

        # typing each value of a non-binary item (integers here) adds points x (value - smallest)
        for box in page.locator(".rs-model-table input[type=number]").all():
            points, low, high = (float(box.get_attribute(name))
                                 for name in ("data-points", "min", "max"))
            for value in range(int(low), int(high) + 1):
                box.fill(str(value))
                assert_readout_follows(score + points * (value - low))
            score += points * (high - low)

        # a sample clicked in one plot's legend is hidden in both plots (a single sample has no
        # legend)
        if len(data["samples"]) > 1:
            hidden = "[...document.querySelectorAll('[data-figure]')].map((plot) => plot.data" \
                     ".filter((trace) => trace.visible === 'legendonly')" \
                     ".map((trace) => trace.meta))"
            page.locator("[data-figure=roc] .legend .traces").first.click()
            # Plotly waits out a possible double click before it acts on a click
            page.wait_for_function(f"{hidden}.flat().length > 0", timeout=5_000)
            roc_hidden, calibration_hidden = page.evaluate(hidden)
            assert len(roc_hidden) == 1 and calibration_hidden == roc_hidden
    finally:
        page.close()

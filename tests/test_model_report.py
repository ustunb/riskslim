"""Tests for riskslim.report.model_report (ModelReport: data block, Plotly figures, HTML page).

Test strategy
-------------
Fixture: the shared report sample in tests/utils.py (8 training rows, 4 test rows, three binary
features, fixed coefficient vectors). One small classifier (max size 3, points -5 to 5) is fit
once on the training rows; each report is built from a copy of it carrying a fixed coefficient
vector and fixed solver statistics, with the test rows passed as X_test / y_test, so expected
values are literals worked out by hand, not recomputed.

One end-to-end test skips the fixture: it reads data/breastcancer_data.csv into a
BinaryClassificationDataset, fits a small model and its 5 fold models (fit, then fit_cv on the
dataset's folds), and checks only that clf.report(data=...) builds a page holding every component
(model card, summary table, ROC, calibration) and both samples (Training, 5-CV). It asserts
presence, not values or appearance.

Dimensions:
  model type:   risk_score (points 2, 1, -1), checklist (+1 items only), checklist (with a -1 item)
                -- inferred from the coefficients; model_type="risk_score" overrides a checklist
  label coding: {0, 1}, {-1, 1}    -- both must map to the same positive class (one fit each)
  samples:      Training only, Training + Test (the test sample has no row with score 1)
  tails:        no score outside 1%..99% (one cell and one bubble per score), a high tail of
                many scores, a low tail holding every score -- a tail of one score is not a
                separate value: it is printed and plotted like any other, which the first case
                already covers. The strip and the calibration bubbles collapse from one rule, so
                the strip carries the three cases and the bubbles one pooling case.
  figure:       roc, calibration   -- one trace per sample, plotted straight from the data block's
                                      roc / calibration sections, each named for its legend entry
                                      (sample, n and outcome rate, AUC / ECE)
  label text:   contains "</script>" and "<!--"  -- must not end the JSON data block early (a
                                      plain label is the same path with nothing to escape, so it
                                      is not a separate case)

Rejection paths owned here (one invalid mutation of a valid call each): non-finite rho, unknown
model_type, X_test column count != the fitted feature count, y_test row count != X_test row
count, y_test labels outside the classes seen in fit, and a sample with a single class.
n/a: non-binary features for the checklist -- the score range rule is shared with the risk score.

Checks that need no browser: `plotly.graph_objects.Figure(fig)` rejects misspelled or invalid
properties, and one test asserts the Plotly template's colours are the ones styles.css writes as
`--rs-*` tokens (the page and the plots hold the palette in their own idiom, so the test is what
keeps them from drifting). The browser test is opt-in (`pytest -m browser`): each model type's page is built and
saved once, then opened in headless Chromium at 1280 px and 375 px, requiring no console or page
errors, 2 rendered Plotly charts and one model row per item; screenshots and the HTML go to a
tmp_path or --report-dir=DIR
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
from utils import (
    CHECKLIST_RHO,
    NAMES,
    RISK_SCORE_RHO,
    TRAINING,
    X_TEST,
    X_TRAIN,
    Y_TEST,
    Y_TRAIN,
)

from riskslim import RiskSLIMClassifier
from riskslim.data import BinaryClassificationDataset
from riskslim.report import ModelReport
from riskslim.report.model_report import ASSETS, TEMPLATE_NAME

DATA_KEYS = {"schema_version", "title", "outcome_name", "samples", "model", "summary", "roc",
             "calibration", "figures"}
CDN_URLS = [
    "https://cdnjs.cloudflare.com/ajax/libs/vue/3.5.43/vue.global.prod.min.js",
    "https://cdn.jsdelivr.net/npm/plotly.js-basic-dist-min@4.1.1/plotly-basic.min.js",
    "https://cdn.jsdelivr.net/npm/@picocss/pico@2.1.1/css/pico.min.css",
]
DATA_BLOCK = re.compile(r'<script type="application/json" id="report-data">(.*?)</script>', re.S)
HOSTILE_NAMES = ["a</script><script>alert(1)</script>", "b<!-- c", "c"]
BREASTCANCER_FILE = Path(__file__).parents[1] / "data" / "breastcancer_data.csv"
COMPONENTS = ["model-card", "summary-table", "roc-plot", "calibration-plot"]


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


def make_report(classifier, rho, test=(X_TEST, Y_TEST), **kwargs):
    """A report on a copy of ``classifier`` with coefficients ``rho`` (intercept first) and the
    shared solver statistics, so expected values do not depend on what the solver found."""
    model = copy.copy(classifier)
    model.intercept_, model.coef_ = float(rho[0]), np.asarray(rho[1:], dtype=float)
    model.solution_info_ = dict(TRAINING)
    X_test, y_test = test
    return ModelReport(model, X_test=X_test, y_test=y_test, **kwargs)


@pytest.fixture(scope="module")
def report(fitted):
    """The default risk-score report."""
    return make_report(fitted, RISK_SCORE_RHO)


def test_risk_score_model_has_items_and_score_range(fitted):
    model = make_report(fitted, RISK_SCORE_RHO).data["model"]

    assert model["type"] == "risk_score"
    assert [(item["name"], item["points"]) for item in model["items"]] == [
        ("a", 2), ("b", 1), ("c", -1)]
    assert model["score_range"] == [-1, 3]
    assert model["checklist_m"] is None


@pytest.fixture(scope="module")
def fitted_wide():
    """A classifier whose first feature takes values 1..8, so scores run 1..17."""
    return fit_classifier(np.column_stack([np.arange(1, 9), X_TRAIN[:, 1:]]))


def test_non_binary_feature_score_range_spans_points_times_value_range(fitted_wide):
    model = make_report(fitted_wide, RISK_SCORE_RHO, test=(None, None)).data["model"]

    assert model["items"][0] == {"name": "a", "points": 2, "binary": False,
                                 "value_range": [1, 8]}
    assert model["score_range"] == [1, 17]


@pytest.mark.parametrize("wide, rho, expected", [
    pytest.param(False, RISK_SCORE_RHO,
                 [("-1", "4.7%"), ("0", "11.9%"), ("1", "26.9%"), ("2", "50.0%"), ("3", "73.1%")],
                 id="narrow-range-keeps-one-cell-per-score"),
    pytest.param(True, RISK_SCORE_RHO,
                 [("1", "26.9%"), ("2", "50.0%"), ("3", "73.1%"), ("4", "88.1%"), ("5", "95.3%"),
                  ("6", "98.2%"), ("7 to 17", "> 99.0%")],
                 id="wide-range-collapses-the-high-tail"),
    pytest.param(False, [-9, 2, 1, -1], [("-1 to 3", "< 1.0%")],
                 id="every-risk-below-one-percent-collapses-to-one-cell"),
])
def test_score_to_risk_collapses_the_tails_of_the_strip(fitted, fitted_wide, wide, rho, expected):
    classifier, test = (fitted_wide, (None, None)) if wide else (fitted, (X_TEST, Y_TEST))

    model = make_report(classifier, rho, test=test).data["model"]

    assert [(cell["score"], cell["risk"]) for cell in model["score_to_risk"]] == expected
    assert not any(cell["positive"] for cell in model["score_to_risk"])


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
def test_checklist_m_is_smallest_net_count_with_positive_prediction(fitted, rho, expected_m,
                                                                    expected_rule):
    model = make_report(fitted, rho).data["model"]

    assert model["type"] == "checklist"
    assert model["checklist_m"] == expected_m
    assert model["rule"] == expected_rule


def test_risk_score_type_can_be_forced_for_unit_coefficients(fitted):
    report = make_report(fitted, CHECKLIST_RHO, model_type="risk_score")

    assert report.data["model"]["type"] == "risk_score"


@pytest.mark.parametrize("negative_label", [0, -1], ids=["labels-01", "labels-pm1"])
def test_calibration_bins_count_rows_per_score(negative_label):
    y_train = np.where(Y_TRAIN == 1, 1, negative_label)
    y_test = np.where(Y_TEST == 1, 1, negative_label)

    data = make_report(fit_classifier(X_TRAIN, y_train), RISK_SCORE_RHO, test=(X_TEST, y_test)).data
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


def test_summary_has_four_blocks_of_formatted_values(report):
    data = report.data

    assert data["samples"] == ["Training", "Test"]
    assert {block["key"]: block["rows"] for block in data["summary"]} == {
        "dataset": [["n", "8", "4"], ["outcome rate", "50.0%", "50.0%"]],
        "constraints": [["model size", "3 (max 3)"], ["point range", "-5 to 5"]],
        "training": [["objective value", "0.5000"], ["optimality gap", "n/a"],
                     ["run time", "1.2 s"]],
        "performance": [["AUC", "0.844", "1.000"], ["ECE", "32.7%", "23.4%"],
                        ["log loss", "0.579", "0.295"]],
    }


@pytest.mark.parametrize("rho", [RISK_SCORE_RHO, CHECKLIST_RHO], ids=["risk-score", "checklist"])
def test_data_is_strict_json(fitted, rho):
    data = make_report(fitted, rho).data

    assert json.loads(json.dumps(data, allow_nan=False)) == data
    assert data["schema_version"] == 1


@pytest.mark.parametrize("invalid, match", [
    pytest.param({"rho": [float("nan"), 2, 1, -1]}, "rho must be finite", id="non-finite-rho"),
    pytest.param({"model_type": "decision_tree"}, "model_type must be one of",
                 id="unknown-model-type"),
    pytest.param({"test": (X_TEST[:, :2], Y_TEST)}, "expecting 3 features", id="column-count"),
    pytest.param({"test": (X_TEST, Y_TEST[:-1])}, "inconsistent numbers of samples",
                 id="row-count"),
    pytest.param({"test": (X_TEST, np.where(Y_TEST == 1, 1, 2))},
                 "labels outside the classes seen in fit", id="unsupported-labels"),
    pytest.param({"test": (X_TEST, np.zeros(4))}, "'Test' has a single class",
                 id="single-class"),
])
def test_invalid_inputs_are_rejected(fitted, invalid, match):
    call = {"rho": RISK_SCORE_RHO, **invalid}
    with pytest.raises(ValueError, match=match):
        make_report(fitted, **call)


def test_html_holds_one_data_block_that_round_trips(fitted):
    hostile_data = BinaryClassificationDataset(X=X_TRAIN, y=Y_TRAIN, X_names=HOSTILE_NAMES,
                                               y_name="y", n_folds=())
    hostile_report = make_report(fitted, RISK_SCORE_RHO, data=hostile_data)

    blocks = DATA_BLOCK.findall(hostile_report.html)

    assert len(blocks) == 1
    assert json.loads(blocks[0]) == hostile_report.data
    assert set(hostile_report.data) == DATA_KEYS
    assert set(hostile_report.data["figures"]) == {"roc", "calibration"}
    assert all(url in hostile_report.html for url in CDN_URLS)


def test_report_of_a_classifier_fit_on_a_dataset_holds_every_component():
    dataset = BinaryClassificationDataset.read_csv(BREASTCANCER_FILE, n_folds=(5,))
    X, y = dataset.X, dataset.y
    classifier = RiskSLIMClassifier(max_size=3, max_coef=5, verbose=False, max_runtime=5,
                                    cplex_randomseed=0)
    classifier.fit(X, y).fit_cv(X, y, data=dataset)

    page = classifier.report(data=dataset).html
    (block,) = DATA_BLOCK.findall(page)
    data = json.loads(block)

    assert data["samples"] == ["Training", "5-CV"]
    for component in COMPONENTS:
        assert f'data-component="{component}"' in page
    assert data["model"]["items"]
    assert [block["key"] for block in data["summary"]] == [
        "dataset", "constraints", "training", "performance"]
    (performance,) = [block for block in data["summary"] if block["key"] == "performance"]
    assert all(len(row) == 1 + len(data["samples"]) for row in performance["rows"])
    for key in ("roc", "calibration"):
        # a trace is named for its legend entry: the sample, then its n, outcome rate and metric
        assert [trace["name"].split("<br>")[0] for trace in data["figures"][key]["data"]] == \
            data["samples"]


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
    assert [trace["name"] for trace in figure["data"]] == legend_names
    for trace, name in zip(figure["data"], ["Training", "Test"]):
        assert trace["x"] == report.data[key][name][x_field]
        assert trace["y"] == report.data[key][name][y_field]
        assert trace["hovertemplate"].endswith(f"<extra>{name}</extra>")  # hover names the sample


def test_calibration_bubbles_are_labelled_with_scores_and_sized_by_n(report):
    train, test = report.data["figures"]["calibration"]["data"]

    assert train["textposition"] == "top center"  # outside the bubble, so the label is readable
    assert train["text"] == ["-1", "0", "1", "2", "3"]
    assert test["text"] == ["-1", "0", "2", "3"]
    # train n = [1, 2, 2, 2, 1]; test n = [1, 1, 1, 1]
    small, large = train["marker"]["size"][0], train["marker"]["size"][1]
    assert small < large
    assert train["marker"]["size"] == [small, large, large, large, small]
    assert test["marker"]["size"] == [small] * 4


def test_calibration_bubbles_pool_the_rows_of_a_collapsed_tail(fitted_wide):
    report = make_report(fitted_wide, RISK_SCORE_RHO, test=(None, None))
    (train,) = report.data["figures"]["calibration"]["data"]
    section = report.data["calibration"]["Training"]

    # 8 rows, 8 distinct scores; the five above 99% risk are one bubble, the other three their own
    assert section["scores"] == [3, 4, 6, 9, 10, 11, 14, 15]
    assert train["text"] == ["3", "4", "6", "9 to 15"]
    n, local_error = zip(*train["customdata"])  # hover: the bubble's n and calibration error
    assert list(n) == [1, 1, 1, 5]
    assert list(local_error) == pytest.approx(
        [0.2689414214, 0.1192029220, 0.0179862100, 0.7997243599])
    assert train["x"][-1] == pytest.approx(0.9997243599)  # the pooled rows' mean predicted risk
    assert train["y"] == [1.0, 1.0, 1.0, pytest.approx(0.2)]  # 1 of the 5 pooled rows is positive
    # pooling moves no reported number: ECE and local_error stay on the distinct scores
    assert len(section["local_error"]) == 8
    assert section["ece"] == pytest.approx(0.5505955802)


def test_roc_points_carry_score_thresholds(report):
    train = report.data["figures"]["roc"]["data"][0]

    assert train["customdata"] == ["none", "score ≥ 3", "score ≥ 2", "score ≥ 1",
                                   "score ≥ 0", "score ≥ -1"]


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
    assert list(layout.colorway) == [tokens[f"--rs-sample-{i}"] for i in (1, 2, 3)]


@pytest.fixture(scope="module")
def report_dir(request, tmp_path_factory):
    directory = request.config.getoption("--report-dir")
    if directory is None:
        return tmp_path_factory.mktemp("report")
    Path(directory).mkdir(parents=True, exist_ok=True)
    return Path(directory)


@pytest.fixture(scope="module", params=["risk_score", "checklist"])
def saved_report(request, fitted, report_dir):
    """One saved report page per model type, built and written once for all viewport widths."""
    rho = {"risk_score": RISK_SCORE_RHO, "checklist": CHECKLIST_RHO}[request.param]
    return make_report(fitted, rho).save(report_dir / f"{request.param}_report.html")


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
    items = json.loads(block)["model"]["items"]
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
    finally:
        page.close()

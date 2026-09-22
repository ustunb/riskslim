"""Tests for riskslim.report.render_report_html and build_plot_figures (Report: Plotly figures,
HTML page, save, notebook display).

Test strategy
-------------
Input: report data built from the shared report sample in tests/utils.py (fixed
coefficients, no solver).

Dimensions:
  figure:      roc, calibration            -- one trace per sample, plotted straight from the
                                              data block's roc / calibration sections, plus a
                                              top-left metrics box (AUC / calibration error)
  model type:  risk_score, checklist       -- only the browser test; the HTML shell is the same
  layout:      default, custom             -- a custom layout replaces the data block's layout
  label text:  contains "</script>" and "<!--"  -- must not end the JSON data block early
                                              (a plain label is the same path with nothing to
                                              escape, so it is not a separate case)

Checks that need no browser: `plotly.graph_objects.Figure(fig)` rejects misspelled or invalid
properties. The browser test is opt-in (`pytest -m browser`): each model type's page is built and
saved once, then opened in headless Chromium at 1280 px and 375 px, requiring no console or page
errors, 2 rendered Plotly charts and one model row per item; screenshots and the HTML go to a
tmp_path or --report-dir=DIR
(write it with "=": with a space, pytest reads an existing DIR as a test path and misses the
config).
"""

import html
import json
import re
from pathlib import Path

import plotly.graph_objects as go
import pytest
from utils import CHECKLIST_RHO, CONSTRAINTS, NAMES, RISK_SCORE_RHO, SAMPLES, TRAINING

from riskslim.report import CalibrationPlot, ModelCard, Report, RocPlot, Row, build_report_data

CDN_URLS = [
    "https://cdnjs.cloudflare.com/ajax/libs/vue/3.5.43/vue.global.prod.min.js",
    "https://cdn.jsdelivr.net/npm/plotly.js-basic-dist-min@4.1.1/plotly-basic.min.js",
    "https://cdn.jsdelivr.net/npm/@picocss/pico@2.1.1/css/pico.min.css",
]
DATA_BLOCK = re.compile(r'<script type="application/json" id="report-data">(.*?)</script>', re.S)
HOSTILE_NAMES = ["(Intercept)", "a</script><script>alert(1)</script>", "b<!-- c", "c"]


def make_report(rho, names=NAMES, layout=None):
    data = build_report_data(rho, names, "y", SAMPLES, training=TRAINING, constraints=CONSTRAINTS)
    return Report(data, layout=layout)


@pytest.fixture(scope="module")
def report():
    """The default risk-score report (no layout override)."""
    return make_report(RISK_SCORE_RHO)


def test_html_holds_one_data_block_that_round_trips():
    data = build_report_data(RISK_SCORE_RHO, HOSTILE_NAMES, "y", SAMPLES,
                             training=TRAINING, constraints=CONSTRAINTS)
    hostile_report = Report(data)

    blocks = DATA_BLOCK.findall(hostile_report.html)

    assert len(blocks) == 1
    assert json.loads(blocks[0]) == hostile_report.data
    assert {key: hostile_report.data[key] for key in data} == data
    assert set(hostile_report.data["figures"]) == {"roc", "calibration"}
    assert all(url in hostile_report.html for url in CDN_URLS)


def test_custom_layout_replaces_default():
    custom_report = make_report(RISK_SCORE_RHO, layout=[Row(RocPlot()),
                                                        Row(CalibrationPlot(), ModelCard())])

    assert custom_report.data["layout"] == [
        {"components": [{"component": "RocPlot", "options": {"title": "ROC"}}]},
        {"components": [{"component": "CalibrationPlot", "options": {"title": "Calibration"}},
                        {"component": "ModelCard", "options": {"title": "Model"}}]},
    ]


def test_save_and_notebook_display_carry_the_same_html(report, tmp_path):
    path = report.save(tmp_path / "report.html")
    iframe = report._repr_html_()
    (srcdoc,) = re.findall(r'<iframe srcdoc="([^"]*)"', iframe)

    assert path.read_text(encoding="utf-8") == report.html
    assert html.unescape(srcdoc) == report.html


@pytest.mark.parametrize("key, coordinates, metrics_text", [
    ("roc", ("fpr", "tpr"), ["AUC", "train</span> 0.844", "test</span> 1.000"]),
    ("calibration", ("predicted", "observed"), ["CAL", "train</span> 32.7%",
                                                "test</span> 23.4%"]),
])
def test_figure_plots_each_sample_section_with_a_top_left_metrics_box(report, key, coordinates,
                                                                     metrics_text):
    figure = report.data["figures"][key]
    x_field, y_field = coordinates

    go.Figure(figure)  # raises on invalid Plotly properties
    assert [trace["name"] for trace in figure["data"]] == ["train", "test"]
    for trace, name in zip(figure["data"], ["train", "test"]):
        assert trace["x"] == report.data[key][name][x_field]
        assert trace["y"] == report.data[key][name][y_field]
    (box,) = figure["layout"]["annotations"]
    assert (box["xref"], box["yref"], box["xanchor"], box["yanchor"]) == (
        "paper", "paper", "left", "top")
    assert box["x"] <= 0.05 and box["y"] >= 0.95
    assert all(text in box["text"] for text in metrics_text), box["text"]


def test_calibration_bubbles_are_labelled_with_scores_and_sized_by_n(report):
    train, test = report.data["figures"]["calibration"]["data"]

    assert train["text"] == ["-1", "0", "1", "2", "3"]
    assert test["text"] == ["-1", "0", "2", "3"]
    # train n = [1, 2, 2, 2, 1]; test n = [1, 1, 1, 1]
    small, large = train["marker"]["size"][0], train["marker"]["size"][1]
    assert small < large
    assert train["marker"]["size"] == [small, large, large, large, small]
    assert test["marker"]["size"] == [small] * 4


def test_roc_points_carry_score_thresholds(report):
    train = report.data["figures"]["roc"]["data"][0]

    assert train["customdata"] == ["none", "score ≥ 3", "score ≥ 2", "score ≥ 1",
                                   "score ≥ 0", "score ≥ -1"]


@pytest.fixture(scope="module")
def report_dir(request, tmp_path_factory):
    directory = request.config.getoption("--report-dir")
    if directory is None:
        return tmp_path_factory.mktemp("report")
    Path(directory).mkdir(parents=True, exist_ok=True)
    return Path(directory)


@pytest.fixture(scope="module", params=["risk_score", "checklist"])
def saved_report(request, report_dir):
    """One saved report page per model type, built and written once for all viewport widths."""
    rho = {"risk_score": RISK_SCORE_RHO, "checklist": CHECKLIST_RHO}[request.param]
    return make_report(rho).save(report_dir / f"{request.param}_report.html")


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

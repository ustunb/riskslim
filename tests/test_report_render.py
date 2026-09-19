"""Tests for riskslim.report.render (Report: HTML page, save, notebook display).

Test strategy
-------------
Input: report data from test_report_data's fixture (fixed coefficients, no solver).

Dimensions:
  model type:  risk_score, checklist        -- only the browser test; the HTML shell is the same
  layout:      default, custom              -- a custom layout replaces the data block's layout
  label text:  plain, contains "</script>"  -- must not end the JSON data block early

The browser test is opt-in (`pytest -m browser`): it opens each report in headless Chromium at
1280 px and 375 px, requires no console or page errors, 2 rendered Plotly charts and one model
row per item, and saves screenshots plus the HTML to tmp_path or --report-dir.
"""

import html
import json
import re
from pathlib import Path

import pytest

from riskslim.report import CalibrationPlot, ModelCard, Report, RocPlot, Row, build_report_data
from test_report_data import (
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

CDN_URLS = [
    "https://cdnjs.cloudflare.com/ajax/libs/vue/3.5.43/vue.global.prod.min.js",
    "https://cdn.jsdelivr.net/npm/plotly.js-basic-dist-min@4.1.1/plotly-basic.min.js",
    "https://cdn.jsdelivr.net/npm/@picocss/pico@2.1.1/css/pico.min.css",
]
DATA_BLOCK = re.compile(r'<script type="application/json" id="report-data">(.*?)</script>', re.S)


def make_report(rho, names=NAMES, layout=None):
    data = build_report_data(rho, names, "y",
                             {"train": (X_TRAIN, Y_TRAIN), "test": (X_TEST, Y_TEST)},
                             training=TRAINING, constraints=CONSTRAINTS)
    return data, Report(data, layout=layout)


@pytest.mark.parametrize("names", [
    pytest.param(NAMES, id="plain"),
    pytest.param(["(Intercept)", "a</script><script>alert(1)</script>", "b", "c"],
                 id="script-tag-in-label"),
])
def test_html_holds_one_data_block_that_round_trips(names):
    data, report = make_report(RISK_SCORE_RHO, names=names)

    blocks = DATA_BLOCK.findall(report.html)

    assert len(blocks) == 1
    assert json.loads(blocks[0]) == report.data
    assert {key: report.data[key] for key in data} == data
    assert set(report.data["figures"]) == {"roc", "calibration"}
    assert all(url in report.html for url in CDN_URLS)


def test_custom_layout_replaces_default():
    _, report = make_report(RISK_SCORE_RHO, layout=[Row(RocPlot()), Row(CalibrationPlot(),
                                                                        ModelCard())])

    assert report.data["layout"] == [
        {"components": [{"component": "RocPlot", "options": {"title": "ROC"}}]},
        {"components": [{"component": "CalibrationPlot", "options": {"title": "Calibration"}},
                        {"component": "ModelCard", "options": {"title": "Model"}}]},
    ]


def test_save_and_notebook_display_carry_the_same_html(tmp_path):
    _, report = make_report(RISK_SCORE_RHO)

    path = report.save(tmp_path / "report.html")
    iframe = report._repr_html_()
    (srcdoc,) = re.findall(r'<iframe srcdoc="([^"]*)"', iframe)

    assert path.read_text(encoding="utf-8") == report.html
    assert html.unescape(srcdoc) == report.html


@pytest.fixture
def report_dir(request, tmp_path):
    directory = request.config.getoption("--report-dir")
    if directory is None:
        return tmp_path
    Path(directory).mkdir(parents=True, exist_ok=True)
    return Path(directory)


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
@pytest.mark.parametrize("name, rho", [("risk_score", RISK_SCORE_RHO),
                                       ("checklist", CHECKLIST_RHO)])
@pytest.mark.parametrize("width", [1280, 375])
def test_report_renders_in_browser_without_errors(chromium, report_dir, name, rho, width):
    _, report = make_report(rho)
    html_path = report.save(report_dir / f"{name}_report.html")
    page = chromium.new_page(viewport={"width": width, "height": 900})
    errors = []
    page.on("console", lambda message: message.type == "error" and errors.append(message.text))
    page.on("pageerror", lambda error: errors.append(str(error)))

    page.goto(html_path.as_uri(), wait_until="networkidle")
    try:
        page.wait_for_function("document.querySelectorAll('.js-plotly-plot').length === 2",
                               timeout=15_000)
    except Exception:
        pass  # the assertions below report what did render
    page.screenshot(path=report_dir / f"{name}_report_{width}px.png", full_page=True)

    assert errors == []
    assert page.locator(".rs-model-table .rs-item-row").count() == len(
        report.data["model"]["items"])
    assert page.locator(".js-plotly-plot").count() == 2
    page.close()

"""Tests for riskslim.report.layout (component specs and the layout JSON).

Test strategy
-------------
The layouts that work are covered where they are observed: the default layout in
test_report_data.py (``data["layout"]``) and a custom one in test_report_render.py. What is left
is the rejection paths this module owns, each one invalid mutation of a valid layout:

  SummaryTable:    a block name outside SUMMARY_BLOCKS
  Row:             an argument that is not a component spec
  layout_to_json:  a bare Row instead of a list, and a list entry that is not a Row

n/a: default_layout's model_type check -- build_report_data rejects an unknown model_type before
it reaches layout.py, so it has no separately observable behavior.
"""

import pytest

from riskslim.report import CalibrationPlot, ModelCard, RocPlot, Row, SummaryTable
from riskslim.report.layout import layout_to_json


@pytest.mark.parametrize("build_layout, match", [
    pytest.param(lambda: SummaryTable(blocks=("dataset", "costs")), "blocks must be among",
                 id="unknown-summary-block"),
    pytest.param(lambda: Row(ModelCard(), "RocPlot"), "Row takes component specs",
                 id="row-given-a-non-component"),
    pytest.param(lambda: layout_to_json(Row(RocPlot(), CalibrationPlot())), "list of Row",
                 id="bare-row"),
    pytest.param(lambda: layout_to_json([Row(RocPlot()), CalibrationPlot()]), "list of Row",
                 id="non-row-entry"),
])
def test_invalid_layouts_are_rejected(build_layout, match):
    with pytest.raises(Exception, match=match):
        build_layout()

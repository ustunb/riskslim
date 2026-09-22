"""Render a report: a Jinja shell, the inlined CSS/JS, one JSON data block and two Plotly figures.

The figures come from ``build_plot_figures``.
"""

import html
import json
from functools import cache
from importlib.resources import files
from pathlib import Path

from jinja2 import Environment
from markupsafe import Markup

from .build_plot_figures import build_figures
from .report_layout_components import layout_to_json

ASSETS = files(__package__) / "assets"


class Report:
    """An HTML report for a fitted model.

    ``report.html`` is a single self-contained page (libraries load from a pinned CDN),
    ``report.save(path)`` writes it, and notebooks display it inline in an iframe.

    Parameters
    ----------
    data : dict
        Output of ``build_report_data``.
    layout : list of Row, optional
        Replaces the default layout in ``data``.
    """

    def __init__(self, data, layout=None):
        self.data = {**data, "figures": build_figures(data)}
        if layout is not None:
            self.data["layout"] = layout_to_json(layout)

    @property
    def html(self):
        """The report page as a string."""
        shell, styles, scripts = load_assets()
        return shell.render(title=self.data["title"], styles=styles, scripts=scripts,
                            data=Markup(json_for_script(self.data)))

    def save(self, path):
        """Write the report to ``path`` (an ``.html`` file) and return the path."""
        path = Path(path)
        path.write_text(self.html, encoding="utf-8")
        return path

    def _repr_html_(self):
        return (f'<iframe srcdoc="{html.escape(self.html, quote=True)}" '
                f'style="width: 100%; height: 1000px; border: 0;"></iframe>')


@cache
def load_assets():
    """The compiled shell template and the CSS/JS it inlines, read once per process."""
    shell = Environment(autoescape=True).from_string((ASSETS / "report_page_template.html").read_text(encoding="utf-8"))
    styles = Markup((ASSETS / "report_styles.css").read_text(encoding="utf-8"))
    scripts = Markup((ASSETS / "draw_report_components.js").read_text(encoding="utf-8"))
    return shell, styles, scripts


def json_for_script(data):
    """JSON safe inside ``<script type="application/json">``: ``</`` becomes ``<\\/``."""
    text = json.dumps(data, allow_nan=False, ensure_ascii=False)
    return text.replace("</", "<\\/").replace("<!--", "<\\u0021--")

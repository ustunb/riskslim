"""Build the report's two Plotly figures: ROC and calibration.

The ROC and calibration figures are plain JSON dicts built from ``build_report_data`` output; the
browser passes them to ``Plotly.newPlot``. All visual choices live in ``STYLE`` and the figure
functions below, so tweaks happen here in Python.
"""

import math
from itertools import cycle

# ---------------------------------------------------------------------------
# Visuals: tweak the look of both figures here.
# nsf-career palette (see report_styles.css for the same tokens)
# ---------------------------------------------------------------------------
STYLE = {
    "ink": "#1F2933",
    "muted": "#98A2AD",
    "grid": "#DFE4E9",
    "stripe": "#F2F2F2",
    "bg": "#FFFFFF",
    "samples": ["#2F6FB3", "#E07B39", "#10B981"],  # train, test, third sample
    "font": '-apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif',
    "height": 360,
    "bubble_px": (14, 34),  # calibration bubble diameter for the smallest and largest n
}


def build_figures(data):
    """``{"roc": fig, "calibration": fig}`` for a report data dict."""
    return {"roc": roc_figure(data), "calibration": calibration_figure(data)}


def roc_figure(data):
    """One ROC curve per sample with a point at each score threshold; AUC box top-left."""
    colors = [color for _, color in zip(data["samples"], cycle(STYLE["samples"]))]
    traces = []
    for name, color in zip(data["samples"], colors):
        roc = data["roc"][name]
        labels = ["none" if t is None else f"score ≥ {t}" for t in roc["thresholds"]]
        traces.append({
            "type": "scatter", "mode": "lines+markers", "name": name,
            "x": roc["fpr"], "y": roc["tpr"], "customdata": labels,
            "line": {"color": color, "width": 2}, "marker": {"color": color, "size": 6},
            "hovertemplate": "%{customdata}<br>FPR %{x:.1%} · TPR %{y:.1%}"
                             f"<extra>{name}</extra>",
        })
    metrics = [(name, f"{data['roc'][name]['auc']:.3f}") for name in data["samples"]]
    layout = base_layout("False positive rate", "True positive rate", "AUC", metrics, colors)
    return {"data": traces, "layout": layout}


def calibration_figure(data):
    """Bubbles per score, sized by n and labelled with the score; CAL box top-left."""
    n_max = max(n for name in data["samples"] for n in data["calibration"][name]["n"])
    d_min, d_max = STYLE["bubble_px"]
    colors = [color for _, color in zip(data["samples"], cycle(STYLE["samples"]))]
    traces = []
    for name, color in zip(data["samples"], colors):
        cal = data["calibration"][name]
        traces.append({
            "type": "scatter", "mode": "markers+text", "name": name,
            "x": cal["predicted"], "y": cal["observed"],
            "text": [str(s) for s in cal["scores"]], "customdata": [[n] for n in cal["n"]],
            "textposition": "middle center",
            "textfont": {"size": 9, "color": STYLE["bg"]},
            "cliponaxis": False,
            "marker": {"color": color, "opacity": 0.85, "line": {"color": STYLE["bg"], "width": 1},
                       "size": [round(d_min + (d_max - d_min) * math.sqrt(n / n_max), 2)
                                for n in cal["n"]]},
            "hovertemplate": "score %{text}<br>predicted risk %{x:.1%}<br>"
                             "observed risk %{y:.1%}<br>n = %{customdata[0]:,}"
                             f"<extra>{name}</extra>",
        })
    metrics = [(name, f"{data['calibration'][name]['error']:.1%}") for name in data["samples"]]
    layout = base_layout("Predicted risk", "Observed risk", "CAL", metrics, colors,
                         axis_overrides={"range": [-0.03, 1.03], "tickformat": ".0%"})
    return {"data": traces, "layout": layout}


def base_layout(x_title, y_title, metric, metrics, colors, axis_overrides=None):
    """Shared axes, fonts, diagonal and the top-left metrics box."""
    axis = {
        "showgrid": True, "gridcolor": STYLE["grid"], "zeroline": False, "showline": True,
        "linecolor": STYLE["ink"], "ticks": "outside", "ticklen": 3, "fixedrange": True,
        "tickfont": {"size": 11, "color": STYLE["muted"]},
        "title": {"font": {"size": 12, "color": STYLE["ink"]}},
        "range": [-0.02, 1.02], "dtick": 0.2,
        **(axis_overrides or {}),
    }
    lines = [f"<b>{metric}</b>"] + [
        f'<span style="color:{color}">{name}</span> {value}'
        for (name, value), color in zip(metrics, colors)
    ]
    return {
        "height": STYLE["height"], "autosize": True,
        "margin": {"t": 36, "r": 12, "b": 48, "l": 56},
        "font": {"family": STYLE["font"], "size": 11, "color": STYLE["ink"]},
        "paper_bgcolor": STYLE["bg"], "plot_bgcolor": STYLE["bg"],
        "xaxis": {**axis, "title": {**axis["title"], "text": x_title}},
        "yaxis": {**axis, "title": {**axis["title"], "text": y_title}},
        "hovermode": "closest", "dragmode": False,
        "legend": {"orientation": "h", "x": 1, "xanchor": "right", "y": 1.01,
                   "yanchor": "bottom"},
        "shapes": [{"type": "line", "x0": 0, "y0": 0, "x1": 1, "y1": 1, "layer": "below",
                    "line": {"color": STYLE["muted"], "width": 1, "dash": "dash"}}],
        "annotations": [{
            "xref": "paper", "yref": "paper", "x": 0.02, "y": 0.98,
            "xanchor": "left", "yanchor": "top", "align": "left", "showarrow": False,
            "text": "<br>".join(lines), "bgcolor": STYLE["stripe"],
            "bordercolor": STYLE["grid"], "borderpad": 4,
        }],
    }

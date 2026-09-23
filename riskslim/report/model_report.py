"""The report: one object that computes the page, builds its figures and renders the HTML.

``ModelReport`` takes plain arrays (no fitted classifier, no solver). It computes one
JSON-serializable data block with no NaN or inf, builds the two Plotly figures from it, and
renders a single self-contained page (libraries load from a pinned CDN). The browser only
displays what Python computed.

Conventions
-----------
- ``rho`` and ``variable_names`` follow riskslim's convention and **include the intercept** at
  index 0, named ``"(Intercept)"``. ``X`` in each sample **excludes** the intercept column (it is
  the matrix passed to ``fit``).
- The **score** of a row is ``X @ rho[1:]`` (points only, no intercept). Its risk is
  ``1 / (1 + exp(-(score + intercept)))``.
- A row is predicted positive when ``score + intercept > 0`` (risk strictly above 0.5), the same
  rule as ``RiskSLIMClassifier.predict``.
- **Checklist M:** with k = (#checked +1 items) - (#checked -1 items), M is the smallest integer
  k with ``k + intercept > 0``, i.e. ``floor(-intercept) + 1``.
- **Score range:** ``sum(points * [min, max])`` of each item over the training data (a binary
  item contributes ``points * [0, 1]``). The score-to-risk row lists every achievable total when
  all item values are integers, and otherwise the totals observed in the samples.
- **Calibration error:** the n-weighted mean of ``|predicted - observed|`` over score bins. Score
  bins with no rows in a sample are absent from that sample.
"""

import html
import json
import math
from functools import cache
from importlib.resources import files
from itertools import cycle
from pathlib import Path

import numpy as np
from jinja2 import Environment
from markupsafe import Markup
from scipy.special import expit
from sklearn.metrics import auc, roc_curve

from ..defaults import INTERCEPT_NAME
from ..loss_functions.log_loss import log_loss_value_from_scores
from ..utils import is_integer

SCHEMA_VERSION = 1
MAX_TOTALS = 10_000  # score totals to enumerate before falling back to observed scores
MODEL_TYPES = {"risk_score": "Risk score", "checklist": "Checklist"}

ASSETS = files(__package__) / "assets"
# Inlined in this order: each component file defines its component globally, and mount_report.js
# mounts it on its slot in assets/template.html, where the slot's markup is its in-DOM template.
SCRIPTS = ("model_card.js", "summary_table.js", "roc_plot.js", "calibration_plot.js",
           "mount_report.js")

# ---------------------------------------------------------------------------
# Visuals: the figures name the page's CSS custom properties; the browser resolves
# them before drawing (see drawFigure in assets/mount_report.js), so assets/styles.css
# is the only place a colour, the font or the plot height is written.
# ---------------------------------------------------------------------------
SAMPLE_COLORS = ["var(--rs-sample-1)", "var(--rs-sample-2)", "var(--rs-sample-3)"]
BUBBLE_PX = (14, 34)  # calibration bubble diameter for the smallest and largest n


class ModelReport:
    """An HTML report for a risk score or a checklist.

    ``report.data`` is everything the page shows, ``report.html`` is the page as a string,
    ``report.save(path)`` writes it, and notebooks display it inline in an iframe.

    Parameters
    ----------
    rho : 1d array
        Coefficients, intercept first.
    variable_names : list of str
        Names matching ``rho``; the first is ``"(Intercept)"``.
    outcome_name : str
        Name of the positive outcome.
    samples : dict
        ``{"train": (X, y), "test": (X, y)}``; ``train`` is required. X excludes the intercept
        column; y is in {0, 1} or {-1, 1}.
    training : dict, optional
        Solver statistics (``RiskSLIMClassifier.solution_info_``); uses ``objective_value``,
        ``optimality_gap`` and ``run_time``.
    constraints : dict, optional
        ``max_size`` (int) and ``point_range`` ((lb, ub) over the non-intercept coefficients).
    model_type : {"risk_score", "checklist"}, optional
        Inferred from the coefficients when None.
    """

    def __init__(self, rho, variable_names, outcome_name, samples, training=None,
                 constraints=None, model_type=None):
        self.rho, self.variable_names = checked_coefficients(rho, variable_names)
        self.outcome_name = str(outcome_name)
        self.training = training
        self.constraints = dict(constraints or {})
        self.model_type = checked_model_type(model_type, self.rho[1:])

        # samples stay local: the page needs only what they produce, and holding them would pin
        # a float64 copy of every X for the report's lifetime
        samples = checked_samples(samples, len(self.variable_names) - 1)
        intercept, points = float(self.rho[0]), self.rho[1:]

        scores = {name: X @ points for name, (X, _) in samples.items()}
        model = model_section(points, intercept, self.variable_names[1:], self.outcome_name,
                              self.model_type, samples["train"][0], scores)
        roc = {name: roc_section(y, scores[name]) for name, (_, y) in samples.items()}
        calibration = {name: calibration_section(y, scores[name], intercept)
                       for name, (_, y) in samples.items()}
        log_loss = {name: float(log_loss_value_from_scores((2 * y - 1) * (scores[name] + intercept)))
                    for name, (_, y) in samples.items()}
        names = list(samples)
        self.data = {
            "schema_version": SCHEMA_VERSION,
            "title": f"{MODEL_TYPES[self.model_type]}: {self.outcome_name}",
            "outcome_name": self.outcome_name,
            "samples": names,
            "model": model,
            "summary": summary_section(samples, model, roc, calibration, log_loss, self.training,
                                       self.constraints),
            "roc": roc,
            "calibration": calibration,
        }
        self.data["figures"] = {"roc": roc_figure(names, roc),
                                "calibration": calibration_figure(names, calibration)}

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


# ---------------------------------------------------------------------------
# Report data: everything the page shows, computed once in Python.
# ---------------------------------------------------------------------------

def infer_model_type(points):
    """``checklist`` when every nonzero coefficient (intercept excluded) is +1 or -1."""
    nonzero = points[points != 0]
    if nonzero.size > 0 and np.all(np.abs(nonzero) == 1):
        return "checklist"
    return "risk_score"


def checked_coefficients(rho, variable_names):
    """Validate the coefficients and their names; return them as an array and a list."""
    rho = np.asarray(rho, dtype=float).ravel()
    names = list(variable_names)
    if len(names) != len(rho) or not names or names[0] != INTERCEPT_NAME:
        raise ValueError(
            f"rho and variable_names must have the same length with {INTERCEPT_NAME!r} first; "
            f"got {len(rho)} coefficients and names {names[:3]}..."
        )
    if not np.all(np.isfinite(rho)):
        raise ValueError(f"rho must be finite; got {rho.tolist()}")
    return rho, names


def checked_model_type(model_type, points):
    """Validate the model type, or infer it from the coefficients when None."""
    if model_type is None:
        return infer_model_type(points)
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {tuple(MODEL_TYPES)}; got {model_type!r}")
    return model_type


def checked_samples(samples, n_features):
    """Validate and normalize: ``{name: (X float 2d, y in {0, 1})}``, ``train`` first."""
    if not isinstance(samples, dict) or "train" not in samples:
        keys = sorted(samples) if isinstance(samples, dict) else type(samples).__name__
        raise ValueError(f"samples must be a dict with a 'train' entry, "
                         f"e.g. {{'train': (X, y)}}; got {keys}")
    checked = {}
    for name in ["train", *[k for k in samples if k != "train"]]:
        X, y = samples[name]
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.ndim != 2 or X.shape[1] != n_features:
            raise ValueError(
                f"sample {name!r}: X has shape {X.shape} but variable_names lists {n_features} "
                f"features (excluding {INTERCEPT_NAME!r}); pass X without the intercept column"
            )
        if len(y) != X.shape[0]:
            raise ValueError(f"sample {name!r}: X has {X.shape[0]} rows but y has {len(y)}")
        labels = set(np.unique(y).tolist())
        if not (labels <= {0, 1} or labels <= {-1, 1}):
            raise ValueError(f"sample {name!r}: y must be in {{0, 1}} or {{-1, 1}}; "
                             f"got values {sorted(labels)}")
        if len(labels) < 2:
            raise ValueError(f"sample {name!r} has a single class ({sorted(labels)}); ROC and "
                             f"calibration need both classes in every sample")
        checked[name] = (X, (y == 1).astype(int))
    return checked


def model_section(points, intercept, names, outcome_name, model_type, X_train, scores):
    """Items, score range, score-to-risk row and (for checklists) M and the rule."""
    items = []
    value_sets = []
    for j in np.flatnonzero(points):
        values = np.unique(X_train[:, j])
        vmin, vmax = float(values.min()), float(values.max())
        binary = bool(set(values.tolist()) <= {0.0, 1.0})
        if binary:
            vmin, vmax, values = 0.0, 1.0, np.array([0.0, 1.0])
        p = float(points[j])
        value_sets.append(p * values)
        items.append({"name": str(names[j]), "points": number(p), "binary": binary,
                      "value_range": [number(vmin), number(vmax)]})
    # order items as print_model does: most positive points first
    items.sort(key=lambda item: -item["points"])

    lo = sum(float(v.min()) for v in value_sets)
    hi = sum(float(v.max()) for v in value_sets)
    totals = achievable_totals(value_sets)
    if totals is None:
        totals = np.unique(np.concatenate(list(scores.values())))
    totals = [number(s) for s in totals]
    model = {
        "type": model_type,
        "intercept": number(intercept),
        "items": items,
        "score_range": [number(lo), number(hi)],
        "score_to_risk": {"scores": totals,
                          "risk": [float(expit(s + intercept)) for s in totals]},
        "checklist_m": None,
        "rule": None,
    }
    if model_type == "checklist":
        m = math.floor(-intercept) + 1
        n_items = len(items)
        n_negative = sum(item["points"] < 0 for item in items)
        k_min, k_max = -n_negative, n_items - n_negative
        if m <= k_min:
            rule = f"Predict {outcome_name} for every row (the intercept alone is positive)"
        elif m > k_max:
            rule = f"Never predict {outcome_name} (no set of checked items reaches the threshold)"
        elif n_negative == 0:
            rule = f"Predict {outcome_name} if at least {m} of {n_items} items are checked"
        else:
            rule = (f"Predict {outcome_name} if the number of checked (+) items minus the number "
                    f"of checked (−) items is at least {m}")
        model["checklist_m"] = m
        model["rule"] = rule
    return model


def achievable_totals(value_sets):
    """Every sum of one value per set, when all values are integers; else None."""
    totals = {0}
    for values in value_sets:
        if not is_integer(values) or len(totals) * len(values) > MAX_TOTALS:
            return None
        totals = {t + int(v) for t in totals for v in values}
    return sorted(totals)


def roc_section(y, score):
    """FPR/TPR at each score threshold (predict positive if score >= threshold) and AUC."""
    fpr, tpr, thresholds = roc_curve(y, score, drop_intermediate=False)
    return {
        "fpr": [float(v) for v in fpr],
        "tpr": [float(v) for v in tpr],
        "thresholds": [None if not np.isfinite(t) else number(t) for t in thresholds],
        "auc": float(auc(fpr, tpr)),
    }


def calibration_section(y, score, intercept):
    """Per observed score: predicted risk, observed rate and n; n-weighted calibration error."""
    bins, inverse, n = np.unique(score, return_inverse=True, return_counts=True)
    positives = np.bincount(inverse, weights=y, minlength=len(bins))
    predicted = expit(bins + intercept)
    observed = positives / n
    return {
        "scores": [number(s) for s in bins],
        "predicted": [float(v) for v in predicted],
        "observed": [float(v) for v in observed],
        "n": [int(v) for v in n],
        "error": float(np.sum(n * np.abs(predicted - observed)) / np.sum(n)),
    }


def summary_section(samples, model, roc, calibration, log_loss, training, constraints):
    """Four display blocks (dataset, constraints, training, performance) of formatted strings."""
    names = list(samples)
    blocks = [{
        "key": "dataset", "title": "Dataset", "columns": ["", *names],
        "rows": [["n", *[f"{len(samples[s][1]):,}" for s in names]],
                 ["outcome rate", *[f"{samples[s][1].mean():.1%}" for s in names]]],
    }]

    size = str(len(model["items"]))
    if constraints.get("max_size") is not None:
        size += f" (max {int(constraints['max_size'])})"
    rows = [["model size", size]]
    if constraints.get("point_range") is not None:
        lb, ub = constraints["point_range"]
        rows.append(["point range", f"{number(lb)} to {number(ub)}"])
    blocks.append({"key": "constraints", "title": "Constraints", "columns": ["", "value"],
                   "rows": rows})

    if training is not None:
        rows = [["objective value", fmt(training.get("objective_value"), "{:.4f}")],
                ["optimality gap", fmt(training.get("optimality_gap"), "{:.1%}")],
                ["run time", fmt(training.get("run_time"),
                               "{:.2f} s" if (training.get("run_time") or 0) < 1 else "{:.1f} s")]]
        blocks.append({"key": "training", "title": "Training", "columns": ["", "value"],
                       "rows": rows})

    blocks.append({
        "key": "performance", "title": "Performance", "columns": ["", *names],
        "rows": [["AUC", *[f"{roc[s]['auc']:.3f}" for s in names]],
                 ["calibration error", *[f"{calibration[s]['error']:.1%}" for s in names]],
                 ["log loss", *[f"{log_loss[s]:.3f}" for s in names]]],
    })
    return blocks


def number(value):
    """A JSON number: int when the value is integral, else float."""
    value = float(value)
    return int(value) if value.is_integer() else value


def fmt(value, template):
    """Format a finite number, or "n/a" for None, NaN or inf."""
    if value is None or not np.isfinite(value):
        return "n/a"
    return template.format(float(value))


# ---------------------------------------------------------------------------
# Plotly figures: plain JSON dicts the browser hands to ``Plotly.newPlot``.
# ---------------------------------------------------------------------------

def sample_colors(names):
    """One palette colour per sample, in order."""
    return [color for _, color in zip(names, cycle(SAMPLE_COLORS))]


def roc_figure(names, sections):
    """One ROC curve per sample with a point at each score threshold; AUC box top-left."""
    colors = sample_colors(names)
    traces = []
    for name, color in zip(names, colors):
        roc = sections[name]
        labels = ["none" if t is None else f"score ≥ {t}" for t in roc["thresholds"]]
        traces.append({
            "type": "scatter", "mode": "lines+markers", "name": name,
            "x": roc["fpr"], "y": roc["tpr"], "customdata": labels,
            "line": {"color": color, "width": 2}, "marker": {"color": color, "size": 6},
            "hovertemplate": "%{customdata}<br>FPR %{x:.1%} · TPR %{y:.1%}"
                             f"<extra>{name}</extra>",
        })
    metrics = [(name, f"{sections[name]['auc']:.3f}") for name in names]
    layout = figure_layout("False positive rate", "True positive rate", "AUC", metrics, colors)
    return {"data": traces, "layout": layout}


def calibration_figure(names, sections):
    """Bubbles per score, sized by n and labelled with the score; CAL box top-left."""
    n_max = max(n for name in names for n in sections[name]["n"])
    d_min, d_max = BUBBLE_PX
    colors = sample_colors(names)
    traces = []
    for name, color in zip(names, colors):
        cal = sections[name]
        traces.append({
            "type": "scatter", "mode": "markers+text", "name": name,
            "x": cal["predicted"], "y": cal["observed"],
            "text": [str(s) for s in cal["scores"]], "customdata": [[n] for n in cal["n"]],
            "textposition": "middle center",
            "textfont": {"size": 9, "color": "var(--rs-bg)"},
            "cliponaxis": False,
            "marker": {"color": color, "opacity": 0.85, "line": {"color": "var(--rs-bg)", "width": 1},
                       "size": [round(d_min + (d_max - d_min) * math.sqrt(n / n_max), 2)
                                for n in cal["n"]]},
            "hovertemplate": "score %{text}<br>predicted risk %{x:.1%}<br>"
                             "observed risk %{y:.1%}<br>n = %{customdata[0]:,}"
                             f"<extra>{name}</extra>",
        })
    metrics = [(name, f"{sections[name]['error']:.1%}") for name in names]
    layout = figure_layout("Predicted risk", "Observed risk", "CAL", metrics, colors,
                           axis_overrides={"range": [-0.03, 1.03], "tickformat": ".0%"})
    return {"data": traces, "layout": layout}


def figure_layout(x_title, y_title, box_label, metrics, colors, axis_overrides=None):
    """Shared axes, fonts, diagonal and the top-left metrics box."""
    axis = {
        "showgrid": True, "gridcolor": "var(--rs-grid)", "zeroline": False, "showline": True,
        "linecolor": "var(--rs-ink)", "ticks": "outside", "ticklen": 3, "fixedrange": True,
        "tickfont": {"size": 11, "color": "var(--rs-muted)"},
        "title": {"font": {"size": 12, "color": "var(--rs-ink)"}},
        "range": [-0.02, 1.02], "dtick": 0.2,
        **(axis_overrides or {}),
    }
    lines = [f"<b>{box_label}</b>"] + [
        f'<span style="color:{color}">{name}</span> {value}'
        for (name, value), color in zip(metrics, colors)
    ]
    return {
        "autosize": True,
        "margin": {"t": 36, "r": 12, "b": 48, "l": 56},
        "font": {"family": "var(--pico-font-family)", "size": 11, "color": "var(--rs-ink)"},
        "paper_bgcolor": "var(--rs-bg)", "plot_bgcolor": "var(--rs-bg)",
        "xaxis": {**axis, "title": {**axis["title"], "text": x_title}},
        "yaxis": {**axis, "title": {**axis["title"], "text": y_title}},
        "hovermode": "closest", "dragmode": False,
        "legend": {"orientation": "h", "x": 1, "xanchor": "right", "y": 1.01,
                   "yanchor": "bottom"},
        "shapes": [{"type": "line", "x0": 0, "y0": 0, "x1": 1, "y1": 1, "layer": "below",
                    "line": {"color": "var(--rs-muted)", "width": 1, "dash": "dash"}}],
        "annotations": [{
            "xref": "paper", "yref": "paper", "x": 0.02, "y": 0.98,
            "xanchor": "left", "yanchor": "top", "align": "left", "showarrow": False,
            "text": "<br>".join(lines), "bgcolor": "var(--rs-stripe)",
            "bordercolor": "var(--rs-grid)", "borderpad": 4,
        }],
    }


# ---------------------------------------------------------------------------
# The page: a Jinja shell with the inlined CSS/JS and one JSON data block.
# ---------------------------------------------------------------------------

@cache
def load_assets():
    """The compiled page template and the CSS/JS it inlines, read once per process."""
    template = (ASSETS / "template.html").read_text(encoding="utf-8")
    shell = Environment(autoescape=True).from_string(template)
    styles = Markup((ASSETS / "styles.css").read_text(encoding="utf-8"))
    scripts = Markup("\n".join((ASSETS / name).read_text(encoding="utf-8") for name in SCRIPTS))
    return shell, styles, scripts


def json_for_script(data):
    """JSON safe inside ``<script type="application/json">``: ``</`` becomes ``<\\/``."""
    text = json.dumps(data, allow_nan=False, ensure_ascii=False)
    return text.replace("</", "<\\/").replace("<!--", "<\\u0021--")

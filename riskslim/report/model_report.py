"""The report: one object that computes the page, builds its figures and renders the HTML.

``ModelReport`` takes a fitted ``RiskSLIMClassifier`` and, optionally, the
``BinaryClassificationDataset`` it was fit on (no solver runs). It computes one
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
- **Samples:** ``Training``, then ``{k}-CV`` (the out-of-fold predictions of ``k`` fold models,
  each scoring its own test rows with its own points and intercept), then the dataset's other
  splits (``Validation``, ``Test``).
- **Item names:** a rule name ``feature_op_value`` (e.g. ``ClumpThickness_geq_5``) is shown as
  ``feature symbol value`` (``ClumpThickness ≥ 5``); any other name is shown as it is.
- **Checklist M:** with k = (#checked +1 items) - (#checked -1 items), M is the smallest integer
  k with ``k + intercept > 0``, i.e. ``floor(-intercept) + 1``.
- **Score range:** ``sum(points * [min, max])`` of each item over the training data (a binary
  item contributes ``points * [0, 1]``). The score-to-risk row lists every achievable total when
  all item values are integers, and otherwise the totals observed in the samples.
- **Calibration error:** ``|predicted - observed|`` per score bin (``local_error``), and their
  n-weighted mean over the sample (``error``). Score bins with no rows in a sample are absent
  from that sample. A CV sample pools fold models, so one score can fall in several bins, one
  per fold intercept.
"""

import html
import json
import math
import warnings
from functools import cache
from importlib.resources import files
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment
from markupsafe import Markup
from scipy.special import expit
from sklearn.metrics import auc, roc_curve
from sklearn.utils.validation import check_is_fitted, validate_data

from ..data import BinaryClassificationDataset, RuleName
from ..defaults import INTERCEPT_NAME
from ..loss_functions.log_loss import log_loss_value_from_scores
from ..utils import is_integer

SCHEMA_VERSION = 1
MAX_TOTALS = 10_000  # score totals to enumerate before falling back to observed scores
MODEL_TYPES = {"risk_score": "Risk score", "checklist": "Checklist"}

# sample names, in page order: the training sample, the CV sample ("5-CV"), then the other splits
TRAINING = "Training"
SPLIT_SAMPLES = {"training": TRAINING, "validation": "Validation", "test": "Test"}

# how each rule-name operator (``feature_op_value``) is shown
OPERATOR_SYMBOLS = {"geq": "≥", "leq": "≤", "lt": "<", "gt": ">", "eq": "=", "neq": "≠",
                    "is": "=", "isnot": "≠", "in": "∈", "notin": "∉"}

ASSETS = files(__package__) / "assets"
# Inlined in this order: each component file defines its component globally, and mount_report.js
# mounts it on its slot in assets/template.html, where the slot's markup is its in-DOM template.
SCRIPTS = ("model_card.js", "summary_table.js", "roc_plot.js", "calibration_plot.js",
           "mount_report.js")

# ---------------------------------------------------------------------------
# Visuals: the plots are styled with Plotly's own mechanism, the template below, which both
# figures reference; the page is styled by the --rs-* custom properties in assets/styles.css.
# The two hold the same palette in their own idiom, and a test asserts they agree.
# ---------------------------------------------------------------------------
TEMPLATE_NAME = "riskslim"
FONT_FAMILY = '"Helvetica Neue", Helvetica, Arial, "Nimbus Sans", "Liberation Sans", sans-serif'
INK = "#1F2933"
AXIS_TEXT = "#4D4D4D"
GRID = "#DFE4E9"
BACKGROUND = "#FFFFFF"
SAMPLE_COLORS = ["#000000", "#D2B48C", "#BEBEBE"]  # one per sample, in page order
BUBBLE_PX = (8, 20)  # calibration bubble diameter for the smallest and largest n
AXIS_RANGE = [-0.02, 1.02]  # both axes of both figures: 0 to 1, with room for a marker on the edge

# Both axes of both figures: a light four-sided frame (no dark axis L), percent ticks every 20%.
AXIS = {
    "showgrid": True, "gridcolor": GRID, "zeroline": False,
    "showline": True, "linecolor": GRID, "linewidth": 2, "mirror": True,
    "ticks": "outside", "ticklen": 3, "tickcolor": GRID,
    "fixedrange": True, "range": AXIS_RANGE, "dtick": 0.2, "tickformat": ".0%",
    "tickfont": {"size": 14, "color": AXIS_TEXT},
    "title": {"font": {"size": 15, "color": INK}},
}

# Registered, not made the default: importing riskslim must not restyle anyone else's plots.
pio.templates[TEMPLATE_NAME] = go.layout.Template(layout=go.Layout(
    autosize=True,
    font={"family": FONT_FAMILY, "size": 12, "color": INK},
    paper_bgcolor=BACKGROUND,
    plot_bgcolor=BACKGROUND,
    colorway=SAMPLE_COLORS,
    # r and t leave room for the last tick label and for a bubble label overhanging the frame
    margin={"t": 24, "r": 24, "b": 64, "l": 72},
    hovermode="closest",
    dragmode=False,
    # inside the panel, bottom right: the ROC curve owns the top left and the calibration points
    # follow the diagonal, so that corner is empty in both figures
    legend={"orientation": "v", "x": 0.98, "xanchor": "right", "y": 0.02, "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0)", "borderwidth": 0, "tracegroupgap": 8,
            "font": {"size": 12, "color": INK}},
    xaxis=AXIS,
    # a square data space to go with the square panel: one unit of risk is one unit either way
    yaxis={**AXIS, "scaleanchor": "x", "scaleratio": 1, "constrain": "domain"},
))


class ModelReport:
    """An HTML report for a risk score or a checklist.

    ``report.data`` is everything the page shows, ``report.html`` is the page as a string,
    ``report.save(path)`` writes it, and notebooks display it inline in an iframe.
    ``RiskSLIMClassifier.report(...)`` builds one.

    Parameters
    ----------
    classifier : riskslim.RiskSLIMClassifier
        A fitted classifier. The report shows its coefficients, its solver statistics
        (``solution_info_``: ``objective_value``, ``optimality_gap``, ``run_time``) and its
        constraints (``max_size_``, and the point range of ``coef_set_``).
    data : riskslim.data.BinaryClassificationDataset, optional
        The dataset the model was fit on. Its feature and outcome names label the page, and when
        it has splits (``data.split(...)``) each split is a sample. Without it, or without
        splits, the training sample is the data passed to ``fit``.
    folds : list of RiskSLIMClassifier, optional
        Fitted per-fold models for the CV sample, one per fold of ``classifier.fit_cv``; each
        scores its own test rows, ``classifier.cv_results_["indices"]["test"]``. None uses
        ``cv_results_["estimator"]``; without ``fit_cv`` there is no CV sample.
    model_type : {"risk_score", "checklist"}, optional
        Inferred from the coefficients when None: a checklist when every nonzero coefficient is
        +1 or -1.
    X_test, y_test : array-like, optional
        A held-out sample, shown as ``Test``. Ignored, with a warning, when ``data`` already has
        a test split.
    """

    def __init__(self, classifier, data=None, folds=None, model_type=None, X_test=None,
                 y_test=None):
        check_is_fitted(classifier)
        dataset = checked_dataset(data, classifier)
        self.rho, self.variable_names = checked_coefficients(classifier, dataset)
        self.outcome_name = str(dataset.names.y)
        self.training = classifier.solution_info_
        self.constraints = fitted_constraints(classifier)
        self.model_type = checked_model_type(model_type, self.rho[1:])

        # samples stay local: the page needs only what they produce, and holding them would pin
        # a float64 copy of every X for the report's lifetime
        samples = split_samples(classifier, data, X_test, y_test)
        intercept, points = float(self.rho[0]), self.rho[1:]

        # {name: (y, score, intercept)}: the model scores each split; a CV row is scored by its
        # own fold model, so the CV sample carries one intercept per row
        scored = {name: (y, X @ points, intercept) for name, (X, y) in samples.items()}
        model = model_section(points, intercept, self.variable_names[1:], self.outcome_name,
                              self.model_type, samples[TRAINING][0],
                              {name: score for name, (_, score, _) in scored.items()})
        cv = cv_sample(classifier, folds)
        if cv is not None:
            cv_name, cv_scored = cv
            scored = {TRAINING: scored.pop(TRAINING), cv_name: cv_scored, **scored}
        checked_classes(scored)

        roc = {name: roc_section(y, score, b) for name, (y, score, b) in scored.items()}
        calibration = {name: calibration_section(y, score, b)
                       for name, (y, score, b) in scored.items()}
        log_loss = {name: float(log_loss_value_from_scores((2 * y - 1) * (score + b)))
                    for name, (y, score, b) in scored.items()}
        names = list(scored)
        summary = summary_section({name: y for name, (y, _, _) in scored.items()}, model,
                                  roc, calibration, log_loss, self.training, self.constraints)
        self.data = {
            "schema_version": SCHEMA_VERSION,
            "title": f"{MODEL_TYPES[self.model_type]}: {self.outcome_name}",
            "outcome_name": self.outcome_name,
            "samples": names,
            "model": model,
            "summary": summary,
            "roc": roc,
            "calibration": calibration,
        }
        labels = sample_labels(summary, names)
        self.data["figures"] = {"roc": roc_figure(names, roc, labels),
                                "calibration": calibration_figure(names, calibration, labels)}

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


def checked_dataset(data, classifier):
    """``data`` when given, checked against the fit; else the dataset ``fit`` built."""
    if data is None:
        return classifier._data
    if not isinstance(data, BinaryClassificationDataset):
        raise TypeError(f"data must be a BinaryClassificationDataset; got {type(data).__name__}")
    if data.d != classifier.n_features_in_:
        raise ValueError(f"data has {data.d} features but the model was fit on "
                         f"{classifier.n_features_in_}")
    return data


def checked_coefficients(classifier, dataset):
    """The coefficients, intercept first, and their names from the dataset."""
    rho = np.concatenate([[classifier.intercept_], classifier.coef_]).astype(float)
    if not np.all(np.isfinite(rho)):
        raise ValueError(f"rho must be finite; got {rho.tolist()}")
    return rho, [INTERCEPT_NAME, *dataset.names.X]


def fitted_constraints(classifier):
    """The model size limit and the point range over the non-intercept coefficients."""
    coef_set = classifier.coef_set_
    features = [j for j, name in enumerate(coef_set.variable_names) if name != INTERCEPT_NAME]
    return {"max_size": classifier.max_size_,
            "point_range": (float(np.min(coef_set.lb[features])),
                            float(np.max(coef_set.ub[features])))}


def checked_model_type(model_type, points):
    """Validate the model type, or infer it from the coefficients when None."""
    if model_type is None:
        return infer_model_type(points)
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {tuple(MODEL_TYPES)}; got {model_type!r}")
    return model_type


def split_samples(classifier, data, X_test, y_test):
    """``{name: (X, y in {0, 1})}``: the splits of ``data``, else the data passed to fit, plus
    ``X_test`` / ``y_test`` as ``Test`` when ``data`` has no test split."""
    if (X_test is None) != (y_test is None):
        raise ValueError("report() needs both X_test and y_test, or neither")
    if data is None or data.splits is None:
        fit_data = classifier._data
        samples = {TRAINING: (fit_data.X, (fit_data.y == fit_data.classes[1]).astype(int))}
    else:
        splits = vars(data.splits)
        samples = {}
        for split_name, sample_name in SPLIT_SAMPLES.items():
            if split_name in splits:
                # each access rebuilds the array from the parent's DataFrame: read once
                X, y = splits[split_name].X, splits[split_name].y
                samples[sample_name] = (X, (y == data.classes[1]).astype(int))
    test = SPLIT_SAMPLES["test"]
    if X_test is None:
        return samples
    if test in samples:
        warnings.warn("report() uses the test split of data; X_test and y_test are ignored",
                      UserWarning, stacklevel=4)
        return samples
    X_test, y_test = validate_data(classifier, X_test, y_test, dtype=np.float64, reset=False)
    if not np.isin(y_test, classifier.classes_).all():
        raise ValueError(f"y_test has labels outside the classes seen in fit "
                         f"{classifier.classes_.tolist()}")
    samples[test] = (X_test, (y_test == classifier.classes_[1]).astype(int))
    return samples


def cv_sample(classifier, folds):
    """``(name, (y, score, intercept))`` of the fold models' out-of-fold predictions, or None.

    The fold models are ``folds``, else ``cv_results_["estimator"]``. Each scores its own test
    rows of the data passed to fit, read from ``cv_results_["indices"]["test"]`` and never
    re-derived, so the report and ``fit_cv`` cannot disagree about which rows a fold held out.
    """
    cv_results = getattr(classifier, "cv_results_", None)
    if folds is None:
        if cv_results is None:
            return None
        folds = list(cv_results["estimator"])
    else:
        folds = list(folds)
        for fold_model in folds:
            check_is_fitted(fold_model)
            if fold_model.n_features_in_ != classifier.n_features_in_:
                raise ValueError(f"a fold model was fit on {fold_model.n_features_in_} features; "
                                 f"this model on {classifier.n_features_in_}")
    if not folds:
        return None

    fit_data = classifier._data
    test_rows = [] if cv_results is None else [np.asarray(rows) for rows in
                                               cv_results["indices"]["test"]]
    if len(test_rows) != len(folds) or any(rows.max(initial=-1) >= fit_data.n
                                           for rows in test_rows):
        warnings.warn(f"report() shows no CV sample: {len(folds)} fold models need their test "
                      f"rows from fit_cv on the data passed to fit, and cv_results_ has "
                      f"{len(test_rows)} folds", UserWarning, stacklevel=4)
        return None
    X, y = fit_data.X, (fit_data.y == fit_data.classes[1]).astype(int)
    score = np.concatenate([X[rows] @ fold.coef_ for fold, rows in zip(folds, test_rows)])
    intercept = np.concatenate([np.full(len(rows), float(fold.intercept_))
                                for fold, rows in zip(folds, test_rows)])
    return f"{len(folds)}-CV", (y[np.concatenate(test_rows)], score, intercept)


def checked_classes(scored):
    """Every sample holds both classes: ROC and calibration need them."""
    for name, (y, _, _) in scored.items():
        labels = sorted(np.unique(y).tolist())
        if len(labels) < 2:
            raise ValueError(f"sample {name!r} has a single class ({labels}); ROC and "
                             f"calibration need both classes in every sample")


def display_name(name):
    """``feature symbol value`` for a rule name ``feature_op_value``; else the name as it is."""
    try:
        feature, operator, value = RuleName.parse(name)
    except ValueError:
        return name
    return f"{feature} {OPERATOR_SYMBOLS[operator]} {value}"


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
        items.append({"name": display_name(str(names[j])), "points": number(p), "binary": binary,
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


def roc_section(y, score, intercept):
    """FPR/TPR at each risk threshold (predict positive if risk >= threshold) and AUC.

    Rows rank by ``score + intercept``. Each threshold is named by the score of its rows; the
    first (no row predicted positive) is None, as is a CV threshold whose rows, scored by fold
    models with different intercepts, have different scores.
    """
    margin = score + intercept
    fpr, tpr, thresholds = roc_curve(y, margin, drop_intermediate=False)
    scores_at = {}
    for m, s in zip(margin.tolist(), score.tolist()):
        scores_at.setdefault(m, set()).add(s)
    return {
        "fpr": [float(v) for v in fpr],
        "tpr": [float(v) for v in tpr],
        "thresholds": [number(next(iter(scores_at[t])))
                       if np.isfinite(t) and len(scores_at[t]) == 1 else None
                       for t in thresholds.tolist()],
        "auc": float(auc(fpr, tpr)),
    }


def calibration_section(y, score, intercept):
    """Per score bin: predicted risk, observed rate, n and local error; and the sample's error.

    A bin is the rows sharing a score and a risk: one bin per score for one model, one per
    score and fold intercept for a CV sample. ``local_error`` is that bin's
    ``|predicted - observed|``; ``error`` is their n-weighted mean.
    """
    margin = np.broadcast_to(score + intercept, score.shape)
    bins, inverse, n = np.unique(np.column_stack([score, margin]), axis=0,
                                 return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    positives = np.bincount(inverse, weights=y, minlength=len(bins))
    predicted = expit(bins[:, 1])
    observed = positives / n
    return {
        "scores": [number(s) for s in bins[:, 0]],
        "predicted": [float(v) for v in predicted],
        "observed": [float(v) for v in observed],
        "n": [int(v) for v in n],
        "local_error": [float(v) for v in np.abs(predicted - observed)],
        "error": float(np.sum(n * np.abs(predicted - observed)) / np.sum(n)),
    }


def summary_section(labels, model, roc, calibration, log_loss, training, constraints):
    """Four display blocks (dataset, constraints, training, performance) of formatted strings.

    ``labels`` is ``{sample name: y in {0, 1}}``, in column order.
    """
    names = list(labels)
    blocks = [{
        "key": "dataset", "title": "Dataset", "columns": ["", *names],
        "rows": [["n", *[f"{len(labels[s]):,}" for s in names]],
                 ["outcome rate", *[f"{labels[s].mean():.1%}" for s in names]]],
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
# Plotly figures: built as ``go.Figure`` objects, so every property is validated here, and
# serialized to the plain JSON the data block carries to ``Plotly.newPlot``.
# ---------------------------------------------------------------------------

def sample_labels(summary, names):
    """``{sample: "Training<br>(n = 8,815 p = 12.4%)"}``: the legend's first two lines.

    n and the outcome rate are the strings the summary's dataset block already shows, so the
    legend and the table cannot round the same number two ways.
    """
    (dataset,) = [block for block in summary if block["key"] == "dataset"]
    counts, rates = dataset["rows"]
    return {name: f"{name}<br>(n = {n} p = {p})"
            for name, n, p in zip(names, counts[1:], rates[1:])}


def roc_figure(names, sections, labels):
    """One ROC curve per sample with a point at each score threshold; AUC in the legend."""
    traces = []
    for name in names:
        roc = sections[name]
        thresholds = ["none" if i == 0 else "scores differ by fold" if t is None else f"score ≥ {t}"
                      for i, t in enumerate(roc["thresholds"])]
        traces.append(go.Scatter(
            mode="lines+markers", name=f"{labels[name]}<br>AUC = {roc['auc']:.3f}",
            x=roc["fpr"], y=roc["tpr"], customdata=thresholds,
            line={"width": 2}, marker={"size": 12},
            hovertemplate="%{customdata}<br>FPR %{x:.1%} · TPR %{y:.1%}"
                          f"<extra>{name}</extra>",
        ))
    return go.Figure(traces, figure_layout("False positive rate",
                                           "True positive rate")).to_plotly_json()


def calibration_figure(names, sections, labels):
    """Bubbles per score, sized by n and labelled with the score; ECE in the legend."""
    n_max = max(n for name in names for n in sections[name]["n"])
    d_min, d_max = BUBBLE_PX
    traces = []
    for name in names:
        cal = sections[name]
        traces.append(go.Scatter(
            mode="markers+text", name=f"{labels[name]}<br>ECE = {cal['error']:.1%}",
            x=cal["predicted"], y=cal["observed"],
            text=[str(s) for s in cal["scores"]],
            customdata=[[n, e] for n, e in zip(cal["n"], cal["local_error"])],
            textposition="top center",
            textfont={"size": 11, "color": INK},
            cliponaxis=False,
            marker={"opacity": 0.85, "line": {"color": BACKGROUND, "width": 1},
                    "size": [round(d_min + (d_max - d_min) * math.sqrt(n / n_max), 2)
                             for n in cal["n"]]},
            hovertemplate="score %{text}<br>predicted risk %{x:.1%}<br>"
                          "observed risk %{y:.1%}<br>calibration error %{customdata[1]:.1%}<br>"
                          "n = %{customdata[0]:,}"
                          f"<extra>{name}</extra>",
        ))
    return go.Figure(traces, figure_layout("Predicted risk", "Observed risk")).to_plotly_json()


def figure_layout(x_title, y_title):
    """What one figure adds to the template: its axis titles and the diagonal.

    The diagonal is drawn in the grid colour, deliberately lighter than the data that crosses it.
    """
    return go.Layout(
        template=TEMPLATE_NAME,
        xaxis={"title": {"text": x_title}},
        yaxis={"title": {"text": y_title}},
        shapes=[go.layout.Shape(type="line", x0=0, y0=0, x1=1, y1=1, layer="below",
                                line={"color": GRID, "width": 1, "dash": "dash"})],
    )


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

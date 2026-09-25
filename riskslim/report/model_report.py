"""The report: one object that computes the page, builds its figures and renders the HTML.

``ModelReport`` takes a fitted ``RiskSLIMClassifier`` and, optionally, the
``BinaryClassificationDataset`` it was fit on (no solver runs). It computes one
JSON-serializable data block with no NaN or inf, builds the two Plotly figures from it, and
renders a single self-contained page (libraries load from a pinned CDN). The browser only
displays what Python computed.

Conventions
-----------
- ``weights`` and ``variable_names`` follow riskslim's convention and **include the intercept** at
  index 0, named ``"(Intercept)"``. ``X`` in each sample **excludes** the intercept column (it is
  the matrix passed to ``fit``).
- The **score** of a row is ``X @ weights[1:]`` (points only, no intercept). Its risk is
  ``1 / (1 + exp(-(score + intercept)))``.
- A row is predicted positive when ``score + intercept > 0`` (risk strictly above 0.5), the same
  rule as ``RiskSLIMClassifier.predict``.
- **Samples:** ``Training``, then ``{k}-CV`` (the out-of-fold predictions of ``k`` fold models,
  each scoring its own test rows with its own points and intercept), then the dataset's other
  splits (``Validation``, ``Test``). The ``samples`` argument names them by stable keys:
  ``"training"``, ``"cv"``, ``"validation"``, ``"test"``.
- **Item names:** a rule name ``feature_op_value`` (e.g. ``ClumpThickness_geq_5``) is shown as
  ``feature symbol value`` (``ClumpThickness ≥ 5``); any other name is shown as it is.
- **Checklist M:** with k = (#checked +1 items) - (#checked -1 items), M is the smallest integer
  k with ``k + intercept > 0``, i.e. ``floor(-intercept) + 1``.
- **Score function:** ``points @ x``, the linear function of a row's item values ``x``; the
  **score** is its value on one row.
- **Score range:** ``sum(points * [min, max])`` of each item over the training data (a binary
  item contributes ``points * [0, 1]``).
- **Score type:** ``discrete`` when every nonzero item is binary on the training data and every
  point is an integer: the score function then takes a few integer values, every one of which
  the score-to-risk strip lists (``all_scores``: model-derived, whatever the sample).
  Otherwise ``continuous``: its rows are binned by predicted risk into ``max_scores_printed``
  risk bins (``risk_bins``, the one place that chooses them), and the strip, the calibration
  points and the ECE are per non-empty bin.
- **Risk bins (continuous):** a sample's rows fall in the bins of their own predicted risk; a
  bin's predicted risk is its rows' mean, its observed risk their outcome rate, and it is
  labelled by the range of its rows' scores (``3.2 to 5.1``, as the R's ``print.score.column``).
  A bin with no rows in a sample is absent from that sample; the strip holds the training
  sample's non-empty bins. Risk is monotone in the score, so each bin is a score interval: the
  Model card finds a score's bin by comparing it with the bins' score edges
  (``logit(risk edge) - intercept``), which Python ships.
- **Collapsed endpoints (discrete):** ``logit(k) ≈ logit(k + 1)`` once the risk is near 0 or 1,
  so the scores whose risk falls outside the printed risk range collapse into one cell of the
  score-to-risk strip and one point of the calibration plot at each end, as they do in the R
  report (``tail_groups``). Risk bins never collapse. The printed range is
  ``low_risk_threshold``..``high_risk_threshold``
  (1%..99% by default); a discrete strip that would still print more than
  ``max_scores_printed`` cells narrows it for this model to the risks of the lowest and highest
  scores it still prints on their own (``collapse_scores``). The tails read ``< {min}`` and
  ``> {max}`` of the printed range; the strip, the calibration points and the Model card all
  use it, and ``data["settings"]`` records it (``min_printed_risk``, ``max_printed_risk``) next
  to the requested thresholds.
  Display only: every reported number is computed per score bin, before any collapsing.
- **Calibration error:** ``|predicted - observed|`` per score bin (``local_error``), and the
  sample's ``ece``: the R's ``avg_cal_err_distinct`` for a discrete model, its
  ``avg_cal_err_binned`` over the risk bins for a continuous one. Score bins with no rows in a
  sample are absent from that sample. A CV sample pools fold models, so one score can fall in
  several discrete bins, one per fold intercept, and each CV row falls in the risk bin of its
  own fold model's risk.
"""

import html
import json
import math
import numbers
import warnings
from functools import cache
from importlib.resources import files
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment
from markupsafe import Markup
from scipy.special import expit, logit
from sklearn.metrics import auc, roc_curve
from sklearn.utils.validation import check_is_fitted, validate_data

from ..data import BinaryClassificationDataset, RuleName
from ..defaults import INTERCEPT_NAME
from ..loss_functions.log_loss import log_loss_value_from_scores
from ..utils import is_integer

SCHEMA_VERSION = 2
MODEL_TYPES = {"risk_score": "Risk Score", "checklist": "Checklist"}

# the page's cards, in their default order
COMPONENTS = ("model", "summary", "roc", "calibration")
# risks below the low threshold (above the high one) collapse into one strip cell and one
# calibration point: the R's defaults
LOW_RISK_THRESHOLD = 0.01
HIGH_RISK_THRESHOLD = 0.99
# the most cells a discrete score-to-risk strip prints, and a continuous model's number of risk
# bins: at 1%..99% a discrete strip holds at most 12 (the logit band, 9.2 wide, holds at most 10
# integer scores, plus the two tails), so the cap only folds scores when a caller has widened the
# thresholds
MAX_SCORES_PRINTED = 12

# sample names, in page order: the training sample, the CV sample ("5-CV"), then the other splits;
# keyed by the stable names the samples argument takes (the CV sample's key is "cv")
TRAINING = "Training"
SPLIT_SAMPLES = {"training": TRAINING, "validation": "Validation", "test": "Test"}

# how each rule-name operator (``feature_op_value``) is shown
OPERATOR_SYMBOLS = {"geq": "≥", "leq": "≤", "lt": "<", "gt": ">", "eq": "=", "neq": "≠",
                    "is": "=", "isnot": "≠", "in": "∈", "notin": "∉"}

ASSETS = files(__package__) / "assets"
# Inlined after the markup, which Jinja writes in full: the script only draws the figures and
# wires the Model card's inputs.
SCRIPTS = ("report.js",)

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
MUTED = "#98A2AD"
# each sample's colour, keyed as the samples argument names them, so a sample keeps its colour
# whichever samples are shown and in whatever order. Validation has no house colour: it borrows
# the palette's muted grey as a placeholder.
SAMPLE_COLORS = {"training": "#000000", "test": "#D2B48C", "cv": "#BEBEBE", "validation": MUTED}
CIRCLE_PX = 24  # calibration circle diameter: room for the widest label ("10+", "≤-3") at LABEL_PX
LABEL_PX = 10
AXIS_RANGE = [-0.02, 1.02]  # both axes of both figures: 0 to 1, with room for a marker on the edge

# Both axes of both figures: a light four-sided frame (no dark axis L), percent ticks every 20%.
AXIS = {
    "showgrid": True, "gridcolor": GRID, "zeroline": False,
    "showline": True, "linecolor": GRID, "linewidth": 2, "mirror": True,
    "ticks": "outside", "ticklen": 3, "tickcolor": GRID,
    "fixedrange": True, "range": AXIS_RANGE, "dtick": 0.2, "tickformat": ".0%",
    # under the data: a circle on the edge overhangs the frame instead of being cut by it
    "layer": "below traces",
    "tickfont": {"size": 14, "color": AXIS_TEXT},
    "title": {"font": {"size": 15, "color": INK}},
}

# Registered, not made the default: importing riskslim must not restyle anyone else's plots.
pio.templates[TEMPLATE_NAME] = go.layout.Template(layout=go.Layout(
    autosize=True,
    font={"family": FONT_FAMILY, "size": 12, "color": INK},
    paper_bgcolor=BACKGROUND,
    plot_bgcolor=BACKGROUND,
    # r and t leave room for the last tick label and for a circle overhanging the frame
    margin={"t": 24, "r": 24, "b": 64, "l": 72},
    hovermode="closest",
    # the hover box names its sample on its first line, inside the box: Plotly's side tag drew it
    # in the trace colour, unreadable for the grey and tan samples
    hoverlabel={"align": "left"},
    dragmode=False,
    # inside the panel, bottom right: the ROC curve owns the top left and the calibration points
    # follow the diagonal, so that corner is empty in both figures
    # traces run last sample first, so the first (Training) is drawn on top, as in the R; the
    # legend reverses them back into page order
    legend={"orientation": "v", "traceorder": "reversed", "x": 0.98, "xanchor": "right", "y": 0.02, "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0)", "borderwidth": 0, "tracegroupgap": 8,
            "font": {"size": 12, "color": INK}},
    xaxis=AXIS,
    # a square data space to go with the square panel: one unit of risk is one unit either way
    yaxis={**AXIS, "scaleanchor": "x", "scaleratio": 1, "constrain": "domain"},
))

# A plot narrower than NARROW_PX (a phone, or a two-column page on a small screen) has no room for
# the legend inside the panel: report.js applies these overrides to the figure there, and removes
# them again when the plot is wider. The legend moves under the plot, the margins and axis type
# tighten so the panel keeps its size, and the calibration circles shrink; report.js then grows
# the figure by the legend's height, so the panel stays square.
NARROW_PX = 480
NARROW_LAYOUT = go.Layout(
    legend={"orientation": "h", "xref": "paper", "x": 0, "xanchor": "left",
            "yref": "container", "y": 0, "yanchor": "bottom"},
    margin={"t": 24, "r": 16, "b": 56, "l": 56},
    xaxis={"tickfont": {"size": 12}, "title": {"font": {"size": 13}}},
    yaxis={"tickfont": {"size": 12}, "title": {"font": {"size": 13}}},
)
NARROW_CIRCLE_PX = 20


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
    cv_models : list of RiskSLIMClassifier, optional
        Fitted per-fold models for the CV sample, one per fold of ``classifier.fit_cv``; each
        scores its own test rows, ``classifier.cv_results_["indices"]["test"]``. None uses
        ``cv_results_["estimator"]``; without ``fit_cv`` there is no CV sample.
    model_type : {"risk_score", "checklist"}, optional
        Inferred when None: a checklist when every nonzero coefficient is +1 or -1 and its item
        is binary on the training data. A checklist needs binary items: an explicit
        ``"checklist"`` with a non-binary item raises ValueError.
    X_test, y_test : array-like, optional
        A held-out sample, shown as ``Test``. Ignored, with a warning, when ``data`` already has
        a test split.
    components : sequence of {"model", "summary", "roc", "calibration"}, optional
        The cards on the page, in this order; a component left out is not shown.
    samples : sequence of {"training", "cv", "validation", "test"}, optional
        The samples shown in the summary, ROC and calibration, in this order. None shows every
        available sample. The Model card reads its value ranges from the training rows either way.
    low_risk_threshold, high_risk_threshold : float, optional
        Risks below the first (above the second) collapse into one ``< x%`` (``> y%``) cell of
        the score-to-risk strip and one calibration point. The R's defaults, 0.01 and 0.99.
    max_scores_printed : int, optional
        The most cells a discrete model's score-to-risk strip prints, at least 2 (the two
        tails). When the thresholds leave more, the most extreme score printed on its own (risk
        nearest 0 or 1) folds into its tail, one at a time, until the strip fits; each tail then
        reads the risk of the nearest score still printed. 12 by default, which the default
        thresholds never exceed. A continuous model bins its rows into this many equal-width
        risk bins instead (``risk_bins``), and prints the bins that hold training rows.
    """

    def __init__(self, classifier, data=None, cv_models=None, model_type=None, X_test=None,
                 y_test=None, *, components=COMPONENTS, samples=None,
                 low_risk_threshold=LOW_RISK_THRESHOLD, high_risk_threshold=HIGH_RISK_THRESHOLD,
                 max_scores_printed=MAX_SCORES_PRINTED):
        check_is_fitted(classifier)
        components = checked_selection("components", components, COMPONENTS)
        low_risk, high_risk = checked_risk_thresholds(low_risk_threshold, high_risk_threshold)
        max_scores_printed = checked_max_scores_printed(max_scores_printed)
        dataset = checked_dataset(data, classifier)
        self.weights, self.variable_names = checked_coefficients(classifier, dataset)
        self.outcome_name = str(dataset.names.y)
        self.training = classifier.solution_info_
        self.constraints = fitted_constraints(classifier)

        # the splits stay local: the page needs only what they produce, and holding them would pin
        # a float64 copy of every X for the report's lifetime
        splits = split_samples(classifier, data, X_test, y_test)
        intercept, points = float(self.weights[0]), self.weights[1:]
        self.model_type = checked_model_type(model_type, points, splits[TRAINING][0],
                                             self.variable_names[1:])
        # a continuous model's risk bins, as edges on the risk scale; None for a discrete model
        bin_edges = (None if infer_score_type(points, splits[TRAINING][0]) == "discrete"
                     else risk_bins(max_scores_printed))

        # {name: (y, score, intercept)}: the model scores each split; a CV row is scored by its
        # own fold model, so the CV sample carries one intercept per row
        scored = {name: (y, X @ points, intercept) for name, (X, y) in splits.items()}
        # the Model card reads the training rows only: a sample left off the page must not
        # change it. printed_risks is (min_printed_risk, max_printed_risk): the thresholds, or
        # narrower
        model, digits, printed_risks = model_section(
            points, intercept, self.variable_names[1:], self.outcome_name, self.model_type,
            splits[TRAINING][0], scored[TRAINING][1], low_risk, high_risk, max_scores_printed,
            bin_edges)
        # {key: sample name} of every available sample, in page order: Training, the CV sample,
        # then the other splits
        available = {key: name for key, name in SPLIT_SAMPLES.items() if name in scored}
        cv = cv_sample(classifier, cv_models)
        if cv is not None:
            cv_name, cv_scored = cv
            scored[cv_name] = cv_scored
            available = {"training": TRAINING, "cv": cv_name, **available}
        shown = (list(available) if samples is None
                 else checked_selection("samples", samples, tuple(available)))
        # {sample name: key} of the samples shown, in their order: the key picks a sample's colour
        sample_keys = {available[key]: key for key in shown}
        scored = {name: scored[name] for name in sample_keys}
        checked_classes(scored)

        roc = {name: roc_section(y, score, b) for name, (y, score, b) in scored.items()}
        calibration = {name: calibration_section(y, score, b) if bin_edges is None
                       else binned_calibration_section(y, score, b, bin_edges)
                       for name, (y, score, b) in scored.items()}
        log_loss = {name: float(log_loss_value_from_scores((2 * y - 1) * (score + b)))
                    for name, (y, score, b) in scored.items()}
        summary = summary_section({name: y for name, (y, _, _) in scored.items()}, model,
                                  roc, calibration, log_loss, self.training, self.constraints)
        self.data = {
            "schema_version": SCHEMA_VERSION,
            "title": f"{MODEL_TYPES[self.model_type]}: {self.outcome_name}",
            "outcome_name": self.outcome_name,
            "samples": list(sample_keys),
            "model": model,
            "summary": summary,
            "roc": roc,
            "calibration": calibration,
            "settings": {"components": components, "samples": shown,
                         "low_risk_threshold": low_risk, "high_risk_threshold": high_risk,
                         "min_printed_risk": printed_risks[0],
                         "max_printed_risk": printed_risks[1],
                         "max_scores_printed": max_scores_printed},
        }
        # only the figures on the page: the sections above hold their numbers either way
        self.data["figures"] = {}
        if "roc" in components:
            self.data["figures"]["roc"] = roc_figure(
                sample_keys, roc, sample_labels(summary, sample_keys, "auc"))
        if "calibration" in components:
            self.data["figures"]["calibration"] = calibration_figure(
                sample_keys, {name: calibration_points(calibration[name], printed_risks, digits)
                              if bin_edges is None else binned_calibration_points(calibration[name])
                              for name in sample_keys},
                sample_labels(summary, sample_keys, "ece"))
        self.data["narrow"] = narrow_overrides(self.data["figures"])

    @property
    def html(self):
        """The report page as a string."""
        shell, styles, scripts = load_assets()
        return shell.render(title=self.data["title"], model=self.data["model"],
                            summary=self.data["summary"],
                            components=self.data["settings"]["components"],
                            styles=styles, scripts=scripts,
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

def is_binary(values):
    """True when every value is 0 or 1."""
    return bool(np.all((values == 0) | (values == 1)))


def infer_model_type(points, X_train):
    """``checklist`` when every nonzero coefficient (intercept excluded) is +1 or -1 and its item
    is binary on the training rows; else ``risk_score``."""
    nonzero = np.flatnonzero(points)
    if (nonzero.size > 0 and np.all(np.abs(points[nonzero]) == 1)
            and all(is_binary(X_train[:, j]) for j in nonzero)):
        return "checklist"
    return "risk_score"


def infer_score_type(points, X_train):
    """``discrete`` when every nonzero item is binary on the training rows and every point is an
    integer; else ``continuous``."""
    if is_integer(points) and all(is_binary(X_train[:, j]) for j in np.flatnonzero(points)):
        return "discrete"
    return "continuous"


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
    weights = np.asarray(classifier._weights, dtype=float)
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"weights must be finite; got {weights.tolist()}")
    return weights, [INTERCEPT_NAME, *dataset.names.X]


def fitted_constraints(classifier):
    """The model size limit and the point range over the non-intercept coefficients.

    Model size counts features, never the intercept. The limit is the one the optimizer
    enforced: ``max_size_`` (``d + 1`` by default) capped at the number of penalized
    coefficients (``RiskSLIMOptimizer``), read from ``coef_set_`` because ``optimizer_`` does not
    survive pickling.
    """
    coef_set = classifier.coef_set_
    features = [j for j, name in enumerate(coef_set.variable_names) if name != INTERCEPT_NAME]
    penalized = int(np.count_nonzero(coef_set.penalized_indices()[features]))
    return {"max_size": min(int(classifier.max_size_), penalized),
            "point_range": (float(np.min(coef_set.lb[features])),
                            float(np.max(coef_set.ub[features])))}


def checked_model_type(model_type, points, X_train, names):
    """Validate the model type, or infer it from the coefficients when None."""
    if model_type is None:
        return infer_model_type(points, X_train)
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {tuple(MODEL_TYPES)}; got {model_type!r}")
    if model_type == "checklist":
        # a checklist counts checked boxes: an item with other values breaks M and the rule
        non_binary = [str(names[j]) for j in np.flatnonzero(points) if not is_binary(X_train[:, j])]
        if non_binary:
            raise ValueError(f"model_type='checklist' needs binary items (0 or 1 on the training "
                             f"data); {non_binary} take other values")
    return model_type


def checked_selection(argument, selection, choices):
    """``selection`` as a list: at least one entry, none repeated, each one of ``choices``."""
    selection = list(selection)
    if not selection:
        raise ValueError(f"{argument} must name at least one of {choices}; got none")
    unknown = [entry for entry in selection if entry not in choices]
    if unknown:
        raise ValueError(f"{argument} must be drawn from {choices}; got {unknown}")
    if len(set(selection)) < len(selection):
        raise ValueError(f"{argument} must not repeat an entry; got {selection}")
    return selection


def checked_risk_thresholds(low_risk_threshold, high_risk_threshold):
    """``(low, high)`` as floats, with ``0 <= low < high <= 1``."""
    if not 0 <= low_risk_threshold < high_risk_threshold <= 1:
        raise ValueError(f"risk thresholds must satisfy 0 <= low_risk_threshold < "
                         f"high_risk_threshold <= 1; got {low_risk_threshold} and "
                         f"{high_risk_threshold}")
    return float(low_risk_threshold), float(high_risk_threshold)


def checked_max_scores_printed(max_scores_printed):
    """``max_scores_printed`` as an int, at least 2: folding scores only moves them into the two
    tails, so a strip of two or more scores never prints fewer than 2 cells."""
    if (isinstance(max_scores_printed, bool) or not isinstance(max_scores_printed, numbers.Integral)
            or max_scores_printed < 2):
        raise ValueError(f"max_scores_printed must be an integer of at least 2 (the strip's two "
                         f"tails); got {max_scores_printed!r}")
    return int(max_scores_printed)


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


def cv_sample(classifier, cv_models):
    """``(name, (y, score, intercept))`` of the fold models' out-of-fold predictions, or None.

    The fold models are ``cv_models``, else ``cv_results_["estimator"]``. Each scores its own test
    rows of the data passed to fit, read from ``cv_results_["indices"]["test"]`` and never
    re-derived, so the report and ``fit_cv`` cannot disagree about which rows a fold held out.
    """
    cv_results = getattr(classifier, "cv_results_", None)
    if cv_models is None:
        if cv_results is None:
            return None
        cv_models = list(cv_results["estimator"])
    else:
        cv_models = list(cv_models)
        for fold_model in cv_models:
            check_is_fitted(fold_model)
            if fold_model.n_features_in_ != classifier.n_features_in_:
                raise ValueError(f"a fold model was fit on {fold_model.n_features_in_} features; "
                                 f"this model on {classifier.n_features_in_}")
    if not cv_models:
        return None

    fit_data = classifier._data
    test_rows = [] if cv_results is None else [np.asarray(rows) for rows in
                                               cv_results["indices"]["test"]]
    if len(test_rows) != len(cv_models) or any(rows.max(initial=-1) >= fit_data.n
                                           for rows in test_rows):
        warnings.warn(f"report() shows no CV sample: {len(cv_models)} fold models need their test "
                      f"rows from fit_cv on the data passed to fit, and cv_results_ has "
                      f"{len(test_rows)} folds", UserWarning, stacklevel=4)
        return None
    X, y = fit_data.X, (fit_data.y == fit_data.classes[1]).astype(int)
    score = np.concatenate([X[rows] @ fold.coef_ for fold, rows in zip(cv_models, test_rows)])
    intercept = np.concatenate([np.full(len(rows), float(fold.intercept_))
                                for fold, rows in zip(cv_models, test_rows)])
    return f"{len(cv_models)}-CV", (y[np.concatenate(test_rows)], score, intercept)


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


def point_label(points, binary, model_type):
    """The item's cell under the Points header: a checklist marks +/-, a risk score counts."""
    if model_type == "checklist":
        return "+" if points > 0 else "−"
    return str(points) if binary else f"{points} × value"


def model_section(points, intercept, names, outcome_name, model_type, X_train, train_scores,
                  low_risk, high_risk, max_scores_printed, bin_edges):
    """Items, score type, score range, score-to-risk strip and (for checklists) M and the rule.

    ``points_header`` is the item table's points column, and None when there is no such column
    (a checklist whose items are all ``+1``: every box counts the same, so a column of ``+`` says
    nothing). ``score_header`` and ``risk_header`` label the score-to-risk strip. The Model card
    sums the points times the values of its items, finds the strip cell of that score and reads
    the risk that cell shows. ``n_scores_printed`` counts the strip's cells.

    A discrete model (``bin_edges`` None) lists every possible score (``all_scores``, counted
    by ``n_scores``) and collapses them into at most ``max_scores_printed`` cells
    (``collapse_scores``), starting from the requested thresholds ``low_risk`` and
    ``high_risk``. ``cell_by_score`` maps each score, keyed by its string (``"3"``), to
    ``{"cell", "label"}``: the index of the strip cell holding it and how the readout prints it.

    A continuous model has one cell per risk bin of ``bin_edges`` that holds training rows
    (``binned_score_to_risk_cells``), no finite list of scores (``n_scores`` is None) and no
    collapsed tails. ``score_bins`` holds the lookup: ``edges``, the score where each bin after
    the first begins, and ``cells``, each bin's strip cell (None for a bin with no training
    rows); and ``score_digits``, the decimals the readout prints a score with, by the rule of
    the bin labels (``score_range_label``): 0 when every score the card reaches is an integer
    (every item's points times each of its values), else 1.

    Returns the section, the decimals every discrete score label on the page prints
    (``score_digits``; None for a continuous model) and the printed risk range,
    ``(min_printed_risk, max_printed_risk)``: the thresholds, or narrower.
    """
    items = []
    value_sets = []
    for j in np.flatnonzero(points):
        column = X_train[:, j]
        binary = is_binary(column)
        values = np.array([0.0, 1.0]) if binary else np.unique(column)
        vmin, vmax = float(values.min()), float(values.max())
        p = float(points[j])
        value_sets.append(p * values)
        name, p, vmin, vmax = display_name(str(names[j])), number(p), number(vmin), number(vmax)
        items.append({"name": name, "points": p, "binary": binary,
                      "value_range": [vmin, vmax],
                      "name_label": name if binary else f"{name} ({vmin}–{vmax})",
                      "points_label": point_label(p, binary, model_type)})
    # order items as print_model does: most positive points first
    items.sort(key=lambda item: -item["points"])

    lo = sum(float(v.min()) for v in value_sets)
    hi = sum(float(v.max()) for v in value_sets)
    m = math.floor(-intercept) + 1 if model_type == "checklist" else None
    checklist = model_type == "checklist"
    shows_points = not checklist or any(item["points"] < 0 for item in items)
    if bin_edges is None:
        scores = all_scores(value_sets)
        risks = expit(np.asarray(scores, dtype=float) + intercept)
        groups, printed_risks = collapse_scores(risks, low_risk, high_risk, max_scores_printed)
        digits = score_digits(scores)
        cells, cell_by_score = score_to_risk_cells(scores, risks, groups, m, printed_risks, digits)
        n_scores, score_bins = len(scores), None
    else:
        printed_risks, digits, n_scores, cell_by_score = (low_risk, high_risk), None, None, None
        cells, score_bins = binned_score_to_risk_cells(train_scores, intercept, bin_edges)
        score_bins["score_digits"] = 0 if all(float(v).is_integer()
                                              for values in value_sets for v in values) else 1
    model = {
        "type": model_type,
        "score_type": "discrete" if bin_edges is None else "continuous",
        "n_scores": n_scores,
        "n_scores_printed": len(cells),
        "intercept": number(intercept),
        "items": items,
        "points_header": "Points" if shows_points else None,
        "score_header": "NET CHECKED" if checklist else "SCORE",
        "risk_header": "RISK",
        "score_range": [number(lo), number(hi)],
        "score_to_risk": cells,
        "cell_by_score": cell_by_score,
        "score_bins": score_bins,
        "checklist_m": None,
        "rule": None,
    }
    if model_type == "checklist":
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
    return model, digits, printed_risks


def all_scores(value_sets):
    """Every value of the score function, ascending: every sum of one value per item, where
    ``value_sets`` holds each item's points times its values, all integers.

    Model-derived, not read off a sample: a score no row reaches is listed all the same. The
    sums stay within the integer score range, so there are few of them.
    """
    scores = {0}
    for values in value_sets:
        scores = {s + int(v) for s in scores for v in values}
    return sorted(scores)


def collapse_scores(risks, low_risk, high_risk, max_scores_printed):
    """``(groups, (min_printed_risk, max_printed_risk))``: the strip cells of the scores whose
    ascending ``risks`` are given (``tail_groups``' groups), at most ``max_scores_printed`` (at
    least 2) of them, and the risk range printed one score per cell.

    The range starts at the thresholds, ``low_risk``..``high_risk``. While the strip prints too
    many cells, the most extreme risk printed on its own (nearest 0 or 1) folds into its tail,
    one at a time: the range then starts (ends) at the risk of the next score in, the lowest
    (highest) still printed on its own, so a tail's label (``< 26.9%``) is the risk of the
    score beside it. A tail that ends up holding a single score is printed as its own cell, as
    at the thresholds, and keeps its threshold. When every score folds (``max_scores_printed``
    of 2), no score is printed on its own and each tail reads the risk of the nearest score in
    the other: ``min_printed_risk`` then exceeds ``max_printed_risk``.
    """
    min_printed_risk, max_printed_risk = low_risk, high_risk
    groups = tail_groups(risks, (low_risk, high_risk))
    while len(groups) > max_scores_printed:
        # with more than 2 cells, some risk is printed on its own and has a risk beyond it
        printed = risks[(risks >= min_printed_risk) & (risks <= max_printed_risk)]
        if printed[0] <= 1 - printed[-1]:
            min_printed_risk = risks[risks > printed[0]][0]
        else:
            max_printed_risk = risks[risks < printed[-1]][-1]
        groups = tail_groups(risks, (min_printed_risk, max_printed_risk))
    if np.count_nonzero(risks < min_printed_risk) < 2:
        min_printed_risk = low_risk
    if np.count_nonzero(risks > max_printed_risk) < 2:
        max_printed_risk = high_risk
    return groups, (float(min_printed_risk), float(max_printed_risk))


def tail_groups(risks, printed_risks):
    """``[(first, last, side)]`` over ascending ``risks``: one group per risk, except that the
    risks below ``low`` become one group and those above ``high`` another, for
    ``printed_risks`` = ``(low, high)``. ``side`` is ``"low"`` or ``"high"`` for a collapsed
    tail, else None.

    The endpoints collapse because ``logit(k) ≈ logit(k + 1)`` once the risk is near 0 or 1: a
    wide score range otherwise ends in a run of cells all reading ``100.0%`` and a run of plot
    points stacked in the corner. The R collapses both the same way, at its
    ``lower_risk_threshold`` and ``upper_risk_threshold`` (0.01 and 0.99, the defaults here) --
    the risk row in ``get.risk.xtable`` (``dev/reference/burn-rules/reporting_utils.R:607-680``)
    and the calibration points in ``collapse.calibration.df`` (``ibid.:1310-1352``) -- and a tail
    holding a single risk is left alone by both.
    """
    risks = np.asarray(risks, dtype=float)
    n = len(risks)
    low_risk, high_risk = printed_risks
    low = int(np.searchsorted(risks, low_risk))  # risks are ascending, so the tails are prefixes
    # the tails never share a risk, even when low > high (``collapse_scores`` folded every score)
    high = min(n - int(np.searchsorted(risks, high_risk, side="right")), n - low)
    groups = [(i, i, None) for i in range(n)]
    if high > 1:
        groups[n - high:] = [(n - high, n - 1, "high")]
    if low > 1:
        groups[:low] = [(0, low - 1, "low")]
    return groups


def score_to_risk_cells(scores, risks, groups, checklist_m, printed_risks, digits):
    """The score-to-risk strip: ``{"score", "risk", "positive"}`` per cell, one per group of
    ``groups`` (``tail_groups``); and ``{str(score): {"cell": index, "label": score label}}``,
    the cell each score falls in.

    ``scores`` is ascending, and ``risks`` holds their risks. A collapsed cell is labelled with
    the score range it covers (``"0 to 1"``) and with the end of ``printed_risks``,
    ``(min_printed_risk, max_printed_risk)``, it lies beyond (``"< 1.0%"``), as in the R's
    ``get.risk.xtable``; every other cell shows its own score and risk.
    """
    min_printed_risk, max_printed_risk = printed_risks
    cells, cell_by_score = [], {}
    for first, last, side in groups:
        risk = (percent(risks[first]) if side is None
                else f"< {percent(min_printed_risk)}" if side == "low"
                else f"> {percent(max_printed_risk)}")
        cell_by_score.update({str(score): {"cell": len(cells),
                                           "label": score_label(score, score, digits)}
                              for score in scores[first:last + 1]})
        cells.append({"score": score_label(scores[first], scores[last], digits), "risk": risk,
                      "positive": checklist_m is not None and scores[first] >= checklist_m})
    return cells, cell_by_score


def risk_bins(n_bins):
    """The edges of ``n_bins`` equal-width risk bins, ascending: ``[0, 1/n_bins, ..., 1]``.

    The one place the report chooses a continuous model's bins: every other function takes the
    edges, so swapping this function rebins the strip, the calibration points, the ECE and the
    Model card together. These are the R's breaks, ``seq(0, n_bins - 1) / n_bins`` in
    ``get.calibration.hist.df`` (``dev/reference/burn-rules/reporting_utils.R:1219``).
    """
    return np.arange(n_bins + 1) / n_bins


def risk_bin_rows(score, risk, edges):
    """The rows grouped by risk bin, over the bins of ``edges`` that hold rows, ascending:
    ``(bins, rows, score_ranges)``. ``bins`` is each such bin's index among all of them,
    ``rows`` each row's position in ``bins``, and ``score_ranges`` each bin's ``(min, max)``
    score.

    A risk falls in the bin whose left edge is the last at or below it, as the R's
    ``findInterval``: only the interior edges decide, so a risk of 1 is in the last bin.
    """
    bins, rows = np.unique(np.searchsorted(edges[1:-1], risk, side="right"), return_inverse=True)
    rows = rows.ravel()
    score_ranges = [(float(score[rows == i].min()), float(score[rows == i].max()))
                    for i in range(len(bins))]
    return bins, rows, score_ranges


def binned_score_to_risk_cells(train_scores, intercept, edges):
    """A continuous model's score-to-risk strip: one ``{"score", "risk", "positive"}`` cell per
    risk bin of ``edges`` that holds training rows, labelled with the range of their scores
    (``score_range_label``) and their mean predicted risk, as the R's risk table of a
    continuous model; and the Model card's lookup, ``{"edges", "cells"}``.

    Risk is monotone in the score, so a bin is a score interval: ``edges`` holds the score
    where each bin after the first begins (``logit(risk edge) - intercept``), and ``cells``
    each bin's strip cell, None for a bin no training row falls in.
    """
    risk = expit(train_scores + intercept)
    bins, rows, score_ranges = risk_bin_rows(train_scores, risk, edges)
    predicted = np.bincount(rows, weights=risk) / np.bincount(rows)
    cells = [{"score": score_range_label(low, high), "risk": percent(bin_risk), "positive": False}
             for (low, high), bin_risk in zip(score_ranges, predicted)]
    cell_by_bin = [None] * (len(edges) - 1)
    for cell, bin_index in enumerate(bins):
        cell_by_bin[bin_index] = cell
    return cells, {"edges": [float(edge) for edge in logit(edges[1:-1]) - intercept],
                   "cells": cell_by_bin}


def score_range_label(low, high):
    """A risk bin's scores as the R's ``print.score.column`` prints them: ``"3.2 to 5.1"``, or
    one score when ``low == high``; no decimals when both are integers, else one."""
    digits = 0 if float(low).is_integer() and float(high).is_integer() else 1
    return score_label(low + 0.0, high + 0.0, digits)  # + 0.0: -0.0 prints as 0


def score_digits(scores):
    """The fewest decimals that print every score apart: 0 when every score is integral, else
    at least 1, and at most 6.

    ``scores`` is ascending, and rounding keeps that order, so two scores print alike only if
    two neighbours do.
    """
    digits = 0 if all(float(s).is_integer() for s in scores) else 1
    neighbours = list(zip(scores, scores[1:]))
    while digits < 6 and any(f"{a:.{digits}f}" == f"{b:.{digits}f}" for a, b in neighbours):
        digits += 1
    return digits


def score_label(low, high, digits):
    """``"3"`` for one score, ``"0 to 1"`` for a range, printed with ``digits`` decimals."""
    if low == high:
        return f"{low:.{digits}f}"
    return f"{low:.{digits}f} to {high:.{digits}f}"


def percent(risk):
    """A risk as the R prints it: ``formatC(100 * risk, format="f", digits=1)`` and ``%``."""
    return f"{100 * risk:.1f}%"


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
    """Per score bin: predicted risk, observed rate, n and local error; and the sample's ECE.

    A bin is the rows sharing a score and a risk: one bin per score for one model, one per
    score and fold intercept for a CV sample. ``local_error`` is that bin's
    ``|predicted - observed|``, the R's per-bin ``cal_err``.
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
        "ece": expected_calibration_error(y, expit(margin)),
    }


def binned_calibration_section(y, score, intercept, edges):
    """A continuous model's ``calibration_section``: per risk bin of ``edges`` that holds rows
    of this sample, the range of its scores, its mean predicted risk, observed rate, n and
    ``local_error``; and the sample's ``ece``, the R's ``avg_cal_err_binned`` over these bins
    (``compute.score.based.metrics``, ``reporting_utils.R:1039``): the n-weighted mean of
    ``local_error``.

    Each row falls in the bin of its own risk, so a CV row is binned by its own fold model.
    """
    risk = expit(score + intercept)
    _, rows, score_ranges = risk_bin_rows(score, risk, edges)
    n = np.bincount(rows)
    predicted = np.bincount(rows, weights=risk) / n
    observed = np.bincount(rows, weights=y) / n
    local_error = np.abs(predicted - observed)
    return {
        "score_ranges": [[number(low), number(high)] for low, high in score_ranges],
        "predicted": [float(v) for v in predicted],
        "observed": [float(v) for v in observed],
        "n": [int(v) for v in n],
        "local_error": [float(v) for v in local_error],
        "ece": float(np.sum(n * local_error) / len(y)),
    }


def expected_calibration_error(y, risk):
    """The R's ``avg_cal_err_distinct``: the n-weighted mean of ``|predicted - observed|`` over
    the rows grouped by distinct predicted risk (``classification_utils.R:204-226``).

    Rows are grouped by the risk they are given, not by the score that gave it, so two fold
    models that predict the same risk from different scores land in one group, as they do in the
    R. A continuous model's ECE is its sibling ``avg_cal_err_binned``
    (``binned_calibration_section``).
    """
    risk, groups, n = np.unique(risk, return_inverse=True, return_counts=True)
    observed = np.bincount(groups.ravel(), weights=y, minlength=len(risk)) / n
    return float(np.sum(n * np.abs(risk - observed)) / len(y))


def summary_row(key, label, values, span=1):
    """One row of the summary table: a stable key, its leftmost-column label, its values, and
    how many sample columns each value spans (a row that is not per-sample spans them all)."""
    return {"key": key, "label": label, "values": values, "span": span}


def summary_section(labels, model, roc, calibration, log_loss, training, constraints):
    """One flat table of formatted strings: ``columns`` (a blank label column, then the samples)
    and ``rows`` in reading order -- the data, the constraints, the solver, the performance.

    There are no block subheaders and no "value" header: the samples name the columns, and a row
    that is not per-sample simply carries one value.

    ``labels`` is ``{sample name: y in {0, 1}}``, in column order.
    """
    names = list(labels)
    rows = [summary_row("n", "N", [f"{len(labels[s]):,}" for s in names]),
            summary_row("outcome_rate", "Outcome Rate",
                        [percent(labels[s].mean()) for s in names])]

    size = str(len(model["items"]))
    if constraints.get("max_size") is not None:
        size += f" (max {int(constraints['max_size'])})"
    rows.append(summary_row("model_size", "Model Size", [size], span=len(names)))
    if constraints.get("point_range") is not None:
        lb, ub = constraints["point_range"]
        rows.append(summary_row("point_range", "Point Range", [f"{number(lb)} to {number(ub)}"],
                                span=len(names)))

    if training is not None:
        run_time = "{:.2f} s" if (training.get("run_time") or 0) < 1 else "{:.1f} s"
        rows += [
            summary_row("objective_value", "Objective Value",
                        [fmt(training.get("objective_value"), "{:.4f}")], span=len(names)),
            summary_row("optimality_gap", "Optimality Gap",
                        [fmt(training.get("optimality_gap"), "{:.1%}")], span=len(names)),
            summary_row("run_time", "Run Time", [fmt(training.get("run_time"), run_time)],
                        span=len(names)),
        ]

    rows += [summary_row("auc", "AUC", [f"{roc[s]['auc']:.3f}" for s in names]),
             summary_row("ece", "ECE", [percent(calibration[s]["ece"]) for s in names]),
             summary_row("log_loss", "Log Loss", [f"{log_loss[s]:.3f}" for s in names])]
    return {"columns": ["", *names], "rows": rows}


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

def sample_labels(summary, names, metric):
    """``{sample: "Training<br>(n = 8,815 p = 12.4%)<br>AUC = 0.950"}``: a legend entry.

    Every number is the string the summary table already shows, so the legend and the table
    cannot round the same number two ways. ``metric`` names the summary row of the figure's own
    headline statistic (``auc`` or ``ece``).
    """
    values = {row["key"]: row["values"] for row in summary["rows"]}
    label = {"auc": "AUC", "ece": "ECE"}[metric]
    return {name: f"{name}<br>(n = {n} p = {p})<br>{label} = {value}"
            for name, n, p, value in zip(names, values["n"], values["outcome_rate"],
                                         values[metric])}


def roc_figure(sample_keys, sections, labels):
    """One ROC curve per sample with a point at each score threshold; AUC in the legend.

    ``sample_keys`` is ``{sample name: key}`` in page order; the key picks the sample's colour.
    Each trace carries its sample's name in ``meta``, which report.js matches across the figures.
    The traces run in reverse page order, so the first sample is drawn on top."""
    traces = []
    for name, key in reversed(sample_keys.items()):
        roc = sections[name]
        color = SAMPLE_COLORS[key]
        thresholds = ["None" if i == 0 else "Scores Differ by Fold" if t is None
                      else f"Score ≥ {t}" for i, t in enumerate(roc["thresholds"])]
        traces.append(go.Scatter(
            mode="lines+markers", name=labels[name], meta=name,
            x=roc["fpr"], y=roc["tpr"], customdata=thresholds,
            line={"width": 2, "color": color}, marker={"size": 12, "color": color},
            hovertemplate=f"<b>{name}</b><br>%{{customdata}}<br>"
                          "False Positive Rate %{x:.1%}<br>True Positive Rate %{y:.1%}"
                          "<extra></extra>",
        ))
    return go.Figure(traces, figure_layout("False Positive Rate",
                                           "True Positive Rate")).to_plotly_json()


def calibration_points(section, printed_risks, digits):
    """The points of one sample, in risk order, in the shape of the section they come from.

    One point per score bin, except that the bins of a collapsed tail become a single point:
    the same grouping as the strip (``tail_groups`` at the strip's ``printed_risks``, over the
    bins in risk order), with the
    group's rows pooled as the R pools them in ``collapse.calibration.df`` -- n adds up, and the
    predicted and observed risk are the n-weighted means, so the point sits where its rows are.
    ``scores`` is the strip's label for the group (``"10 to 13"``), and ``labels`` the short one
    printed inside the circle: ``"10+"`` for the high tail (its lowest score), ``"≤1"`` for the
    low tail (its highest score). Scores print with ``digits`` decimals, as in the strip.

    Display only: ``ece`` and the per-bin ``local_error`` in the data block are computed before
    any of this, so a collapsed tail does not move a reported number.
    """
    order = sorted(range(len(section["n"])),
                   key=lambda i: (section["predicted"][i], section["scores"][i]))
    risks = [section["predicted"][i] for i in order]
    points = []
    for first, last, side in tail_groups(risks, printed_risks):
        rows = order[first:last + 1]
        n = sum(section["n"][i] for i in rows)
        predicted = sum(section["predicted"][i] * section["n"][i] for i in rows) / n
        observed = sum(section["observed"][i] * section["n"][i] for i in rows) / n
        scores = [section["scores"][i] for i in rows]
        low, high = min(scores), max(scores)
        scores_label = score_label(low, high, digits)
        label = (scores_label if side is None
                 else f"≤{score_label(high, high, digits)}" if side == "low"
                 else f"{score_label(low, low, digits)}+")
        points.append((label, scores_label, predicted, observed, n, abs(predicted - observed)))
    keys = ("labels", "scores", "predicted", "observed", "n", "local_error")
    return dict(zip(keys, (list(values) for values in zip(*points))))


def binned_calibration_points(section):
    """A continuous model's ``calibration_points``: one point per risk bin of the sample, in risk
    order, with no label in the circle (``labels`` is None) and the bin's score range
    (``score_range_label``) for the hover. Bins never pool."""
    return {"labels": None,
            "scores": [score_range_label(low, high) for low, high in section["score_ranges"]],
            **{key: section[key] for key in ("predicted", "observed", "n", "local_error")}}


def calibration_figure(sample_keys, points, labels):
    """Per sample, equal circles with the score inside, joined in risk order; ECE in the legend.

    ``sample_keys`` is ``{sample name: key}`` in page order; the key picks the sample's colour.
    ``points`` is each sample's ``calibration_points``, or ``binned_calibration_points`` for a
    continuous model, whose circles are plain (no label inside; the score range is in the
    hover). The circles follow the R report: one size, since n is in the hover, and a line in
    the sample's colour. Each trace carries its sample's name in ``meta``. The traces run in
    reverse page order, so the first sample is drawn on top."""
    traces = []
    for name, key in reversed(sample_keys.items()):
        cal = points[name]
        color = SAMPLE_COLORS[key]
        # the score inside the circle: white on the black training circles, black on the
        # lighter ones
        text = {} if cal["labels"] is None else {
            "text": cal["labels"], "textposition": "middle center",
            "textfont": {"size": LABEL_PX, "color": BACKGROUND if key == "training" else "#000000"}}
        traces.append(go.Scatter(
            mode="lines+markers" if cal["labels"] is None else "lines+markers+text",
            name=labels[name], meta=name,
            x=cal["predicted"], y=cal["observed"],
            customdata=[[s, n, e] for s, n, e in zip(cal["scores"], cal["n"], cal["local_error"])],
            **text,
            cliponaxis=False,
            line={"width": 2, "color": color},
            # the white outline keeps overlapping circles apart
            marker={"size": CIRCLE_PX, "color": color, "line": {"color": BACKGROUND, "width": 1}},
            hovertemplate=f"<b>{name}</b><br>Score %{{customdata[0]}}<br>"
                          "Predicted Risk %{x:.1%}<br>Observed Risk %{y:.1%}<br>"
                          "Calibration Error %{customdata[2]:.1%}<br>n = %{customdata[1]:,}"
                          "<extra></extra>",
        ))
    return go.Figure(traces, figure_layout("Predicted Risk", "Observed Risk")).to_plotly_json()


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


def narrow_overrides(figures):
    """What report.js changes in each of ``figures`` on a plot narrower than ``max_width``.

    Per figure, ``apply`` holds the ``traces`` (for ``Plotly.restyle``, every trace alike) and
    the ``layout`` (for ``Plotly.relayout``): ``NARROW_LAYOUT``, and smaller calibration circles.
    ``undo`` takes them back to the figure's own values, a layout key to None (Plotly then reads
    it from the figure and its template again). Both are Plotly attribute strings
    (``"legend.x"``), so they set just those attributes; the values are validated as Plotly
    objects here.
    """
    layout = attribute_strings(NARROW_LAYOUT.to_plotly_json())
    overrides = {}
    for key in figures:
        circles = key == "calibration"
        overrides[key] = {
            "apply": {"traces": circle_sizes(NARROW_CIRCLE_PX) if circles else {},
                      "layout": layout},
            "undo": {"traces": circle_sizes(CIRCLE_PX) if circles else {},
                     "layout": dict.fromkeys(layout)},
        }
    return {"max_width": NARROW_PX, "figures": overrides}


def circle_sizes(circle_px):
    """The calibration circles' diameter, and the label inside scaled with it."""
    return attribute_strings({
        "marker": go.scatter.Marker(size=circle_px).to_plotly_json(),
        "textfont": go.scatter.Textfont(size=LABEL_PX * circle_px / CIRCLE_PX).to_plotly_json(),
    })


def attribute_strings(spec, prefix=""):
    """``{"legend": {"x": 0}}`` as Plotly attribute strings, ``{"legend.x": 0}``."""
    flat = {}
    for key, value in spec.items():
        if isinstance(value, dict):
            flat.update(attribute_strings(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


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
    """JSON safe inside ``<script type="application/json">``: every ``<`` becomes ``\\u003c``.

    With no ``<`` left, no end tag (in any case) and no ``<!--`` can end or re-mode the block;
    ``<`` only ever appears in JSON inside a string, where the escape is valid JSON.
    """
    return json.dumps(data, allow_nan=False, ensure_ascii=False).replace("<", "\\u003c")

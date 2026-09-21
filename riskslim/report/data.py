"""Report data: everything the HTML report shows, computed once in Python.

``build_report_data`` takes plain arrays (no fitted classifier, no solver) and returns one
JSON-serializable dict with no NaN or inf. The browser only displays it.

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

import math

import numpy as np
from scipy.special import expit
from sklearn.metrics import auc, roc_curve

from ..defaults import INTERCEPT_NAME
from ..loss_functions.log_loss import log_loss_value_from_scores
from ..utils import is_integer
from .layout import default_layout, layout_to_json

SCHEMA_VERSION = 1
MODEL_TYPES = ("risk_score", "checklist")


def build_report_data(rho, variable_names, outcome_name, samples, training=None,
                      constraints=None, model_type=None):
    """Compute the report data block.

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

    Returns
    -------
    data : dict
        Sections ``schema_version``, ``title``, ``outcome_name``, ``samples``, ``model``, ``summary``, ``roc``,
        ``calibration`` and ``layout``.
    """
    rho = np.asarray(rho, dtype=float).ravel()
    variable_names = list(variable_names)
    if len(variable_names) != len(rho) or not variable_names or variable_names[0] != INTERCEPT_NAME:
        raise ValueError(
            f"rho and variable_names must have the same length with {INTERCEPT_NAME!r} first; "
            f"got {len(rho)} coefficients and names {variable_names[:3]}..."
        )
    if not np.all(np.isfinite(rho)):
        raise ValueError(f"rho must be finite; got {rho.tolist()}")
    samples = check_samples(samples, n_features=len(variable_names) - 1)

    intercept, points = float(rho[0]), rho[1:]
    if model_type is None:
        model_type = infer_model_type(points)
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}; got {model_type!r}")

    scores = {name: X @ points for name, (X, _) in samples.items()}
    model = model_section(points, intercept, variable_names[1:], outcome_name, model_type,
                          samples["train"][0], scores)
    roc = {name: roc_section(samples[name][1], scores[name]) for name in samples}
    calibration = {name: calibration_section(samples[name][1], scores[name], intercept)
                   for name in samples}
    log_loss = {name: float(log_loss_value_from_scores((2 * y - 1) * (scores[name] + intercept)))
                for name, (_, y) in samples.items()}

    return {
        "schema_version": SCHEMA_VERSION,
        "title": f"{'Checklist' if model_type == 'checklist' else 'Risk score'}: {outcome_name}",
        "outcome_name": str(outcome_name),
        "samples": list(samples),
        "model": model,
        "summary": summary_section(samples, model, roc, calibration, log_loss, training,
                                   constraints),
        "roc": roc,
        "calibration": calibration,
        "layout": layout_to_json(default_layout(model_type)),
    }


def infer_model_type(points):
    """``checklist`` when every nonzero coefficient (intercept excluded) is +1 or -1."""
    nonzero = points[points != 0]
    if nonzero.size > 0 and np.all(np.abs(nonzero) == 1):
        return "checklist"
    return "risk_score"


def check_samples(samples, n_features):
    """Validate samples and return ``{name: (X float 2d, y in {0, 1})}``, train first."""
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
    lo, hi = 0.0, 0.0
    value_sets = []
    for j in np.flatnonzero(points):
        values = np.unique(X_train[:, j])
        vmin, vmax = float(values.min()), float(values.max())
        binary = bool(set(values.tolist()) <= {0.0, 1.0})
        if binary:
            vmin, vmax, values = 0.0, 1.0, np.array([0.0, 1.0])
        p = float(points[j])
        lo += min(p * vmin, p * vmax)
        hi += max(p * vmin, p * vmax)
        value_sets.append(p * values)
        items.append({"name": str(names[j]), "points": number(p), "binary": binary,
                      "value_range": [number(vmin), number(vmax)]})
    # order items as print_model does: most positive points first
    items.sort(key=lambda item: -item["points"])

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


def achievable_totals(value_sets, max_totals=10_000):
    """Every sum of one value per set, when all values are integers; else None."""
    totals = {0}
    for values in value_sets:
        if not is_integer(values) or len(totals) * len(values) > max_totals:
            return None
        totals = {t + int(v) for t in totals for v in values}
        if len(totals) > max_totals:
            return None
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

    constraints = constraints or {}
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

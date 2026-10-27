"""Generate the cached exact answers for the synthetic training tests.

The answers come from brute-force enumeration of bounded integer score models, never from
RiskSLIM's loss or solver code. Enumeration takes about 50 minutes, so the answers are cached in
training_test_cases.pkl. The datasets are not stored: the tests regenerate them from fixed seeds
and check each one against the digest stored with its answers.

    uv run python tests/generate_training_test_cases.py --regenerate [--case CASE_ID]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pickle
from pathlib import Path

import numpy as np
from scipy.special import expit

from riskslim.coefficient_set import CoefficientSet
from riskslim.defaults import INTERCEPT_NAME


TEST_CASE_PATH = Path(__file__).resolve().parent / "training_test_cases.pkl"
REGENERATE_COMMAND = "uv run python tests/generate_training_test_cases.py --regenerate"
N_SAMPLES = 10_000
BASE_FEATURE_COUNT = 4
REDUNDANT_FEATURE_COUNT = 2 * BASE_FEATURE_COUNT
TRUTHS = {
    "integer": (-1.0, 1.0, 2.0, 3.0, 4.0),
    "fractional": (-1.0, 1.0, 2.0, 0.5, 1.5),
    "shifted_fractional": (1.5, 1.0, 2.0, 0.5, 1.5),
}
FEATURE_DISTRIBUTIONS = ("binary", "continuous", "mixed")
REDUNDANT_FEATURES = ("duplicates", "noisy_duplicates")
FEATURE_SEED = 0
LABEL_SEED = [0, 0, 1]
CASE_LABEL_SEEDS = {"binary__shifted_fractional": [1, 0, 1]}
NOISE_SEED = [0, 1, 2]
BINARY_FEATURE_PROBABILITY = 0.5
CONTINUOUS_FEATURE_STANDARD_DEVIATION = 2.0
CONTINUOUS_DECIMAL_PLACES = 10
BIT_FLIP_PROBABILITY = 0.05
GAUSSIAN_NOISE_STANDARD_DEVIATION = 0.25
MODEL_FEATURE_BOUNDS = {"risk-score": (-5, 5), "checklist": (-1, 1)}
INTERCEPT_DELTA_EXPONENT = math.expm1(1.0)
BATCH_SIZE = 256
BASELINE_QUERY_KEYS = (
    (1e-6, 4),
    (1e-6, 2),
    (1e-6, 1),
    (1e-6, 0),
    (0.01, 4),
    (0.1, 4),
    (1.0, 4),
)
REDUNDANCY_PENALTIES = (0.01, 0.1, 0.03)
REDUNDANCY_GAP_TOLERANCE = 1e-6
BASELINE_CASE_IDS = tuple(
    f"{feature_distribution}__{truth_name}"
    for feature_distribution in FEATURE_DISTRIBUTIONS
    for truth_name in TRUTHS
)
REDUNDANT_CASE_IDS = tuple(
    f"{case_id}__{redundant_features}"
    for case_id in BASELINE_CASE_IDS
    for redundant_features in REDUNDANT_FEATURES
)
ALL_CASE_IDS = BASELINE_CASE_IDS + REDUNDANT_CASE_IDS


def compute_data_digest(X, y, variable_names, outcome_name) -> str:
    """Return a stable digest of a training dataset."""
    digest = hashlib.sha256()
    for array in (X, y):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    digest.update(json.dumps(variable_names).encode("utf-8"))
    digest.update(outcome_name.encode("utf-8"))
    return digest.hexdigest()


def identity_digest(identity: dict) -> str:
    """Hash a JSON-compatible identity dictionary."""
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_dataset(case_id, X, y, variable_names, redundant_features=None) -> dict:
    """Bundle generated arrays with their names and digest."""
    return {
        "case_id": case_id,
        "X": X,
        "y": y,
        "variable_names": variable_names,
        "outcome_name": "y",
        "data_digest": compute_data_digest(X, y, variable_names, "y"),
        "redundant_features": redundant_features,
    }


def generate_data(case_id: str) -> dict:
    """Generate one fixed-seed dataset; redundant cases add copies of a baseline dataset."""
    if case_id in REDUNDANT_CASE_IDS:
        source_case_id, redundant_features = case_id.rsplit("__", 1)
        return generate_redundant_data(generate_data(source_case_id), redundant_features)

    feature_distribution, truth_name = case_id.split("__")
    truth = np.asarray(TRUTHS[truth_name], dtype=np.float64)
    feature_rng = np.random.RandomState(FEATURE_SEED)
    if feature_distribution == "mixed":
        variable_names = ("binary_1", "binary_2", "continuous_1", "continuous_2")
    else:
        variable_names = tuple(
            f"{feature_distribution}_{index}" for index in range(1, BASE_FEATURE_COUNT + 1)
        )
    if feature_distribution == "binary":
        X = feature_rng.binomial(
            1, BINARY_FEATURE_PROBABILITY, size=(N_SAMPLES, BASE_FEATURE_COUNT)
        ).astype(np.int8)
    elif feature_distribution == "continuous":
        X = np.round(
            feature_rng.normal(
                0.0, CONTINUOUS_FEATURE_STANDARD_DEVIATION, size=(N_SAMPLES, BASE_FEATURE_COUNT)
            ),
            CONTINUOUS_DECIMAL_PLACES,
        )
    else:
        half = BASE_FEATURE_COUNT // 2
        binary = feature_rng.binomial(1, BINARY_FEATURE_PROBABILITY, size=(N_SAMPLES, half))
        continuous = np.round(
            feature_rng.normal(0.0, CONTINUOUS_FEATURE_STANDARD_DEVIATION, size=(N_SAMPLES, half)),
            CONTINUOUS_DECIMAL_PLACES,
        )
        X = np.hstack((binary, continuous)).astype(np.float64)

    label_rng = np.random.RandomState(CASE_LABEL_SEEDS.get(case_id, LABEL_SEED))
    y = label_rng.binomial(1, expit(truth[0] + X @ truth[1:])).astype(np.int8)
    return build_dataset(case_id, X, y, variable_names)


def generate_redundant_data(source: dict, redundant_features: str) -> dict:
    """Append exact or noisy copies of the baseline features, keeping the baseline labels."""
    clean_X = source["X"]
    feature_distribution = source["case_id"].split("__")[0]
    if redundant_features == "duplicates":
        suffix = "duplicate"
        copied_X = clean_X.copy()
    else:
        suffix = "noisy_duplicate"
        n_binary = {"binary": BASE_FEATURE_COUNT, "continuous": 0, "mixed": BASE_FEATURE_COUNT // 2}
        all_columns = list(range(BASE_FEATURE_COUNT))
        binary_columns = all_columns[: n_binary[feature_distribution]]
        continuous_columns = all_columns[n_binary[feature_distribution] :]
        noise_rng = np.random.RandomState(NOISE_SEED)
        copied_X = clean_X.astype(np.float64, copy=True)
        if binary_columns:
            flips = noise_rng.binomial(
                1, BIT_FLIP_PROBABILITY, size=(N_SAMPLES, len(binary_columns))
            )
            copied_X[:, binary_columns] = np.abs(clean_X[:, binary_columns] - flips)
        if continuous_columns:
            copied_X[:, continuous_columns] += noise_rng.normal(
                0.0, GAUSSIAN_NOISE_STANDARD_DEVIATION, size=(N_SAMPLES, len(continuous_columns))
            )
            copied_X[:, continuous_columns] = np.round(
                copied_X[:, continuous_columns], CONTINUOUS_DECIMAL_PLACES
            )
        copied_X = copied_X.astype(clean_X.dtype)

    variable_names = source["variable_names"] + tuple(
        f"{name}_{suffix}" for name in source["variable_names"]
    )
    return build_dataset(
        f"{source['case_id']}__{redundant_features}",
        np.hstack((clean_X, copied_X)),
        source["y"].copy(),
        variable_names,
        redundant_features,
    )


def determine_coefficient_bounds(data: dict, feature_bounds: tuple[int, int]):
    """Derive the coefficient bounds, including the intercept's, with RiskSLIM's coefficient set."""
    X_with_intercept = np.column_stack([np.ones(data["X"].shape[0]), data["X"]])
    y_signed = np.where(data["y"] == 0, -1, data["y"])
    coefficient_set = CoefficientSet(
        [INTERCEPT_NAME] + list(data["variable_names"]),
        lb=feature_bounds[0],
        ub=feature_bounds[1],
        vtype="I",
        print_flag=False,
    )
    coefficient_set.update_intercept_bounds(
        X=X_with_intercept,
        y=y_signed,
        max_offset=None,
        max_L0_value=data["X"].shape[1],
    )
    return coefficient_set.lb.astype(np.float64), coefficient_set.ub.astype(np.float64)


def build_feature_coefficients(
    feature_bounds: tuple[int, int], n_features: int = BASE_FEATURE_COUNT
) -> np.ndarray:
    """Materialize every bounded integer feature vector compactly."""
    values = range(feature_bounds[0], feature_bounds[1] + 1)
    coefficients = np.fromiter(
        itertools.chain.from_iterable(itertools.product(values, repeat=n_features)), dtype=np.int8
    )
    return coefficients.reshape(-1, n_features)


def build_duplicate_group_coefficients(feature_bounds: tuple[int, int]):
    """Return four pair sums and their minimum-support eight-vector representatives."""
    lower, upper = feature_bounds
    pair_sums = build_feature_coefficients((2 * lower, 2 * upper))
    first = np.clip(pair_sums, lower, upper).astype(np.int8)
    second = (pair_sums - first).astype(np.int8)
    representatives = np.hstack((first, second))
    supports = np.count_nonzero(representatives, axis=1).astype(np.uint8)
    return pair_sums, representatives, supports


def minimum_grouped_pair_violating_supports(
    coefficients: np.ndarray, feature_bound: int
) -> np.ndarray:
    """Return minimum supports among original splits selecting both members of a pair."""
    pair_sums = coefficients[:, :BASE_FEATURE_COUNT] + coefficients[:, BASE_FEATURE_COUNT:]
    pair_supports = (coefficients[:, :BASE_FEATURE_COUNT] != 0).astype(np.uint8) + (
        coefficients[:, BASE_FEATURE_COUNT:] != 0
    ).astype(np.uint8)
    nonzero_values = [value for value in range(-feature_bound, feature_bound + 1) if value]
    sums_with_two_nonzero_terms = {
        left + right for left in nonzero_values for right in nonzero_values
    }
    can_select_both = np.isin(pair_sums, tuple(sums_with_two_nonzero_terms))
    added_support = np.where(can_select_both, 2 - pair_supports, np.inf)
    return np.count_nonzero(coefficients, axis=1) + np.min(added_support, axis=1)


def profile_integer_intercepts(scores, y, intercept_lower, intercept_upper):
    """Find every bounded integer-intercept minimum by discrete convex search."""
    n_vectors = scores.shape[1]
    lower = np.full(n_vectors, intercept_lower, dtype=np.int64)
    upper = np.full(n_vectors, intercept_upper, dtype=np.int64)
    positive_rate = float(np.mean(y))
    while np.any(lower < upper):
        active_indices = np.flatnonzero(lower < upper)
        midpoints = lower + (upper - lower) // 2
        active_midpoints = midpoints[active_indices]
        probabilities = expit(scores[:, active_indices] + active_midpoints[None, :])
        differences = (
            np.mean(np.log1p(INTERCEPT_DELTA_EXPONENT * probabilities), axis=0) - positive_rate
        )
        move_upper = differences >= 0.0
        upper[active_indices[move_upper]] = active_midpoints[move_upper]
        lower[active_indices[~move_upper]] = active_midpoints[~move_upper] + 1

    intercepts = lower.astype(np.int16)
    signed_scores = (1.0 - 2.0 * y[:, None]) * (scores + intercepts[None, :])
    losses = np.mean(np.logaddexp(0.0, signed_scores), axis=0)
    ties = np.ones(n_vectors, dtype=np.uint8)
    can_tie_right = intercepts < intercept_upper
    if np.any(can_tie_right):
        indices = np.flatnonzero(can_tie_right)
        probabilities = expit(scores[:, indices] + intercepts[indices][None, :])
        differences = (
            np.mean(np.log1p(INTERCEPT_DELTA_EXPONENT * probabilities), axis=0) - positive_rate
        )
        ties[indices[differences == 0.0]] = 2
    return intercepts, losses, ties


def enumerate_losses(X, y, coefficients, intercept_bounds):
    """Profile every feature vector in bounded-memory NumPy batches."""
    intercepts = np.empty(len(coefficients), dtype=np.int16)
    losses = np.empty(len(coefficients), dtype=np.float64)
    ties = np.empty(len(coefficients), dtype=np.uint8)
    for start in range(0, len(coefficients), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(coefficients))
        intercepts[start:stop], losses[start:stop], ties[start:stop] = profile_integer_intercepts(
            X @ coefficients[start:stop].T, y, *intercept_bounds
        )
    return intercepts, losses, ties


def select_query(
    loss_digest, coefficients, intercepts, losses, supports, c0_value, max_size
) -> dict:
    """Select one exact optimum from an enumerated loss table."""
    feasible_indices = np.flatnonzero(supports <= max_size)
    objectives = losses[feasible_indices] + c0_value * supports[feasible_indices]
    index = int(feasible_indices[int(np.argmin(objectives))])
    objective = float(objectives.min())
    return {
        "query_identity": {
            "loss_identity_digest": loss_digest,
            "c0": float(c0_value),
            "max_size": int(max_size),
        },
        "representative_weights": np.r_[intercepts[index], coefficients[index]].astype(np.int16),
        "pure_logistic_loss": float(losses[index]),
        "objective": objective,
        "representative_tie_count": int(np.count_nonzero(objectives == objective)),
        "proven_complete": True,
    }


def choose_redundancy_penalty(losses, supports, violating_supports) -> float:
    """Return the first penalty whose optima are all nonempty and use one feature per pair.

    `violating_supports` holds, per enumerated state, the smallest support of a model that selects
    both members of some pair (infinity when no such model exists).
    """
    empty_loss = float(np.min(losses[supports == 0]))
    for c0_value in REDUNDANCY_PENALTIES:
        best_objective = float(np.min(losses + c0_value * supports))
        violating_objective = float(np.min(losses + c0_value * violating_supports))
        if (
            empty_loss - best_objective > REDUNDANCY_GAP_TOLERANCE
            and violating_objective - best_objective > REDUNDANCY_GAP_TOLERANCE
        ):
            return c0_value
    raise AssertionError("no penalty yields nonempty optima that drop every redundant copy")


def enumerate_task(data: dict, model_type: str) -> dict:
    """Enumerate one model class on one dataset and answer its cached queries."""
    feature_bounds = MODEL_FEATURE_BOUNDS[model_type]
    lower_bounds, upper_bounds = determine_coefficient_bounds(data, feature_bounds)
    intercept_bounds = (math.ceil(lower_bounds[0]), math.floor(upper_bounds[0]))
    if data["redundant_features"] == "duplicates":
        # exact copies: enumerate pair sums on the clean columns, one representative per sum
        score_coefficients, coefficients, supports = build_duplicate_group_coefficients(
            feature_bounds
        )
        score_X = np.ascontiguousarray(data["X"][:, :BASE_FEATURE_COUNT])
        violating_supports = minimum_grouped_pair_violating_supports(
            coefficients, feature_bounds[1]
        )
    else:
        coefficients = build_feature_coefficients(feature_bounds, n_features=data["X"].shape[1])
        score_coefficients = coefficients
        supports = np.count_nonzero(coefficients, axis=1).astype(np.uint8)
        score_X = data["X"]
        violating_supports = None
        if data["redundant_features"] == "noisy_duplicates":
            selects_both = (coefficients[:, :BASE_FEATURE_COUNT] != 0) & (
                coefficients[:, BASE_FEATURE_COUNT:] != 0
            )
            violating_supports = np.where(np.any(selects_both, axis=1), supports, np.inf)
    intercepts, losses, _ = enumerate_losses(
        score_X, data["y"], score_coefficients, intercept_bounds
    )

    if data["redundant_features"] is None:
        query_keys = BASELINE_QUERY_KEYS
    else:
        c0_value = choose_redundancy_penalty(losses, supports, violating_supports)
        query_keys = ((c0_value, REDUNDANT_FEATURE_COUNT),)
    loss_identity = {
        "data_digest": data["data_digest"],
        "model_type": model_type,
        "intercept_bounds": list(intercept_bounds),
        "feature_bounds": list(feature_bounds),
    }
    loss_digest = identity_digest(loss_identity)
    return {
        "loss_identity": loss_identity,
        "loss_identity_digest": loss_digest,
        "coefficient_set_lower_bounds": lower_bounds,
        "coefficient_set_upper_bounds": upper_bounds,
        "queries": {
            key: select_query(loss_digest, coefficients, intercepts, losses, supports, *key)
            for key in query_keys
        },
    }


def generate_case_record(case_id: str) -> dict:
    """Generate the cached answers for one case."""
    data = generate_data(case_id)
    model_types = (
        ("checklist",)
        if data["redundant_features"] == "noisy_duplicates"
        else tuple(MODEL_FEATURE_BOUNDS)
    )
    return {
        "case_id": case_id,
        "variable_names": data["variable_names"],
        "outcome_name": data["outcome_name"],
        "data_digest": data["data_digest"],
        # data_identity, loss_identity and query_identity feed test_training.py's precondition checks
        "data_identity": {"data_digest": data["data_digest"]},
        "tasks": {model_type: enumerate_task(data, model_type) for model_type in model_types},
    }


def regenerate_cases(case_ids: tuple[str, ...]) -> None:
    """Replace the selected cases in the cache, saving after each one."""
    with TEST_CASE_PATH.open("rb") as file_handle:
        store = pickle.load(file_handle)
    for case_id in case_ids:
        store["synthetic_oracles"][case_id] = generate_case_record(case_id)
        with TEST_CASE_PATH.open("wb") as file_handle:
            pickle.dump(store, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"generated {case_id}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true", required=True)
    parser.add_argument("--case", choices=ALL_CASE_IDS)
    arguments = parser.parse_args()
    regenerate_cases((arguments.case,) if arguments.case else ALL_CASE_IDS)

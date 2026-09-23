"""Generate the checked-in exact training test cases.

This maintenance script independently enumerates bounded integer score models.
It never calls RiskSLIM loss or solver code to choose an oracle solution.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from riskslim.coefficient_set import CoefficientSet
from riskslim.defaults import INTERCEPT_NAME


SCHEMA_VERSION = 2
SYNTHETIC_RECORD_VERSION = 2
ORACLE_ALGORITHM_VERSION = "integer-intercept-profile-v1"
DATA_GENERATION_VERSION = "independent-bernoulli-logistic-randomstate-rounded-v3"
REDUNDANT_DATA_GENERATION_VERSION = "redundant-features-randomstate-rounded-v3"
RANDOM_NUMBER_GENERATOR = "numpy.random.RandomState(MT19937)"
TEST_CASE_PATH = Path(__file__).resolve().parent / "training_test_cases.pkl"
N_SAMPLES = 10_000
SEED = 0
CHUNK_SIZE = 256
BENCHMARK_VECTORS = 256
MAX_PROFILE_WORKSPACE_BYTES = 512 * 1024**2
PROFILE_WORKSPACE_ARRAY_MULTIPLIER = 6
INTERCEPT_DELTA_EXPONENT = math.expm1(1.0)
MODEL_FEATURE_BOUNDS = {"risk-score": (-5, 5), "checklist": (-1, 1)}
TRUTHS = {
    "integer": (-1.0, 1.0, 2.0, 3.0, 4.0),
    "fractional": (-1.0, 1.0, 2.0, 0.5, 1.5),
    "shifted_fractional": (1.5, 1.0, 2.0, 0.5, 1.5),
}
FEATURE_DISTRIBUTIONS = ("binary", "continuous", "mixed")
REDUNDANT_FEATURES = ("duplicates", "noisy_duplicates")
BINARY_FEATURE_PROBABILITY = 0.5
CONTINUOUS_FEATURE_STANDARD_DEVIATION = 2.0
CONTINUOUS_DECIMAL_PLACES = 10
LABEL_SEED_SEQUENCE = [SEED, 0, 1]
CASE_LABEL_SEEDS = {"binary__shifted_fractional": [1, 0, 1]}
NOISE_SEED_SEQUENCE = [SEED, 1, 2]
BIT_FLIP_PROBABILITY = 0.05
GAUSSIAN_NOISE_STANDARD_DEVIATION = 0.25
CONTINUOUS_REFERENCE_TOLERANCE = 1e-12
CONTINUOUS_REFERENCE_MAX_ITERATIONS = 10_000
PROVISIONAL_REDUNDANCY_PENALTY = 0.01
ALTERNATIVE_REDUNDANCY_PENALTIES = (0.1, 0.03)
C0_VALUES = (1e-6, PROVISIONAL_REDUNDANCY_PENALTY, 0.1, 1.0)
CERTIFICATE_GAP_TOLERANCE = 1e-6
CONTINUOUS_GRADIENT_TOLERANCE = 1e-6
QUERY_PATH_TOLERANCE = 5e-14
QUERY_RECOMPUTATION_TOLERANCE = 1e-12
PROFILE_COMPARISON_TOLERANCE = 2e-15
BASE_FEATURE_COUNT = len(next(iter(TRUTHS.values()))) - 1
REDUNDANT_FEATURE_COUNT = 2 * BASE_FEATURE_COUNT
FIXED_SIZE_LIMITS = (2, 1, 0)
MAX_SIZES = (BASE_FEATURE_COUNT, *FIXED_SIZE_LIMITS)
REDUNDANT_MAX_SIZES = (REDUNDANT_FEATURE_COUNT, *FIXED_SIZE_LIMITS)
BASELINE_QUERY_KEYS = (
    *((1e-6, max_size) for max_size in MAX_SIZES),
    *((c0_value, BASE_FEATURE_COUNT) for c0_value in C0_VALUES[1:]),
)
MODEL_GENERATION_ORDER = ("checklist", "risk-score")
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


ARGUMENT_PARSER = argparse.ArgumentParser(description=__doc__)
OPERATION_ARGUMENTS = ARGUMENT_PARSER.add_mutually_exclusive_group()
OPERATION_ARGUMENTS.add_argument("--regenerate", action="store_true")
OPERATION_ARGUMENTS.add_argument("--refresh-queries", action="store_true")
OPERATION_ARGUMENTS.add_argument("--preflight", action="store_true")
ARGUMENT_PARSER.add_argument("--case", choices=ALL_CASE_IDS)


def load_store(allow_legacy_schema: bool = False) -> dict:
    """Load the shared store without altering known-reference records."""
    if not TEST_CASE_PATH.exists():
        raise FileNotFoundError(
            "known expectations cannot be recomputed; restore them with "
            "`git restore tests/training_test_cases.pkl`"
        )
    with TEST_CASE_PATH.open("rb") as file_handle:
        store = pickle.load(file_handle)
    if not isinstance(store, dict):
        raise ValueError("unsupported training test-case store")
    schema_version = store.get("schema_version")
    if schema_version != SCHEMA_VERSION and not (allow_legacy_schema and schema_version == 1):
        raise ValueError(
            "unsupported training test-case store; migrate it with "
            "`uv run python tests/generate_training_test_cases.py --regenerate`"
        )
    if not isinstance(store.get("synthetic_oracles"), dict):
        raise ValueError("synthetic_oracles must be a dictionary")
    if not isinstance(store.get("known_reference_results"), dict):
        raise ValueError("known_reference_results must be a dictionary")
    return store


def write_store(store: dict) -> None:
    """Atomically replace the trusted local pickle."""
    temporary_path = TEST_CASE_PATH.with_name(f".{TEST_CASE_PATH.name}.{os.getpid()}.tmp")
    with temporary_path.open("wb") as file_handle:
        pickle.dump(store, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(TEST_CASE_PATH)


def hash_array(digest, array: np.ndarray) -> None:
    """Add an array's dtype, shape, and contiguous bytes to a digest."""
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.view(np.uint8))


def compute_data_digest(X, y, variable_names, outcome_name) -> str:
    """Return a stable identity for a generated training dataset."""
    digest = hashlib.sha256()
    for array in (X, y):
        hash_array(digest, array)
    digest.update(json.dumps(variable_names).encode("utf-8"))
    digest.update(outcome_name.encode("utf-8"))
    return digest.hexdigest()


def identity_digest(identity: dict) -> str:
    """Hash a JSON-compatible identity dictionary."""
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_data_generation_spec(case_id: str) -> dict:
    """Build the complete deterministic specification for one baseline dataset."""
    feature_distribution, truth_name = case_id.split("__")
    variable_names = tuple(
        f"{feature_distribution}_{index}" for index in range(1, BASE_FEATURE_COUNT + 1)
    )
    if feature_distribution == "mixed":
        variable_names = ("binary_1", "binary_2", "continuous_1", "continuous_2")
    spec = {
        "generation_version": DATA_GENERATION_VERSION,
        "random_number_generator": RANDOM_NUMBER_GENERATOR,
        "case_id": case_id,
        "feature_distribution": feature_distribution,
        "rho_true": list(TRUTHS[truth_name]),
        "n_samples": N_SAMPLES,
        "feature_seed": SEED,
        "label_seed": list(CASE_LABEL_SEEDS.get(case_id, LABEL_SEED_SEQUENCE)),
        "variable_names": variable_names,
        "outcome_name": "y",
        "label_distribution": "bernoulli-logistic",
    }
    if feature_distribution in ("binary", "mixed"):
        spec["binary_feature_probability"] = BINARY_FEATURE_PROBABILITY
        spec["binary_feature_count"] = (
            BASE_FEATURE_COUNT if feature_distribution == "binary" else BASE_FEATURE_COUNT // 2
        )
    if feature_distribution in ("continuous", "mixed"):
        spec["continuous_feature_mean"] = 0.0
        spec["continuous_feature_standard_deviation"] = CONTINUOUS_FEATURE_STANDARD_DEVIATION
        spec["continuous_decimal_places"] = CONTINUOUS_DECIMAL_PLACES
        spec["continuous_feature_count"] = (
            BASE_FEATURE_COUNT if feature_distribution == "continuous" else BASE_FEATURE_COUNT // 2
        )
    return spec


def build_data_identity(generation_spec: dict, data_digest: str) -> dict:
    """Build the stable identity for a deterministically generated dataset."""
    return {"generation_spec": generation_spec, "data_digest": data_digest}


def redundant_perturbation(redundant_features: str, feature_distribution: str) -> dict:
    """Describe the deterministic duplicate transformation."""
    if redundant_features == "duplicates":
        return {"kind": "exact_copy"}
    binary_columns = []
    continuous_columns = []
    if feature_distribution == "binary":
        binary_columns = list(range(BASE_FEATURE_COUNT))
    elif feature_distribution == "continuous":
        continuous_columns = list(range(BASE_FEATURE_COUNT))
    else:
        binary_columns = list(range(BASE_FEATURE_COUNT // 2))
        continuous_columns = list(range(BASE_FEATURE_COUNT // 2, BASE_FEATURE_COUNT))
    return {
        "kind": "distribution_specific_noise",
        "noise_seed": list(NOISE_SEED_SEQUENCE),
        "binary_bit_flip_probability": BIT_FLIP_PROBABILITY,
        "continuous_gaussian_mean": 0.0,
        "continuous_gaussian_standard_deviation": GAUSSIAN_NOISE_STANDARD_DEVIATION,
        "continuous_decimal_places": CONTINUOUS_DECIMAL_PLACES,
        "binary_columns": binary_columns,
        "continuous_columns": continuous_columns,
    }


def build_redundant_data_generation_spec(source: dict, redundant_features: str) -> dict:
    """Build the complete deterministic specification for redundant features."""
    source_spec = source["generation_spec"]
    feature_distribution = source_spec["feature_distribution"]
    suffix = "duplicate" if redundant_features == "duplicates" else "noisy_duplicate"
    variable_names = tuple(source["variable_names"]) + tuple(
        f"{name}_{suffix}" for name in source["variable_names"]
    )
    return {
        "generation_version": REDUNDANT_DATA_GENERATION_VERSION,
        "random_number_generator": RANDOM_NUMBER_GENERATOR,
        "case_id": f"{source['case_id']}__{redundant_features}",
        "source_case_id": source["case_id"],
        "source_data_digest": source["data_digest"],
        "feature_distribution": feature_distribution,
        "rho_true": list(source_spec["rho_true"]),
        "n_samples": N_SAMPLES,
        "redundant_features": redundant_features,
        "perturbation": redundant_perturbation(redundant_features, feature_distribution),
        "labels_from_source": True,
        "variable_names": variable_names,
        "outcome_name": source["outcome_name"],
    }


def values_match(left, right) -> bool:
    """Compare nested cache metadata, including NumPy arrays."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and np.array_equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(values_match(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                values_match(left_value, right_value)
                for left_value, right_value in zip(left, right)
            )
        )
    return bool(left == right)


def generate_data(case_id: str, generation_spec: dict | None = None) -> dict:
    """Generate one fixed-seed independent Bernoulli-logistic dataset."""
    expected_spec = build_data_generation_spec(case_id)
    spec = expected_spec if generation_spec is None else generation_spec
    if not values_match(spec, expected_spec):
        raise ValueError(f"{case_id} has an unsupported baseline generation specification")
    feature_distribution = spec["feature_distribution"]
    truth = np.asarray(spec["rho_true"], dtype=np.float64)
    feature_rng = np.random.RandomState(spec["feature_seed"])
    if feature_distribution == "binary":
        X = feature_rng.binomial(
            1,
            spec["binary_feature_probability"],
            size=(spec["n_samples"], spec["binary_feature_count"]),
        ).astype(np.int8)
    elif feature_distribution == "continuous":
        X = np.round(
            feature_rng.normal(
                spec["continuous_feature_mean"],
                spec["continuous_feature_standard_deviation"],
                size=(spec["n_samples"], spec["continuous_feature_count"]),
            ),
            spec["continuous_decimal_places"],
        )
    elif feature_distribution == "mixed":
        binary = feature_rng.binomial(
            1,
            spec["binary_feature_probability"],
            size=(spec["n_samples"], spec["binary_feature_count"]),
        )
        continuous = np.round(
            feature_rng.normal(
                spec["continuous_feature_mean"],
                spec["continuous_feature_standard_deviation"],
                size=(spec["n_samples"], spec["continuous_feature_count"]),
            ),
            spec["continuous_decimal_places"],
        )
        X = np.hstack((binary, continuous)).astype(np.float64)
    else:
        raise ValueError(f"unknown feature distribution: {feature_distribution}")

    label_rng = np.random.RandomState(spec["label_seed"])
    probabilities = expit(truth[0] + X @ truth[1:])
    y = label_rng.binomial(1, probabilities).astype(np.int8)
    variable_names = tuple(spec["variable_names"])
    outcome_name = spec["outcome_name"]
    data_digest = compute_data_digest(X, y, variable_names, outcome_name)
    data_identity = build_data_identity(spec, data_digest)
    return {
        "case_id": case_id,
        "X": X,
        "y": y,
        "variable_names": variable_names,
        "outcome_name": outcome_name,
        "data_digest": data_digest,
        "data_identity": data_identity,
        "generation_spec": spec,
        "rho_true": truth,
    }


def generate_redundant_data(
    source: dict, redundant_features: str, generation_spec: dict | None = None
) -> dict:
    """Add deterministic copies while retaining labels from the clean features."""
    if redundant_features not in REDUNDANT_FEATURES:
        raise ValueError(f"unknown redundant-feature arrangement: {redundant_features}")
    expected_spec = build_redundant_data_generation_spec(source, redundant_features)
    spec = expected_spec if generation_spec is None else generation_spec
    case_id = f"{source['case_id']}__{redundant_features}"
    if not values_match(spec, expected_spec):
        raise ValueError(f"{case_id} has an unsupported redundant generation specification")
    clean_X = source["X"]
    perturbation = spec["perturbation"]
    if redundant_features == "duplicates":
        if perturbation != {"kind": "exact_copy"}:
            raise ValueError(f"{case_id} exact-copy specification does not match")
        copied_X = clean_X.copy()
    else:
        feature_distribution = spec["feature_distribution"]
        noise_rng = np.random.RandomState(perturbation["noise_seed"])
        copied_X = clean_X.astype(np.float64, copy=True)
        binary_columns = perturbation["binary_columns"]
        continuous_columns = perturbation["continuous_columns"]
        if binary_columns:
            flips = noise_rng.binomial(
                1,
                perturbation["binary_bit_flip_probability"],
                size=(spec["n_samples"], len(binary_columns)),
            )
            copied_X[:, binary_columns] = np.abs(clean_X[:, binary_columns] - flips)
        if continuous_columns:
            copied_X[:, continuous_columns] += noise_rng.normal(
                perturbation["continuous_gaussian_mean"],
                perturbation["continuous_gaussian_standard_deviation"],
                size=(spec["n_samples"], len(continuous_columns)),
            )
            copied_X[:, continuous_columns] = np.round(
                copied_X[:, continuous_columns], perturbation["continuous_decimal_places"]
            )
        if feature_distribution == "binary":
            copied_X = copied_X.astype(np.int8)

    X = np.hstack((clean_X, copied_X))
    variable_names = tuple(spec["variable_names"])
    outcome_name = spec["outcome_name"]
    data_digest = compute_data_digest(X, source["y"], variable_names, outcome_name)
    data_identity = build_data_identity(spec, data_digest)
    return {
        "case_id": case_id,
        "source_case_id": source["case_id"],
        "redundant_features": redundant_features,
        "X": X,
        "y": source["y"].copy(),
        "variable_names": variable_names,
        "outcome_name": outcome_name,
        "data_digest": data_digest,
        "data_identity": data_identity,
        "generation_spec": spec,
        "rho_true": np.r_[spec["rho_true"], np.zeros(BASE_FEATURE_COUNT)],
        "source_rho_true": np.asarray(spec["rho_true"], dtype=np.float64),
    }


def regenerate_case_data(record: dict, source: dict | None = None) -> dict:
    """Regenerate synthetic arrays from a stored specification without fitting or enumeration."""
    case_id = record.get("case_id")
    generation_spec = record.get("generation_spec")
    if not isinstance(generation_spec, dict):
        raise ValueError(f"{case_id} is missing its generation specification")
    redundant_features = generation_spec.get("redundant_features")
    if redundant_features is None:
        if source is not None:
            raise ValueError(f"{case_id} baseline regeneration does not accept a source")
        return generate_data(case_id, generation_spec=generation_spec)
    if source is None:
        raise ValueError(f"{case_id} redundant regeneration requires its hydrated source")
    return generate_redundant_data(source, redundant_features, generation_spec=generation_spec)


def determine_coefficient_bounds(data: dict, feature_bounds: tuple[int, int]) -> dict:
    """Derive the intercept domain with the repository's coefficient set."""
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
    raw_lower = coefficient_set.lb.astype(np.float64)
    raw_upper = coefficient_set.ub.astype(np.float64)
    integer_lower = np.r_[math.ceil(raw_lower[0]), raw_lower[1:]].astype(np.int16)
    integer_upper = np.r_[math.floor(raw_upper[0]), raw_upper[1:]].astype(np.int16)
    if np.any(integer_lower > integer_upper):
        raise AssertionError("automatic coefficient bounds contain no integer")
    return {
        "coefficient_set_lower_bounds": raw_lower,
        "coefficient_set_upper_bounds": raw_upper,
        "effective_integer_lower_bounds": integer_lower,
        "effective_integer_upper_bounds": integer_upper,
    }


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
    if lower != -upper or upper <= 0:
        raise ValueError("duplicate grouping requires symmetric nonzero integer bounds")
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


def validate_intercept_profiler() -> dict:
    """Compare profiling with independent direct enumeration on small fixtures."""
    fixture_y = np.array([1, 0], dtype=np.int8)
    fixtures = {
        "lower_endpoint": (np.array([10.0, 10.0]), {-2}),
        "upper_endpoint": (np.array([-10.0, -10.0]), {2}),
        "adjacent_tie": (np.array([0.0, -1.0]), {0, 1}),
        "unique_zero": (np.array([0.0, 0.0]), {0}),
    }
    results = {}
    for name, (scores, expected_minimizers) in fixtures.items():
        intercepts, losses, _ = profile_integer_intercepts(scores[:, None], fixture_y, -2, 2)
        direct_losses = np.asarray(
            [
                np.mean(np.logaddexp(0.0, (1.0 - 2.0 * fixture_y) * (scores + intercept)))
                for intercept in range(-2, 3)
            ]
        )
        minimum_loss = float(np.min(direct_losses))
        minimizers = set(np.arange(-2, 3)[direct_losses == minimum_loss].tolist())
        if minimizers != expected_minimizers or int(intercepts[0]) not in minimizers:
            raise AssertionError(f"intercept profiler failed {name}")
        if not np.isclose(losses[0], minimum_loss, rtol=0.0, atol=PROFILE_COMPARISON_TOLERANCE):
            raise AssertionError(f"intercept profiler loss failed {name}")
        results[name] = sorted(minimizers)

    rng = np.random.RandomState(9173)
    X = rng.normal(size=(37, 2))
    y = rng.binomial(1, expit(-0.3 + X @ np.array([0.7, -1.1]))).astype(np.int8)
    coefficients = np.asarray(list(itertools.product(range(-2, 3), repeat=2)))
    scores = X @ coefficients.T
    intercepts, losses, _ = profile_integer_intercepts(scores, y, -4, 4)
    direct_intercepts = np.arange(-4, 5)
    for index, score in enumerate(scores.T):
        direct_losses = np.asarray(
            [
                np.mean(np.logaddexp(0.0, (1.0 - 2.0 * y) * (score + intercept)))
                for intercept in direct_intercepts
            ]
        )
        best_loss = np.min(direct_losses)
        if not np.isclose(losses[index], best_loss, rtol=0.0, atol=PROFILE_COMPARISON_TOLERANCE):
            raise AssertionError("profiled loss disagrees with direct enumeration")
        if int(intercepts[index]) not in direct_intercepts[direct_losses == best_loss]:
            raise AssertionError("profiled intercept is not a direct minimizer")
    return {"passed": True, "deterministic_minimizers": results, "full_models_checked": 225}


def validate_duplicate_grouping() -> dict:
    """Exhaustively compare grouped and direct duplicate grids at a small bound."""
    rng = np.random.RandomState(3107)
    clean_X = rng.normal(size=(31, BASE_FEATURE_COUNT))
    X = np.hstack((clean_X, clean_X))
    y = rng.binomial(1, expit(-0.2 + clean_X @ np.array([0.8, -0.6, 0.4, 1.1])))
    direct_coefficients = build_feature_coefficients((-1, 1), n_features=REDUNDANT_FEATURE_COUNT)
    pair_sums, grouped_coefficients, grouped_supports = build_duplicate_group_coefficients((-1, 1))
    direct_intercepts, direct_losses, _ = profile_integer_intercepts(
        X @ direct_coefficients.T, y, -5, 5
    )
    grouped_intercepts, grouped_losses, _ = profile_integer_intercepts(
        clean_X @ pair_sums.T, y, -5, 5
    )
    grouped_indices = {tuple(pair_sum): index for index, pair_sum in enumerate(pair_sums)}
    minimum_direct_supports = np.full(len(pair_sums), 9, dtype=np.uint8)
    for coefficient_index, coefficient in enumerate(direct_coefficients):
        pair_sum = coefficient[:BASE_FEATURE_COUNT] + coefficient[BASE_FEATURE_COUNT:]
        grouped_index = grouped_indices[tuple(pair_sum)]
        if not np.isclose(
            direct_losses[coefficient_index],
            grouped_losses[grouped_index],
            rtol=0.0,
            atol=PROFILE_COMPARISON_TOLERANCE,
        ):
            raise AssertionError("duplicate grouping changed a profiled loss")
        if direct_intercepts[coefficient_index] != grouped_intercepts[grouped_index]:
            raise AssertionError("duplicate grouping changed a profiled intercept")
        minimum_direct_supports[grouped_index] = min(
            minimum_direct_supports[grouped_index],
            np.count_nonzero(coefficient),
        )
    if not np.array_equal(grouped_supports, minimum_direct_supports):
        raise AssertionError("duplicate grouping changed minimum support")

    direct_supports = np.count_nonzero(direct_coefficients, axis=1).astype(np.uint8)
    direct_violates_pair = np.any(
        (direct_coefficients[:, :BASE_FEATURE_COUNT] != 0)
        & (direct_coefficients[:, BASE_FEATURE_COUNT:] != 0),
        axis=1,
    )
    grouped_violating_supports = minimum_grouped_pair_violating_supports(
        grouped_coefficients, feature_bound=1
    )
    for c0_value in C0_VALUES:
        for max_size in REDUNDANT_MAX_SIZES:
            direct_feasible = direct_supports <= max_size
            grouped_feasible = grouped_supports <= max_size
            direct_objective = np.min(
                direct_losses[direct_feasible] + c0_value * direct_supports[direct_feasible]
            )
            grouped_objective = np.min(
                grouped_losses[grouped_feasible] + c0_value * grouped_supports[grouped_feasible]
            )
            if not np.isclose(
                direct_objective,
                grouped_objective,
                rtol=0.0,
                atol=PROFILE_COMPARISON_TOLERANCE,
            ):
                raise AssertionError("duplicate grouping changed a query optimum")
            direct_violating = direct_violates_pair & direct_feasible
            grouped_violating = grouped_violating_supports <= max_size
            if np.any(direct_violating) != np.any(grouped_violating):
                raise AssertionError("duplicate grouping changed pair-violation feasibility")
            if np.any(direct_violating):
                direct_violating_objective = np.min(
                    direct_losses[direct_violating] + c0_value * direct_supports[direct_violating]
                )
                grouped_violating_objective = np.min(
                    grouped_losses[grouped_violating]
                    + c0_value * grouped_violating_supports[grouped_violating]
                )
                if not np.isclose(
                    direct_violating_objective,
                    grouped_violating_objective,
                    rtol=0.0,
                    atol=PROFILE_COMPARISON_TOLERANCE,
                ):
                    raise AssertionError("duplicate grouping changed pair-violation optimum")
    return {
        "passed": True,
        "bound": [-1, 1],
        "direct_vectors_checked": int(len(direct_coefficients)),
        "grouped_states_checked": int(len(grouped_coefficients)),
        "queries_checked": len(C0_VALUES) * len(REDUNDANT_MAX_SIZES),
        "pair_violation_queries_checked": len(C0_VALUES) * len(REDUNDANT_MAX_SIZES),
    }


def limited_batch_size(n_samples: int, requested_vectors: int) -> int:
    """Cap batches under the declared six-float-array workspace."""
    bytes_per_vector = n_samples * np.dtype(np.float64).itemsize
    maximum_vectors = MAX_PROFILE_WORKSPACE_BYTES // (
        PROFILE_WORKSPACE_ARRAY_MULTIPLIER * bytes_per_vector
    )
    if maximum_vectors < 1:
        raise ValueError("profile exceeds the 512 MiB workspace even for one vector")
    return min(requested_vectors, maximum_vectors)


def benchmark_profile(X, y, coefficients, intercept_bounds, batch_size) -> dict:
    """Estimate complete enumeration time from one bounded batch."""
    count = min(batch_size, len(coefficients))
    started = time.perf_counter()
    profile_integer_intercepts(X @ coefficients[:count].T, y, *intercept_bounds)
    seconds = time.perf_counter() - started
    return {
        "vectors_benchmarked": count,
        "benchmark_seconds": seconds,
        "estimated_enumeration_seconds": seconds * len(coefficients) / count,
    }


def enumerate_losses(X, y, coefficients, intercept_bounds, batch_size):
    """Profile every feature vector in bounded-memory NumPy batches."""
    intercepts = np.empty(len(coefficients), dtype=np.int16)
    losses = np.empty(len(coefficients), dtype=np.float64)
    ties = np.empty(len(coefficients), dtype=np.uint8)
    started = time.perf_counter()
    for start in range(0, len(coefficients), batch_size):
        stop = min(start + batch_size, len(coefficients))
        intercepts[start:stop], losses[start:stop], ties[start:stop] = profile_integer_intercepts(
            X @ coefficients[start:stop].T, y, *intercept_bounds
        )
    return intercepts, losses, ties, time.perf_counter() - started


def validate_loss_arrays(coefficients, intercepts, losses, supports, ties, lower, upper):
    """Reject incomplete or invalid exact-enumeration output."""
    n_vectors, n_features = coefficients.shape
    if n_features != len(lower) - 1:
        raise ValueError("coefficient arrays do not match the bounded feature count")
    if any(array.shape != (n_vectors,) for array in (intercepts, losses, supports, ties)):
        raise ValueError("loss arrays must share the feature-vector count")
    if not np.isfinite(losses).all() or np.any(losses < 0.0):
        raise ValueError("losses must be finite and nonnegative")
    if not np.equal(intercepts, np.rint(intercepts)).all():
        raise ValueError("intercepts must be integers")
    if np.any(intercepts < lower[0]) or np.any(intercepts > upper[0]):
        raise ValueError("intercepts fall outside their integer bounds")
    if np.any(coefficients < lower[1:]) or np.any(coefficients > upper[1:]):
        raise ValueError("feature coefficients fall outside their integer bounds")
    if not np.array_equal(supports, np.count_nonzero(coefficients, axis=1)):
        raise ValueError("supports do not match feature coefficients")
    if not np.isin(ties, (1, 2)).all():
        raise ValueError("intercept tie multiplicities must be 1 or 2")


def select_queries(
    loss_digest, coefficients, intercepts, losses, supports, max_sizes=MAX_SIZES
) -> dict:
    """Answer every size and penalty query from one shared loss table."""
    return {
        (float(c0_value), int(max_size)): select_query(
            loss_digest,
            coefficients,
            intercepts,
            losses,
            supports,
            c0_value,
            max_size,
        )
        for c0_value in C0_VALUES
        for max_size in max_sizes
    }


def select_query(
    loss_digest, coefficients, intercepts, losses, supports, c0_value, max_size
) -> dict:
    """Select one exact optimum from a cached loss table."""
    feasible_indices = np.flatnonzero(supports <= max_size)
    objectives = losses[feasible_indices] + c0_value * supports[feasible_indices]
    index = int(feasible_indices[int(np.argmin(objectives))])
    objective = float(losses[index] + c0_value * supports[index])
    query_identity = {
        "loss_identity_digest": loss_digest,
        "c0": float(c0_value),
        "max_size": int(max_size),
    }
    return {
        "query_identity": query_identity,
        "query_identity_digest": identity_digest(query_identity),
        "representative_rho": np.r_[intercepts[index], coefficients[index]].astype(np.int16),
        "pure_logistic_loss": float(losses[index]),
        "penalty": float(c0_value * supports[index]),
        "objective": objective,
        "cardinality": int(supports[index]),
        "representative_tie_count": int(np.count_nonzero(objectives == objective)),
        "proven_complete": True,
    }


def validate_query_paths(queries: dict, max_sizes=MAX_SIZES) -> None:
    """Check exact-query monotonicity across the admitted paths."""
    for max_size in max_sizes:
        path = [queries[(c0, max_size)] for c0 in C0_VALUES]
        support_path = [query["cardinality"] for query in path]
        loss_path = [query["pure_logistic_loss"] for query in path]
        if any(right > left for left, right in zip(support_path, support_path[1:], strict=False)):
            raise AssertionError("support increased with c0")
        if any(
            right + QUERY_PATH_TOLERANCE < left
            for left, right in zip(loss_path, loss_path[1:], strict=False)
        ):
            raise AssertionError("loss decreased with c0")
    for c0_value in C0_VALUES:
        path = [queries[(c0_value, max_size)] for max_size in max_sizes]
        objective_path = [query["objective"] for query in path]
        loss_path = [query["pure_logistic_loss"] for query in path]
        if any(
            right + QUERY_PATH_TOLERANCE < left
            for left, right in zip(objective_path, objective_path[1:], strict=False)
        ):
            raise AssertionError("objective decreased as max size shrank")
        if any(
            right + QUERY_PATH_TOLERANCE < left
            for left, right in zip(loss_path, loss_path[1:], strict=False)
        ):
            raise AssertionError("loss decreased as max size shrank")
    if any(queries[(1.0, max_size)]["cardinality"] != 0 for max_size in max_sizes):
        raise AssertionError("c0=1 baseline optimum must have zero feature support")


def fit_continuous_reference(data: dict) -> dict | None:
    """Fit the diagnostic unconstrained model for fractional planted truths."""
    truth = data["rho_true"]
    if np.array_equal(truth, np.rint(truth)):
        return None
    settings = {
        "method": "sklearn.LogisticRegression",
        "penalty": None,
        "solver": "lbfgs",
        "tolerance": CONTINUOUS_REFERENCE_TOLERANCE,
        "max_iterations": CONTINUOUS_REFERENCE_MAX_ITERATIONS,
    }
    started = time.perf_counter()
    model = LogisticRegression(
        penalty=None,
        fit_intercept=True,
        solver="lbfgs",
        tol=settings["tolerance"],
        max_iter=settings["max_iterations"],
        random_state=SEED,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="'penalty' was deprecated.*", category=FutureWarning
        )
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(data["X"], data["y"])
    rho = np.r_[model.intercept_[0], model.coef_[0]]
    probabilities = expit(rho[0] + data["X"] @ rho[1:])
    augmented_X = np.column_stack((np.ones(len(data["X"])), data["X"]))
    gradient = augmented_X.T @ (probabilities - data["y"]) / len(data["y"])
    signed_scores = (1.0 - 2.0 * data["y"]) * (rho[0] + data["X"] @ rho[1:])
    converged = bool(model.n_iter_[0] < model.max_iter)
    gradient_infinity_norm = float(np.max(np.abs(gradient)))
    if (
        not converged
        or not np.isfinite(rho).all()
        or gradient_infinity_norm > CONTINUOUS_GRADIENT_TOLERANCE
    ):
        raise AssertionError("continuous diagnostic did not reach a finite stationary fit")
    identity = {"data_digest": data["data_digest"], **settings}
    return {
        "identity": identity,
        "identity_digest": identity_digest(identity),
        "rho": rho,
        "pure_logistic_loss": float(np.mean(np.logaddexp(0.0, signed_scores))),
        "converged": converged,
        "gradient_infinity_norm": gradient_infinity_norm,
        "n_iterations": int(model.n_iter_[0]),
        "fit_seconds": time.perf_counter() - started,
        "interpretation": "diagnostic primal fit; not a rigorous lower bound or integer oracle",
    }


def exact_strategy_for(data: dict, model_type: str) -> dict:
    """Declare the approved exact enumeration strategy for a redundant profile."""
    if data["redundant_features"] == "duplicates":
        return {
            "name": "duplicate-pair-group-sums",
            "pair_count": BASE_FEATURE_COUNT,
            "pairing": [[index, index + BASE_FEATURE_COUNT] for index in range(BASE_FEATURE_COUNT)],
            "representative": "clip_pair_sum_then_assign_remainder",
            "support": "minimum_nonzero_coefficients_for_each_pair_sum",
        }
    if model_type == "checklist":
        return {"name": "direct-feature-vector-enumeration"}
    raise ValueError("no approved exact strategy for noisy-duplicate risk-score data")


def preflight_case(case_id: str) -> dict:
    """Build and benchmark every approved exact profile for one case."""
    source = None
    if case_id in BASELINE_CASE_IDS:
        data = generate_data(case_id)
    else:
        source_case_id, redundant_features = case_id.rsplit("__", 1)
        source = generate_data(source_case_id)
        data = generate_redundant_data(source, redundant_features)

    redundant_features = data.get("redundant_features")
    model_types = (
        MODEL_FEATURE_BOUNDS
        if redundant_features in (None, "duplicates")
        else {"checklist": MODEL_FEATURE_BOUNDS["checklist"]}
    )
    profiles = {}
    for model_type, feature_bounds in model_types.items():
        bounds = determine_coefficient_bounds(data, feature_bounds)
        if redundant_features == "duplicates":
            score_coefficients, coefficients, supports = build_duplicate_group_coefficients(
                feature_bounds
            )
            score_X = source["X"]
        else:
            coefficients = build_feature_coefficients(feature_bounds, n_features=data["X"].shape[1])
            score_coefficients = coefficients
            supports = np.count_nonzero(coefficients, axis=1).astype(np.uint8)
            score_X = data["X"]
        intercept_bounds = (
            int(bounds["effective_integer_lower_bounds"][0]),
            int(bounds["effective_integer_upper_bounds"][0]),
        )
        batch_size = limited_batch_size(N_SAMPLES, CHUNK_SIZE)
        benchmark = benchmark_profile(
            score_X, data["y"], score_coefficients, intercept_bounds, BENCHMARK_VECTORS
        )
        nominal_models = (feature_bounds[1] - feature_bounds[0] + 1) ** data["X"].shape[1] * (
            intercept_bounds[1] - intercept_bounds[0] + 1
        )
        profile = {
            "bounds": bounds,
            "coefficients": coefficients,
            "intercept_bounds": intercept_bounds,
            "batch_size": batch_size,
            "nominal_full_models": nominal_models,
            "benchmark": benchmark,
        }
        strategy_message = ""
        if redundant_features:
            strategy = exact_strategy_for(data, model_type)
            profile.update(
                score_coefficients=score_coefficients,
                score_X=score_X,
                supports=supports,
                exact_strategy=strategy,
                max_sizes=REDUNDANT_MAX_SIZES,
            )
            strategy_message = f", strategy {strategy['name']}"
        profiles[model_type] = profile
        state_name = "exact states" if redundant_features else "vectors"
        print(
            f"preflight {case_id} {model_type}: {len(coefficients):,} {state_name}, "
            f"intercept [{intercept_bounds[0]}, {intercept_bounds[1]}], "
            f"{nominal_models:,} nominal models, batch {batch_size}, "
            f"estimate {benchmark['estimated_enumeration_seconds']:.1f}s{strategy_message}",
            flush=True,
        )
    return {"data": data, "profiles": profiles}


def preflight_cases(case_ids: tuple[str, ...]) -> dict:
    """Preflight each selected baseline or redundant case."""
    return {case_id: preflight_case(case_id) for case_id in case_ids}


def build_loss_identity(
    data: dict, model_type: str, bounds: dict, exact_strategy: dict | None = None
) -> dict:
    """Build the stable identity shared by generated and validated loss tables."""
    identity = {
        "algorithm_version": ORACLE_ALGORITHM_VERSION,
        "data_identity": data["data_identity"],
        "data_digest": data["data_digest"],
        "model_type": model_type,
        "effective_integer_lower_bounds": bounds["effective_integer_lower_bounds"].tolist(),
        "effective_integer_upper_bounds": bounds["effective_integer_upper_bounds"].tolist(),
    }
    if exact_strategy is not None:
        identity["exact_strategy"] = exact_strategy
    return identity


def generate_task_record(data: dict, model_type: str, profile: dict) -> dict:
    """Generate one complete exact loss table and its reusable queries."""
    coefficients = profile["coefficients"]
    score_coefficients = profile.get("score_coefficients", coefficients)
    score_X = profile.get("score_X", data["X"])
    intercepts, losses, ties, enumeration_seconds = enumerate_losses(
        score_X,
        data["y"],
        score_coefficients,
        profile["intercept_bounds"],
        profile["batch_size"],
    )
    supports = profile.get("supports")
    if supports is None:
        supports = np.count_nonzero(coefficients, axis=1).astype(np.uint8)
    bounds = profile["bounds"]
    validate_loss_arrays(
        coefficients,
        intercepts,
        losses,
        supports,
        ties,
        bounds["effective_integer_lower_bounds"],
        bounds["effective_integer_upper_bounds"],
    )
    exact_strategy = profile.get("exact_strategy")
    loss_identity = build_loss_identity(data, model_type, bounds, exact_strategy)
    loss_digest = identity_digest(loss_identity)
    max_sizes = profile.get("max_sizes", MAX_SIZES)
    queries = select_queries(
        loss_digest, coefficients, intercepts, losses, supports, max_sizes=max_sizes
    )
    validate_query_paths(queries, max_sizes=max_sizes)
    record = {
        "loss_identity": loss_identity,
        "loss_identity_digest": loss_digest,
        "coefficient_set_lower_bounds": bounds["coefficient_set_lower_bounds"],
        "coefficient_set_upper_bounds": bounds["coefficient_set_upper_bounds"],
        "effective_integer_lower_bounds": bounds["effective_integer_lower_bounds"],
        "effective_integer_upper_bounds": bounds["effective_integer_upper_bounds"],
        "feature_coefficients": coefficients,
        "optimal_intercepts": intercepts,
        "pure_logistic_losses": losses,
        "supports": supports,
        "intercept_tie_multiplicities": ties,
        "queries": queries,
        "coverage": {
            "proven_complete": True,
            "profiled_feature_vectors": int(len(coefficients)),
            "nominal_full_models": int(profile["nominal_full_models"]),
        },
        "runtime": {
            "preflight": profile["benchmark"],
            "enumeration_seconds": enumeration_seconds,
            "effective_batch_size": profile["batch_size"],
            "max_workspace_bytes": MAX_PROFILE_WORKSPACE_BYTES,
        },
        "tie_count_scope": "adjacent equal float64 intercept minima for each feature vector",
    }
    if exact_strategy is not None:
        record["exact_strategy"] = exact_strategy
        record["query_max_sizes"] = max_sizes
    return record


def initialize_case_record(data: dict) -> dict:
    """Build the data and diagnostic fields shared by every case record."""
    return {
        "record_version": SYNTHETIC_RECORD_VERSION,
        "case_id": data["case_id"],
        "generation_spec": data["generation_spec"],
        "variable_names": data["variable_names"],
        "outcome_name": data["outcome_name"],
        "data_digest": data["data_digest"],
        "data_identity": data["data_identity"],
        "continuous_reference": fit_continuous_reference(data),
        "tasks": {},
    }


def query_task(task: dict, c0_value: float, max_size: int) -> dict:
    """Select one exact query from cached losses, including an ad hoc penalty."""
    cached = task["queries"].get((c0_value, max_size))
    if cached is not None:
        return cached
    return select_query(
        task["loss_identity_digest"],
        task["feature_coefficients"],
        task["optimal_intercepts"],
        task["pure_logistic_losses"],
        task["supports"],
        c0_value,
        max_size,
    )


def build_redundancy_certificate(
    task: dict,
    c0_value: float = PROVISIONAL_REDUNDANCY_PENALTY,
    max_size: int = REDUNDANT_FEATURE_COUNT,
) -> dict:
    """Certify whether a penalty removes unnecessary paired selections."""
    query = query_task(task, c0_value, max_size)
    coefficients = task["feature_coefficients"]
    supports = task["supports"]
    if task["exact_strategy"]["name"] == "duplicate-pair-group-sums":
        feature_bound = int(task["effective_integer_upper_bounds"][1])
        violating_supports = minimum_grouped_pair_violating_supports(coefficients, feature_bound)
        violates_pair_property = violating_supports <= max_size
    else:
        paired_supports = (coefficients[:, :BASE_FEATURE_COUNT] != 0) & (
            coefficients[:, BASE_FEATURE_COUNT:] != 0
        )
        violates_pair_property = np.any(paired_supports, axis=1) & (supports <= max_size)
        violating_supports = supports
    violating_objectives = task["pure_logistic_losses"] + c0_value * violating_supports
    violating_objective = (
        float(np.min(violating_objectives[violates_pair_property]))
        if np.any(violates_pair_property)
        else math.inf
    )
    gap = violating_objective - query["objective"]
    empty_objective = float(np.min(task["pure_logistic_losses"][supports == 0]))
    empty_gap = empty_objective - query["objective"]
    representative = query["representative_rho"][1:]
    selected_pair_members = [
        [
            int(representative[index] != 0),
            int(representative[index + BASE_FEATURE_COUNT] != 0),
        ]
        for index in range(BASE_FEATURE_COUNT)
    ]
    return {
        "c0": c0_value,
        "max_size": max_size,
        "nonempty_useful_support": bool(query["cardinality"] > 0),
        "at_most_one_selected_per_pair": bool(
            all(sum(selected) <= 1 for selected in selected_pair_members)
        ),
        "selected_pair_members": selected_pair_members,
        "best_objective": query["objective"],
        "best_empty_model_objective": empty_objective,
        "empty_model_objective_gap": empty_gap,
        "all_optima_nonempty": bool(empty_gap > CERTIFICATE_GAP_TOLERANCE),
        "best_pair_violating_objective": violating_objective,
        "pair_violation_objective_gap": gap,
        "all_optima_drop_unneeded_pair_members": bool(gap > CERTIFICATE_GAP_TOLERANCE),
        "representative_tie_count": query["representative_tie_count"],
    }


def find_redundancy_certificate(task: dict) -> dict:
    """Choose the first approved penalty that proves useful pair removal."""
    for c0_value in (PROVISIONAL_REDUNDANCY_PENALTY, *ALTERNATIVE_REDUNDANCY_PENALTIES):
        certificate = build_redundancy_certificate(task, c0_value)
        if (
            certificate["nonempty_useful_support"]
            and certificate["all_optima_nonempty"]
            and certificate["at_most_one_selected_per_pair"]
            and certificate["all_optima_drop_unneeded_pair_members"]
        ):
            return certificate
    raise AssertionError("no approved penalty proves positive nonempty pair removal")


def compact_query(query: dict) -> dict:
    """Keep only the exact optimum information consumed by solver tests."""
    return {
        "query_identity": query["query_identity"],
        "query_identity_digest": query["query_identity_digest"],
        "representative_rho": query["representative_rho"],
        "pure_logistic_loss": query["pure_logistic_loss"],
        "objective": query["objective"],
        "representative_tie_count": query["representative_tie_count"],
        "proven_complete": query["proven_complete"],
    }


def compact_task_record(task: dict, query_keys: tuple[tuple[float, int], ...]) -> dict:
    """Discard enumeration arrays after retaining the exact answers used by tests."""
    return {
        "loss_identity": task["loss_identity"],
        "loss_identity_digest": task["loss_identity_digest"],
        "coefficient_set_lower_bounds": task["coefficient_set_lower_bounds"],
        "coefficient_set_upper_bounds": task["coefficient_set_upper_bounds"],
        "queries": {key: compact_query(query_task(task, key[0], key[1])) for key in query_keys},
        "coverage": task["coverage"],
    }


def generate_case_records(preflights: dict, diagnostics: dict) -> dict:
    """Generate records with every cheap checklist profile before risk-score work."""
    if not diagnostics.get("passed"):
        raise AssertionError("profiler diagnostics did not pass")
    records = {
        case_id: initialize_case_record(preflight["data"])
        for case_id, preflight in preflights.items()
    }
    for model_type in MODEL_GENERATION_ORDER:
        for case_id, preflight in preflights.items():
            profile = preflight["profiles"].get(model_type)
            if profile is None:
                continue
            task_started = time.perf_counter()
            task = generate_task_record(preflight["data"], model_type, profile)
            redundant = bool(preflight["data"].get("redundant_features"))
            if redundant:
                certificate = find_redundancy_certificate(task)
                query_keys = ((certificate["c0"], REDUNDANT_FEATURE_COUNT),)
            else:
                query_keys = BASELINE_QUERY_KEYS
            records[case_id]["tasks"][model_type] = compact_task_record(task, query_keys)
            certificate_message = ""
            if redundant:
                certificate_message = (
                    f"; c0={certificate['c0']:.6g} certificate "
                    f"nonempty={certificate['nonempty_useful_support']} "
                    "one-per-pair="
                    f"{certificate['all_optima_drop_unneeded_pair_members']} "
                    f"gap={certificate['pair_violation_objective_gap']:.12g}"
                )
            print(
                f"generated {case_id} {model_type} in "
                f"{time.perf_counter() - task_started:.1f}s{certificate_message}",
                flush=True,
            )
    return records


def hydrate_stored_data(record: dict, source: dict | None = None) -> dict:
    """Regenerate and validate one compact synthetic record without mutating it."""
    case_id = record.get("case_id")
    if record.get("record_version") != SYNTHETIC_RECORD_VERSION:
        raise ValueError(f"{case_id} has an unsupported record version")
    expected_fields = {
        "record_version",
        "case_id",
        "generation_spec",
        "variable_names",
        "outcome_name",
        "data_digest",
        "data_identity",
        "continuous_reference",
        "tasks",
    }
    if set(record) != expected_fields:
        raise ValueError(f"{case_id} compact record fields do not match")
    generated = regenerate_case_data(record, source=source)
    for name in ("variable_names", "outcome_name", "data_digest", "data_identity"):
        if not values_match(record.get(name), generated[name]):
            raise ValueError(f"{case_id} {name} does not match deterministic generation")
    hydrated = {**record, **generated}
    validate_continuous_reference(hydrated)
    return hydrated


def validate_continuous_reference(record: dict) -> None:
    """Validate a fractional-truth diagnostic against its stored dataset."""
    case_id = record["case_id"]
    data_digest = record["data_digest"]
    continuous_reference = record.get("continuous_reference")
    fractional_truth = not np.array_equal(record["rho_true"], np.rint(record["rho_true"]))
    if not fractional_truth:
        if continuous_reference is not None:
            raise ValueError(f"{case_id} should not have a continuous diagnostic")
    elif (
        not isinstance(continuous_reference, dict)
        or not continuous_reference.get("converged")
        or not np.isfinite(continuous_reference.get("pure_logistic_loss", np.nan))
        or not np.isfinite(continuous_reference.get("rho", np.nan)).all()
        or continuous_reference.get("gradient_infinity_norm", np.inf)
        > CONTINUOUS_GRADIENT_TOLERANCE
        or continuous_reference.get("identity", {}).get("data_digest") != data_digest
        or np.asarray(continuous_reference.get("rho", [])).shape != (record["X"].shape[1] + 1,)
    ):
        raise ValueError(f"{case_id} continuous diagnostic is invalid")


def expected_coverage(data: dict, model_type: str, bounds: dict) -> dict:
    """Return exact enumeration counts without materializing coefficient grids."""
    lower, upper = MODEL_FEATURE_BOUNDS[model_type]
    n_features = data["X"].shape[1]
    if data.get("redundant_features") == "duplicates":
        profiled_vectors = (2 * (upper - lower) + 1) ** BASE_FEATURE_COUNT
    else:
        profiled_vectors = (upper - lower + 1) ** n_features
    intercept_count = (
        int(bounds["effective_integer_upper_bounds"][0])
        - int(bounds["effective_integer_lower_bounds"][0])
        + 1
    )
    return {
        "proven_complete": True,
        "profiled_feature_vectors": profiled_vectors,
        "nominal_full_models": (upper - lower + 1) ** n_features * intercept_count,
    }


def validate_stored_query(data: dict, task: dict, key: tuple[float, int], query: dict) -> dict:
    """Validate one compact query against its identity, bounds, and regenerated data."""
    c0_value, max_size = key
    expected_identity = {
        "loss_identity_digest": task["loss_identity_digest"],
        "c0": c0_value,
        "max_size": max_size,
    }
    expected_fields = {
        "query_identity",
        "query_identity_digest",
        "representative_rho",
        "pure_logistic_loss",
        "objective",
        "representative_tie_count",
        "proven_complete",
    }
    if not isinstance(query, dict) or set(query) != expected_fields:
        raise ValueError(f"query {key} compact fields do not match")
    if (
        query["query_identity"] != expected_identity
        or query["query_identity_digest"] != identity_digest(expected_identity)
        or query["proven_complete"] is not True
    ):
        raise ValueError(f"query {key} identity does not match")
    rho = query["representative_rho"]
    lower = np.asarray(task["loss_identity"]["effective_integer_lower_bounds"])
    upper = np.asarray(task["loss_identity"]["effective_integer_upper_bounds"])
    if (
        not isinstance(rho, np.ndarray)
        or rho.shape != lower.shape
        or not np.equal(rho, np.rint(rho)).all()
        or np.any(rho < lower)
        or np.any(rho > upper)
    ):
        raise ValueError(f"query {key} representative is outside its integer bounds")
    cardinality = int(np.count_nonzero(rho[1:]))
    if cardinality > max_size:
        raise ValueError(f"query {key} representative exceeds its size limit")
    signed_scores = (1.0 - 2.0 * data["y"]) * (rho[0] + data["X"] @ rho[1:])
    loss = float(np.mean(np.logaddexp(0.0, signed_scores)))
    objective = loss + c0_value * cardinality
    if not np.isclose(
        query["pure_logistic_loss"], loss, rtol=0.0, atol=QUERY_RECOMPUTATION_TOLERANCE
    ):
        raise ValueError(f"query {key} loss does not match its representative")
    if not np.isclose(query["objective"], objective, rtol=0.0, atol=QUERY_RECOMPUTATION_TOLERANCE):
        raise ValueError(f"query {key} objective does not match its representative")
    if (
        not isinstance(query["representative_tie_count"], int)
        or query["representative_tie_count"] < 1
    ):
        raise ValueError(f"query {key} tie count is invalid")
    return {"cardinality": cardinality, "loss": loss, "objective": objective}


def validate_compact_query_paths(validated: dict) -> None:
    """Check the two baseline paths represented in the compact cache."""
    size_path = [validated[(1e-6, max_size)] for max_size in MAX_SIZES]
    penalty_path = [validated[(c0_value, BASE_FEATURE_COUNT)] for c0_value in C0_VALUES]
    if any(
        right["objective"] + QUERY_PATH_TOLERANCE < left["objective"]
        for left, right in zip(size_path, size_path[1:], strict=False)
    ):
        raise ValueError("objective decreased as max size shrank")
    if any(
        right["loss"] + QUERY_PATH_TOLERANCE < left["loss"]
        for path in (size_path, penalty_path)
        for left, right in zip(path, path[1:], strict=False)
    ):
        raise ValueError("stored query loss path is not monotone")
    if any(
        right["cardinality"] > left["cardinality"]
        for left, right in zip(penalty_path, penalty_path[1:], strict=False)
    ):
        raise ValueError("stored query support increased with c0")
    if penalty_path[-1]["cardinality"] != 0:
        raise ValueError("c0=1 baseline optimum must have zero feature support")


def validate_stored_task(data: dict, model_type: str) -> None:
    """Validate compact bounds, identity, coverage, and exact query answers."""
    case_id = data["case_id"]
    task = data.get("tasks", {}).get(model_type)
    if not isinstance(task, dict):
        raise ValueError(f"{case_id} is missing {model_type}")
    expected_fields = {
        "loss_identity",
        "loss_identity_digest",
        "coefficient_set_lower_bounds",
        "coefficient_set_upper_bounds",
        "queries",
        "coverage",
    }
    if set(task) != expected_fields:
        raise ValueError(f"{case_id} {model_type} compact task fields do not match")
    feature_bounds = MODEL_FEATURE_BOUNDS[model_type]
    expected_bounds = determine_coefficient_bounds(data, feature_bounds)
    for name in ("coefficient_set_lower_bounds", "coefficient_set_upper_bounds"):
        expected = expected_bounds[name]
        if not np.array_equal(task.get(name), expected):
            raise ValueError(f"{case_id} {model_type} {name} does not match")
    strategy = exact_strategy_for(data, model_type) if data.get("redundant_features") else None
    expected_identity = build_loss_identity(data, model_type, expected_bounds, strategy)
    stored_identity = task.get("loss_identity")
    if stored_identity != expected_identity:
        raise ValueError(f"{case_id} {model_type} loss identity does not match")
    stored_digest = task.get("loss_identity_digest")
    if stored_digest != identity_digest(stored_identity):
        raise ValueError(f"{case_id} {model_type} loss identity digest does not match")
    if task.get("coverage") != expected_coverage(data, model_type, expected_bounds):
        raise ValueError(f"{case_id} {model_type} coverage does not match")
    queries = task.get("queries")
    if not isinstance(queries, dict):
        raise ValueError(f"{case_id} {model_type} queries are invalid")
    if data.get("redundant_features"):
        if len(queries) != 1:
            raise ValueError(f"{case_id} {model_type} must store one certifying query")
        key = next(iter(queries))
        if (
            key[0]
            not in (
                PROVISIONAL_REDUNDANCY_PENALTY,
                *ALTERNATIVE_REDUNDANCY_PENALTIES,
            )
            or key[1] != REDUNDANT_FEATURE_COUNT
        ):
            raise ValueError(f"{case_id} {model_type} certifying query is not approved")
        result = validate_stored_query(data, task, key, queries[key])
        representative = queries[key]["representative_rho"][1:]
        if result["cardinality"] == 0 or np.any(
            (representative[:BASE_FEATURE_COUNT] != 0) & (representative[BASE_FEATURE_COUNT:] != 0)
        ):
            raise ValueError(f"{case_id} {model_type} representative does not remove redundancy")
    else:
        if set(queries) != set(BASELINE_QUERY_KEYS):
            raise ValueError(f"{case_id} {model_type} baseline queries do not match")
        validated = {
            key: validate_stored_query(data, task, key, query) for key, query in queries.items()
        }
        validate_compact_query_paths(validated)


def validate_synthetic_oracles(store: dict) -> None:
    """Validate compact records after regenerating each synthetic dataset once."""
    synthetic_oracles = store["synthetic_oracles"]
    if set(synthetic_oracles) != set(ALL_CASE_IDS):
        raise ValueError("stored synthetic cases do not match the admitted registry")
    hydrated = {
        case_id: hydrate_stored_data(synthetic_oracles[case_id]) for case_id in BASELINE_CASE_IDS
    }
    for case_id in REDUNDANT_CASE_IDS:
        record = synthetic_oracles[case_id]
        source_case_id = record.get("generation_spec", {}).get("source_case_id")
        hydrated[case_id] = hydrate_stored_data(record, hydrated.get(source_case_id))
    for case_id in ALL_CASE_IDS:
        data = hydrated[case_id]
        expected_models = (
            set(MODEL_FEATURE_BOUNDS)
            if data.get("redundant_features") != "noisy_duplicates"
            else {"checklist"}
        )
        if set(data.get("tasks", {})) != expected_models:
            raise ValueError(f"{case_id} task profiles do not match")
        for model_type in data["tasks"]:
            validate_stored_task(data, model_type)


def build_profiler_diagnostics() -> dict:
    """Run the independent profiler and duplicate-grouping microchecks."""
    return {
        "passed": True,
        "intercept_profiler": validate_intercept_profiler(),
        "duplicate_grouping": validate_duplicate_grouping(),
    }


def run_preflight(case_id: str | None) -> None:
    """Run diagnostics and report resource estimates without enumerating."""
    diagnostics = build_profiler_diagnostics()
    selected_case_ids = (case_id,) if case_id else ALL_CASE_IDS
    preflights = preflight_cases(selected_case_ids)
    estimate = sum(
        profile["benchmark"]["estimated_enumeration_seconds"]
        for preflight in preflights.values()
        for profile in preflight["profiles"].values()
    )
    print(
        f"preflight passed for {len(preflights)} records; "
        f"estimated enumeration {estimate:.1f}s; diagnostics {diagnostics}",
        flush=True,
    )


def refresh_store_queries(store: dict, case_id: str | None) -> None:
    """Re-enumerate selected query answers while preserving other stored metadata."""
    started = time.perf_counter()
    diagnostics = build_profiler_diagnostics()
    selected_case_ids = (case_id,) if case_id else ALL_CASE_IDS
    generated_records = generate_case_records(preflight_cases(selected_case_ids), diagnostics)
    known_reference_bytes = pickle.dumps(
        store["known_reference_results"], protocol=pickle.HIGHEST_PROTOCOL
    )
    synthetic_oracles = dict(store["synthetic_oracles"])
    for selected_case_id, generated_record in generated_records.items():
        if selected_case_id in synthetic_oracles:
            refreshed_record = dict(synthetic_oracles[selected_case_id])
            refreshed_record["tasks"] = generated_record["tasks"]
        else:
            refreshed_record = generated_record
        synthetic_oracles[selected_case_id] = refreshed_record
    store["synthetic_oracles"] = synthetic_oracles
    if known_reference_bytes != pickle.dumps(
        store["known_reference_results"], protocol=pickle.HIGHEST_PROTOCOL
    ):
        raise AssertionError("known-reference records changed during query refresh")
    validate_synthetic_oracles(store)
    write_store(store)
    print(
        f"re-enumerated queries for {len(selected_case_ids)} records in "
        f"{time.perf_counter() - started:.2f}s"
    )


def regenerate_store(store: dict, case_id: str | None) -> None:
    """Regenerate selected records while protecting every unselected section."""
    diagnostics = build_profiler_diagnostics()
    selected_case_ids = (case_id,) if case_id else ALL_CASE_IDS
    preflights = preflight_cases(selected_case_ids)
    known_reference_bytes = pickle.dumps(
        store["known_reference_results"], protocol=pickle.HIGHEST_PROTOCOL
    )
    unchanged_bytes = {
        stored_case_id: pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
        for stored_case_id, record in store["synthetic_oracles"].items()
        if stored_case_id not in selected_case_ids
    }
    synthetic_oracles = dict(store["synthetic_oracles"])
    synthetic_oracles.update(generate_case_records(preflights, diagnostics))
    store["synthetic_oracles"] = synthetic_oracles
    store["schema_version"] = SCHEMA_VERSION
    if known_reference_bytes != pickle.dumps(
        store["known_reference_results"], protocol=pickle.HIGHEST_PROTOCOL
    ):
        raise AssertionError("known-reference records changed during synthetic regeneration")
    for stored_case_id, expected_bytes in unchanged_bytes.items():
        if expected_bytes != pickle.dumps(
            synthetic_oracles[stored_case_id], protocol=pickle.HIGHEST_PROTOCOL
        ):
            raise AssertionError(f"unselected record changed during regeneration: {stored_case_id}")
    validate_synthetic_oracles(store)
    write_store(store)
    print(f"saved {TEST_CASE_PATH} with {len(synthetic_oracles)} synthetic records")


if __name__ == "__main__":
    arguments = ARGUMENT_PARSER.parse_args()
    if arguments.case and not (
        arguments.regenerate or arguments.refresh_queries or arguments.preflight
    ):
        raise ValueError("--case requires --regenerate, --refresh-queries, or --preflight")
    if arguments.preflight:
        run_preflight(arguments.case)
    else:
        full_regeneration = arguments.regenerate and arguments.case is None
        training_case_store = load_store(allow_legacy_schema=full_regeneration)
        if arguments.refresh_queries:
            refresh_store_queries(training_case_store, arguments.case)
        elif arguments.regenerate:
            regenerate_store(training_case_store, arguments.case)
        else:
            validation_started = time.perf_counter()
            validate_synthetic_oracles(training_case_store)
            print(
                f"validated {len(training_case_store['synthetic_oracles'])} synthetic records in "
                f"{time.perf_counter() - validation_started:.2f}s"
            )

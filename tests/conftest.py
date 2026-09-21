"""Configuration file for pytest."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from generate_training_test_cases import (
    DATA_GENERATION_VERSION,
    REDUNDANT_DATA_GENERATION_VERSION,
    SCHEMA_VERSION,
    SYNTHETIC_RECORD_VERSION,
    identity_digest,
    regenerate_case_data,
)
from utils import generate_random_normal


TRAINING_TEST_CASES_PATH = Path(__file__).parent / "training_test_cases.pkl"
KNOWN_REFERENCE_DATASETS = {"breastcancer", "mammo"}
BASELINE_SYNTHETIC_CASE_IDS = {
    f"{feature_distribution}__{truth_name}"
    for feature_distribution in ("binary", "continuous", "mixed")
    for truth_name in ("integer", "fractional", "shifted_fractional")
}
REDUNDANT_FEATURES = {"duplicates", "noisy_duplicates"}
SYNTHETIC_CASE_IDS = BASELINE_SYNTHETIC_CASE_IDS | {
    f"{case_id}__{redundant_features}"
    for case_id in BASELINE_SYNTHETIC_CASE_IDS
    for redundant_features in REDUNDANT_FEATURES
}
SYNTHETIC_MODEL_TYPES = {"risk-score", "checklist"}
REGENERATION_COMMAND = "uv run python tests/generate_training_test_cases.py --regenerate"
BASELINE_QUERY_KEYS = {
    (1e-6, 4),
    (1e-6, 2),
    (1e-6, 1),
    (1e-6, 0),
    (0.01, 4),
    (0.1, 4),
    (1.0, 4),
}
QUERY_FIELDS = {
    "query_identity",
    "query_identity_digest",
    "representative_rho",
    "pure_logistic_loss",
    "objective",
    "representative_tie_count",
    "proven_complete",
}
REMOVED_SYNTHETIC_DATA_FIELDS = {"X", "y", "rho_true", "source_rho_true"}
REMOVED_ENUMERATION_FIELDS = {
    "feature_coefficients",
    "optimal_intercepts",
    "pure_logistic_losses",
    "supports",
    "intercept_tie_multiplicities",
}

# Report test data, shared by tests/test_report_*.py: a hand-written 8-row training sample and a
# 4-row test sample over three binary features (a, b, c), with fixed coefficients (no solver).
NAMES = ["(Intercept)", "a", "b", "c"]
X_TRAIN = np.array([
    [1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0],
    [0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1],
])
Y_TRAIN = np.array([1, 1, 1, 0, 0, 0, 1, 0])
X_TEST = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1]])
Y_TEST = np.array([1, 0, 1, 0])
RISK_SCORE_RHO = [-2, 2, 1, -1]  # train scores: 3, 2, 2, 1, 0, 1, 0, -1
CHECKLIST_RHO = [-1, 1, 1, 0]
TRAINING = {"objective_value": 0.5, "optimality_gap": float("inf"), "run_time": 1.25}
CONSTRAINTS = {"max_size": 3, "point_range": (-5, 5)}


def pytest_addoption(parser):
    """Register --report-dir for the opt-in browser test's screenshots and HTML."""
    parser.addoption(
        "--report-dir",
        default=None,
        help="Directory for report screenshots and HTML from `pytest -m browser`. Write it as "
        "--report-dir=DIR: with a space, pytest reads an existing DIR as a test path.",
    )


def compute_training_data_digest(X, y, variable_names, outcome_name):
    """Compute the digest stored with a training-data record."""
    digest = hashlib.sha256()
    for array in (X, y):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    digest.update(json.dumps(tuple(variable_names)).encode("utf-8"))
    digest.update(outcome_name.encode("utf-8"))
    return digest.hexdigest()


def validate_synthetic_oracles(synthetic_oracles):
    """Check the compact identities and answers needed by the synthetic tests."""
    if not isinstance(synthetic_oracles, dict) or set(synthetic_oracles) != SYNTHETIC_CASE_IDS:
        raise ValueError("synthetic oracle cases do not match the admitted registry")
    for case_id, record in synthetic_oracles.items():
        if not isinstance(record, dict) or record.get("record_version") != SYNTHETIC_RECORD_VERSION:
            raise ValueError(f"{case_id} has an unsupported record version")
        removed_fields = REMOVED_SYNTHETIC_DATA_FIELDS.intersection(record)
        if removed_fields:
            raise ValueError(
                f"{case_id} still contains removed data fields: {sorted(removed_fields)}"
            )
        generation_spec = record.get("generation_spec")
        data_digest = record.get("data_digest")
        data_identity = record.get("data_identity")
        redundant_features = (
            generation_spec.get("redundant_features") if isinstance(generation_spec, dict) else None
        )
        n_features = 4 if redundant_features is None else 8
        variable_names = record.get("variable_names")
        outcome_name = record.get("outcome_name")
        if (
            record.get("case_id") != case_id
            or not isinstance(generation_spec, dict)
            or generation_spec.get("case_id") != case_id
            or not isinstance(data_digest, str)
            or len(data_digest) != 64
            or data_identity != {"generation_spec": generation_spec, "data_digest": data_digest}
            or not isinstance(variable_names, tuple)
            or len(variable_names) != n_features
            or generation_spec.get("variable_names") != variable_names
            or outcome_name != "y"
            or generation_spec.get("outcome_name") != outcome_name
        ):
            raise ValueError(f"{case_id} generation metadata is invalid")

        if redundant_features is None:
            if (
                case_id not in BASELINE_SYNTHETIC_CASE_IDS
                or generation_spec.get("generation_version") != DATA_GENERATION_VERSION
            ):
                raise ValueError(f"{case_id} baseline metadata does not match")
        else:
            source_case_id = generation_spec.get("source_case_id")
            source = synthetic_oracles.get(source_case_id)
            if (
                generation_spec.get("generation_version") != REDUNDANT_DATA_GENERATION_VERSION
                or redundant_features not in REDUNDANT_FEATURES
                or source_case_id not in BASELINE_SYNTHETIC_CASE_IDS
                or case_id != f"{source_case_id}__{redundant_features}"
                or not isinstance(source, dict)
                or generation_spec.get("source_data_digest") != source.get("data_digest")
            ):
                raise ValueError(f"{case_id} redundant-feature metadata does not match")

        continuous_reference = record.get("continuous_reference")
        has_fractional_truth = "fractional" in case_id
        if not has_fractional_truth and continuous_reference is not None:
            raise ValueError(f"{case_id} should not have a continuous reference")
        if has_fractional_truth:
            continuous_rho = (
                continuous_reference.get("rho") if isinstance(continuous_reference, dict) else None
            )
            if (
                not isinstance(continuous_reference, dict)
                or continuous_reference.get("converged") is not True
                or not isinstance(continuous_rho, np.ndarray)
                or continuous_rho.shape != (n_features + 1,)
                or not np.isfinite(continuous_rho).all()
                or not np.isfinite(continuous_reference.get("pure_logistic_loss", np.nan))
                or not np.isfinite(continuous_reference.get("gradient_infinity_norm", np.nan))
                or continuous_reference["gradient_infinity_norm"] > 1e-6
                or continuous_reference.get("identity", {}).get("data_digest") != data_digest
            ):
                raise ValueError(f"{case_id} continuous reference is invalid")

        tasks = record.get("tasks")
        expected_models = (
            SYNTHETIC_MODEL_TYPES if redundant_features in (None, "duplicates") else {"checklist"}
        )
        if not isinstance(tasks, dict) or set(tasks) != expected_models:
            raise ValueError(f"{case_id} task profiles do not match")
        for model_type, task in tasks.items():
            if not isinstance(task, dict):
                raise ValueError(f"{case_id} {model_type} oracle is invalid")
            removed_fields = REMOVED_ENUMERATION_FIELDS.intersection(task)
            loss_identity = task.get("loss_identity")
            if (
                removed_fields
                or not isinstance(loss_identity, dict)
                or loss_identity.get("algorithm_version") != "integer-intercept-profile-v1"
                or loss_identity.get("data_identity") != data_identity
                or loss_identity.get("data_digest") != data_digest
                or loss_identity.get("model_type") != model_type
                or not task.get("coverage", {}).get("proven_complete")
                or not isinstance(task.get("queries"), dict)
            ):
                raise ValueError(f"{case_id} {model_type} oracle identity is invalid")
            queries = task["queries"]
            if redundant_features is None and set(queries) != BASELINE_QUERY_KEYS:
                raise ValueError(f"{case_id} {model_type} query keys are invalid")
            if redundant_features is not None:
                if len(queries) != 1:
                    raise ValueError(f"{case_id} {model_type} query keys are invalid")
                c0_value, max_size = next(iter(queries))
                if not np.isfinite(c0_value) or c0_value <= 0.0 or max_size != n_features:
                    raise ValueError(f"{case_id} {model_type} query keys are invalid")
            for query_key, query in queries.items():
                rho = query.get("representative_rho") if isinstance(query, dict) else None
                query_identity = query.get("query_identity") if isinstance(query, dict) else None
                expected_query_identity = {
                    "loss_identity_digest": task.get("loss_identity_digest"),
                    "c0": query_key[0],
                    "max_size": query_key[1],
                }
                if (
                    not isinstance(query, dict)
                    or set(query) != QUERY_FIELDS
                    or query_identity != expected_query_identity
                    or query.get("query_identity_digest")
                    != identity_digest(expected_query_identity)
                    or not isinstance(rho, np.ndarray)
                    or rho.shape != (n_features + 1,)
                    or not np.isfinite(rho).all()
                    or not np.isfinite(query.get("pure_logistic_loss", np.nan))
                    or not np.isfinite(query.get("objective", np.nan))
                    or not isinstance(query.get("representative_tie_count"), (int, np.integer))
                    or query["representative_tie_count"] < 1
                    or query.get("proven_complete") is not True
                ):
                    raise ValueError(f"{case_id} {model_type} query {query_key} is invalid")


def regenerate_and_verify_synthetic_data(case_id, record, source=None):
    """Regenerate one dataset, verify its stored identity, and inject only X/y."""
    generated = regenerate_case_data(record, source=source)
    X = generated.get("X")
    y = generated.get("y")
    variable_names = generated.get("variable_names")
    outcome_name = generated.get("outcome_name")
    n_features = len(record["variable_names"])
    if (
        not isinstance(X, np.ndarray)
        or X.shape != (10_000, n_features)
        or not np.isfinite(X).all()
        or not isinstance(y, np.ndarray)
        or y.shape != (10_000,)
        or not np.isin(y, (0, 1)).all()
        or variable_names != record["variable_names"]
        or outcome_name != record["outcome_name"]
    ):
        raise ValueError(f"{case_id} regenerated training data is invalid")
    regenerated_digest = compute_training_data_digest(X, y, variable_names, outcome_name)
    if (
        generated.get("case_id") != case_id
        or generated.get("data_digest") != regenerated_digest
        or regenerated_digest != record["data_digest"]
        or generated.get("data_identity") != record["data_identity"]
    ):
        raise ValueError(f"{case_id} regenerated data digest does not match the stored digest")
    record["X"] = X
    record["y"] = y


def validate_and_hydrate_synthetic_oracles(store):
    """Validate compact oracles and regenerate each synthetic dataset once."""
    if store.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"all synthetic cases use unsupported schema version {store.get('schema_version')}"
        )
    synthetic_oracles = store.get("synthetic_oracles")
    if not isinstance(synthetic_oracles, dict) or set(synthetic_oracles) != SYNTHETIC_CASE_IDS:
        raise ValueError("all synthetic cases do not match the admitted registry")
    validate_synthetic_oracles(synthetic_oracles)
    for case_id in sorted(BASELINE_SYNTHETIC_CASE_IDS):
        regenerate_and_verify_synthetic_data(case_id, synthetic_oracles[case_id])
    for case_id in sorted(SYNTHETIC_CASE_IDS - BASELINE_SYNTHETIC_CASE_IDS):
        record = synthetic_oracles[case_id]
        source = synthetic_oracles[record["generation_spec"]["source_case_id"]]
        regenerate_and_verify_synthetic_data(case_id, record, source=source)


def validate_known_reference_results(store):
    """Reject a malformed shared store or non-regenerable real-data records."""
    if not isinstance(store, dict):
        raise ValueError("top level is not a dictionary")
    known_results = store.get("known_reference_results")
    if not isinstance(known_results, dict):
        raise ValueError("known_reference_results is not a dictionary")
    missing_datasets = KNOWN_REFERENCE_DATASETS.difference(known_results)
    if missing_datasets:
        raise ValueError(f"missing known-reference records: {sorted(missing_datasets)}")

    for dataset_name in KNOWN_REFERENCE_DATASETS:
        record = known_results[dataset_name]
        if not isinstance(record, dict) or record.get("record_version") != 1:
            raise ValueError(f"{dataset_name} has an unsupported record version")
        X = record.get("X")
        y = record.get("y")
        variable_names = record.get("variable_names")
        outcome_name = record.get("outcome_name")
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError(f"{dataset_name} X is not a two-dimensional array")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise ValueError(f"{dataset_name} y is not a one-dimensional array")
        if X.shape[0] != y.shape[0] or X.shape[1] != len(variable_names):
            raise ValueError(f"{dataset_name} data dimensions are inconsistent")
        expected = record.get("expected")
        if not isinstance(expected, dict) or len(expected.get("predictions", ())) != len(y):
            raise ValueError(f"{dataset_name} saved expectations are malformed")
        objective_value = expected.get("objective_value")
        if (
            isinstance(objective_value, bool)
            or not isinstance(objective_value, (int, float, np.number))
            or not np.isfinite(objective_value)
        ):
            raise ValueError(f"{dataset_name} saved objective is not a finite scalar")
        actual_digest = compute_training_data_digest(X, y, variable_names, outcome_name)
        if record.get("data_digest") != actual_digest:
            raise ValueError(f"{dataset_name} data digest does not match its arrays")


@pytest.fixture(scope="session")
def training_test_cases():
    """Load the checked-in training inputs and saved reference results once."""
    restore_command = "git restore tests/training_test_cases.pkl"
    if not TRAINING_TEST_CASES_PATH.exists():
        pytest.fail(
            f"Missing {TRAINING_TEST_CASES_PATH}. Restore the checked-in artifact with "
            f"`{restore_command}`; known expectations cannot be recomputed from the solver.",
            pytrace=False,
        )
    try:
        with TRAINING_TEST_CASES_PATH.open("rb") as file_handle:
            store = pickle.load(file_handle)
        validate_known_reference_results(store)
    except Exception as error:
        pytest.fail(
            f"Stale or malformed {TRAINING_TEST_CASES_PATH}: {error}. Restore the checked-in "
            f"artifact with `{restore_command}`.",
            pytrace=False,
        )
    try:
        validate_and_hydrate_synthetic_oracles(store)
    except Exception as error:
        pytest.fail(
            f"Stale or malformed synthetic records in {TRAINING_TEST_CASES_PATH}: {error}. "
            f"Regenerate them with `{REGENERATION_COMMAND}`.",
            pytrace=False,
        )
    return store


@pytest.fixture(scope="module")
def generated_normal_data():
    """Generate data with a solution."""

    # Size of problem
    n_columns = 12
    n_rows = 200
    n_targets = 4
    n_iters = 25

    random_state = np.random.get_state()
    try:
        X = np.zeros((n_iters, n_rows, n_columns))
        rho_true = np.zeros((n_iters, n_columns))
        for i, seed in enumerate(range(n_iters)):
            # Simulate data
            _data, _rho_true = generate_random_normal(n_rows, n_columns, n_targets, seed)

            # Track features and true rho
            X[i] = _data["X"]
            rho_true[i] = _rho_true
    finally:
        np.random.set_state(random_state)

    # Labels
    y = np.ravel(_data["y"])

    rho = np.ones(n_columns)

    Z = X * y[:, None]

    names = ["var_" + str(i).zfill(2) for i in range(n_columns)]

    yield {
        "X": X,
        "y": y,
        "Z": Z,
        "rho": rho,
        "rho_true": rho_true,
        "variable_names": names,
        "outcome_name": _data["outcome_name"],
    }

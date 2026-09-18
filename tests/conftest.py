"""Configuration file for pytest."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from utils import generate_random_normal


TRAINING_TEST_CASES_PATH = Path(__file__).parent / "training_test_cases.pkl"
TRAINING_TEST_CASES_SCHEMA_VERSION = 1
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
    """Check the identity and inputs needed by the synthetic solver tests."""
    if not isinstance(synthetic_oracles, dict) or set(synthetic_oracles) != SYNTHETIC_CASE_IDS:
        raise ValueError("synthetic oracle cases do not match the admitted registry")
    for case_id, record in synthetic_oracles.items():
        if not isinstance(record, dict) or record.get("record_version") != 1:
            raise ValueError(f"{case_id} has an unsupported record version")
        redundant_features = record.get("redundant_features")
        n_features = 4 if redundant_features is None else 8
        X = record.get("X")
        y = record.get("y")
        variable_names = record.get("variable_names")
        outcome_name = record.get("outcome_name")
        if (
            not isinstance(X, np.ndarray)
            or X.shape != (10_000, n_features)
            or not np.isfinite(X).all()
            or not isinstance(y, np.ndarray)
            or y.shape != (10_000,)
            or not np.isin(y, (0, 1)).all()
            or not isinstance(variable_names, tuple)
            or len(variable_names) != n_features
            or outcome_name != "y"
        ):
            raise ValueError(f"{case_id} training data or metadata is invalid")
        data_digest = compute_training_data_digest(X, y, variable_names, outcome_name)
        data_identity = record.get("data_identity")
        feature_distribution, _ = case_id.split("__", maxsplit=1)
        if (
            record.get("case_id") != case_id
            or record.get("data_digest") != data_digest
            or not isinstance(data_identity, dict)
            or data_identity.get("case_id") != case_id
            or data_identity.get("feature_distribution") != feature_distribution
            or data_identity.get("n_samples") != 10_000
            or data_identity.get("data_digest") != data_digest
        ):
            raise ValueError(f"{case_id} data identity or digest does not match")

        if redundant_features is None:
            if (
                case_id not in BASELINE_SYNTHETIC_CASE_IDS
                or data_identity.get("generation_version") != "independent-bernoulli-logistic-v1"
                or data_identity.get("seed") != 0
            ):
                raise ValueError(f"{case_id} baseline metadata does not match")
        else:
            source_case_id = record.get("source_case_id")
            source = synthetic_oracles.get(source_case_id)
            expected_perturbation = (
                {"kind": "exact_copy"}
                if redundant_features == "duplicates"
                else {
                    "kind": "distribution_specific_noise",
                    "seed_sequence": [0, 1, 2],
                    "binary_bit_flip_probability": 0.05,
                    "continuous_gaussian_standard_deviation": 0.25,
                }
            )
            if (
                redundant_features not in REDUNDANT_FEATURES
                or source_case_id not in BASELINE_SYNTHETIC_CASE_IDS
                or case_id != f"{source_case_id}__{redundant_features}"
                or not isinstance(source, dict)
                or data_identity.get("generation_version") != "redundant-features-v1"
                or data_identity.get("source_case_id") != source_case_id
                or data_identity.get("source_data_digest") != source.get("data_digest")
                or data_identity.get("redundant_features") != redundant_features
                or data_identity.get("perturbation") != expected_perturbation
                or data_identity.get("labels_from_clean_features") is not True
                or not np.array_equal(y, source.get("y"))
                or not np.array_equal(record.get("source_rho_true"), source.get("rho_true"))
            ):
                raise ValueError(f"{case_id} redundant-feature metadata does not match")

        continuous_reference = record.get("continuous_reference")
        has_fractional_truth = "fractional" in case_id
        if not has_fractional_truth and continuous_reference is not None:
            raise ValueError(f"{case_id} should not have a continuous reference")
        if has_fractional_truth:
            if not isinstance(continuous_reference, dict):
                raise ValueError(f"{case_id} is missing its continuous reference")
            continuous_rho = continuous_reference.get("rho")
            continuous_loss = continuous_reference.get("pure_logistic_loss", np.nan)
            gradient_norm = continuous_reference.get("gradient_infinity_norm", np.nan)
            if (
                continuous_reference.get("converged") is not True
                or not isinstance(continuous_rho, np.ndarray)
                or continuous_rho.shape != (n_features + 1,)
                or not np.isfinite(continuous_rho).all()
                or not np.isfinite(continuous_loss)
                or not np.isfinite(gradient_norm)
                or gradient_norm > 1e-6
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
            loss_identity = task.get("loss_identity") if isinstance(task, dict) else None
            if (
                not isinstance(loss_identity, dict)
                or loss_identity.get("algorithm_version") != "integer-intercept-profile-v1"
                or loss_identity.get("data_identity") != data_identity
                or loss_identity.get("data_digest") != data_digest
                or loss_identity.get("model_type") != model_type
                or not task.get("coverage", {}).get("proven_complete")
                or not isinstance(task.get("queries"), dict)
            ):
                raise ValueError(f"{case_id} {model_type} oracle identity is invalid")
            if redundant_features is not None:
                evidence = task.get("redundancy_evidence")
                certificate = (
                    evidence.get("certifying_penalty") if isinstance(evidence, dict) else None
                )
                query = evidence.get("certifying_query") if isinstance(evidence, dict) else None
                query_identity = query.get("query_identity") if isinstance(query, dict) else None
                if (
                    not isinstance(certificate, dict)
                    or not isinstance(query_identity, dict)
                    or query_identity.get("loss_identity_digest")
                    != task.get("loss_identity_digest")
                    or query_identity.get("c0") != certificate.get("c0")
                    or query_identity.get("max_size") != certificate.get("max_size")
                    or query.get("proven_complete") is not True
                ):
                    raise ValueError(f"{case_id} {model_type} redundancy evidence is invalid")


def validate_training_test_cases(store):
    """Reject a stale or malformed shared training test-case store."""
    if not isinstance(store, dict):
        raise ValueError("top level is not a dictionary")
    if store.get("schema_version") != TRAINING_TEST_CASES_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {TRAINING_TEST_CASES_SCHEMA_VERSION}")
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
        validate_training_test_cases(store)
    except Exception as error:
        pytest.fail(
            f"Stale or malformed {TRAINING_TEST_CASES_PATH}: {error}. Restore the checked-in "
            f"artifact with `{restore_command}`.",
            pytrace=False,
        )
    try:
        validate_synthetic_oracles(store.get("synthetic_oracles"))
    except Exception as error:
        regeneration_command = "uv run python tests/generate_training_test_cases.py --regenerate"
        pytest.fail(
            f"Stale or malformed synthetic records in {TRAINING_TEST_CASES_PATH}: {error}. "
            f"Regenerate them with `{regeneration_command}`.",
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

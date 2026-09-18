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


def validate_training_test_cases(store):
    """Reject a stale or malformed shared training test-case store."""
    if not isinstance(store, dict):
        raise ValueError("top level is not a dictionary")
    if store.get("schema_version") != TRAINING_TEST_CASES_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {TRAINING_TEST_CASES_SCHEMA_VERSION}")
    if not isinstance(store.get("synthetic_oracles"), dict):
        raise ValueError("synthetic_oracles is not a dictionary")
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
    return store


@pytest.fixture(scope="module")
def generated_normal_data():
    """Generate data with a solution."""

    # Size of problem
    n_columns = 12
    n_rows = 200
    n_targets = 4
    n_iters = 25

    X = np.zeros((n_iters, n_rows, n_columns))
    rho_true = np.zeros((n_iters, n_columns))
    for i, seed in enumerate(range(n_iters)):
        # Simulate data
        _data, _rho_true = generate_random_normal(n_rows, n_columns, n_targets, seed)

        # Track features and true rho
        X[i] = _data["X"]
        rho_true[i] = _rho_true

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

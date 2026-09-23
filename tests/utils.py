"""Shared test data: simulated training data and the report sample."""

import numpy as np

# Report test data, used by tests/test_model_report.py: a hand-written 8-row training sample and a
# 4-row test sample over three binary features (a, b, c), with fixed coefficients (no solver).
NAMES = ["(Intercept)", "a", "b", "c"]
X_TRAIN = np.array([
    [1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0],
    [0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1],
])
Y_TRAIN = np.array([1, 1, 1, 0, 0, 0, 1, 0])
X_TEST = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1]])
Y_TEST = np.array([1, 0, 1, 0])
SAMPLES = {"train": (X_TRAIN, Y_TRAIN), "test": (X_TEST, Y_TEST)}
RISK_SCORE_RHO = [-2, 2, 1, -1]  # train scores: 3, 2, 2, 1, 0, 1, 0, -1
CHECKLIST_RHO = [-1, 1, 1, 0]
TRAINING = {"objective_value": 0.5, "optimality_gap": float("inf"), "run_time": 1.25}
CONSTRAINTS = {"max_size": 3, "point_range": (-5, 5)}


def generate_random_normal(n_rows, n_columns, n_targets, seed):
    """Simulated data from random normals.

    Parameters
    ----------
    n_rows : int
        Number of observations.
    n_cols : int
        Number of features.
    n_targets : int
        Number of features that will not be noise.
    seed : int
        Random seed.

    Returns
    -------
    data : dict
        Contains features and labels.
    rho : 1d array
        True weights.
    """
    np.random.seed(seed)

    # Initialize arrays
    X = np.zeros((n_rows, n_columns), dtype=np.float64)
    y = np.ones((n_rows, 1), dtype=np.int32)
    y[n_rows//2:, 0] = -1

    # Ranomly select target columns
    inds = np.arange(1, n_columns)
    selected = np.random.choice(inds, n_targets, replace=False)
    selected = np.sort(selected)

    # Simulate random int normals with varying overlap
    stdevs = iter(np.linspace(100, 50, n_targets))

    for ind in range(0, n_columns):

        if ind in selected:
            # Class A
            X[:, ind] = np.random.normal(100, next(stdevs), n_rows).astype(np.int32)
            # Class B
            X[:n_rows//2, ind] *= -1
        else:
            # Noise
            X[:, ind] = np.random.normal(0, 100, n_rows).astype(np.int32)

    # Variale names
    variable_names = ['var_' + str(i).zfill(2) for i in range(n_columns)]

    # Data
    data = {}
    data['X'] = X
    data['y'] = y
    data['variable_names'] = variable_names
    data['outcome_name'] = '1'

   # True weights
    rho = np.zeros(n_columns)
    rho[selected] = -1

    # Get predictions
    true_preds = np.count_nonzero(np.sign(np.dot(X, rho)) == y[:, 0])

    # Compute accuracy
    acc = true_preds / len(y)

    # Distributions of target features between the two classes overlap.
    #   Sometimes outliers in the tails of the distributions will be miss-classified.
    assert acc > .95

    return data, rho

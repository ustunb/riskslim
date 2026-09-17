"""Configuration file for pytest."""

import pytest

import numpy as np

from .utils import generate_random_normal


@pytest.fixture(scope='module')
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
        X[i] = _data['X']
        rho_true[i] = _rho_true

    # Labels
    y = np.ravel(_data['y'])

    rho = np.ones(n_columns)

    Z = X * y[:, None]

    names = ['var_' + str(i).zfill(2) for i in range(n_columns)]

    yield {'X':X, 'y':y, 'Z':Z, 'rho':rho, 'rho_true':rho_true, 'variable_names':names,
           'outcome_name': _data['outcome_name']}

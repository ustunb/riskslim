"""Configuration file for pytest."""

import os
import pytest

import numpy as np

from riskslim.bounds import get_score_bounds
import riskslim.loss_functions.lookup_log_loss as lookup

from .utils import generate_random_normal


@pytest.fixture(scope='module')
def generated_data():

    np.random.seed(seed=0)
    n_rows = 1000000
    n_cols = 20
    rho_ub = 100
    rho_lb = -100

    def generate_binary_data(n_rows=1000000, n_cols=20):
        X = np.random.randint(low=0, high=2, size=(n_rows, n_cols))
        y = np.random.randint(low=0, high=2, size=(n_rows, 1))
        pos_ind = y == 1
        y[~pos_ind] = -1
        return X, y

    def generate_integer_model(n_cols=20, rho_ub=100, rho_lb=-100, sparse_pct=0.5):
        rho = np.random.randint(low=rho_lb, high=rho_ub, size=n_cols)
        rho = np.require(rho, dtype=np.float64, requirements=['F'])
        nnz_count = int(sparse_pct * np.floor(n_cols / 2))
        set_to_zero = np.random.choice(range(0, n_cols), size=nnz_count, replace=False)
        rho[set_to_zero] = 0.0
        return rho

    # Initialize data matrix X and label vector y
    X, y = generate_binary_data(n_rows, n_cols)
    X[:, 0] = 1.

    Z = X * y
    Z = np.require(Z, requirements=['F'], dtype=np.float64)

    rho = generate_integer_model(n_cols, rho_ub, rho_lb)

    L0_reg_ind = np.ones(n_cols, dtype='bool')
    L0_reg_ind[0] = False

    Z_min = np.min(Z, axis=0)
    Z_max = np.max(Z, axis=0)

    # Setup weights
    weights = np.ones(len(X))

    # Create lookup table
    min_score, max_score = get_score_bounds(Z_min, Z_max, rho_lb, rho_ub, max_size=n_cols)

    loss_value_tbl, prob_value_tbl, loss_tbl_offset = \
            lookup.get_loss_value_and_prob_tables(min_score, max_score)

    loss_tbl_offset = int(loss_tbl_offset)

    generated = {
        'X': X,
        'y': y,
        'Z': Z,
        'rho': rho,
        'min_score': min_score,
        'max_score': max_score,
        'weights': weights,
        'loss_value_tbl': loss_value_tbl,
        'loss_tbl_offset': loss_tbl_offset,
        'prob_value_tbl': prob_value_tbl
    }

    yield generated


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
    y = _data['y']

    rho = np.ones(n_columns)

    Z = X * y

    names = ['var_' + str(i).zfill(2) for i in range(n_columns)]

    yield {'X':X, 'y':y, 'Z':Z, 'rho':rho, 'rho_true':rho_true, 'variable_names':names}

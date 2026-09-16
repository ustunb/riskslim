"""Test data objects."""

import numpy as np
import pandas as pd
from riskslim.data import ClassificationDataset
from riskslim.utils import Stats
from riskslim.bounds import Bounds


def test_ClassificationDataset():

    n_obs = 10
    n_variables = 100

    X = np.random.rand(n_obs, n_variables)
    y = np.random.choice([1, -1], n_obs).reshape(-1, 1)

    variable_names = ['var_' + str(i) for i in range(n_variables)]

    ds = ClassificationDataset(X, y, variable_names, outcome_name='outcome')
    assert np.all(ds.X[:, 1:] == X)
    assert np.all(ds.y == y)
    assert ds.variable_names[1:] == variable_names
    assert ds.sample_weights is None
    assert ds.outcome_name == 'outcome'
    assert isinstance(ds.df, pd.DataFrame)
    assert isinstance(ds.__str__(), str)
    assert isinstance(ds.__repr__(), str)

    ds.__check_rep__()


def test_bounds():

    bounds = Bounds(objval_min=0., objval_max=1., loss_min=0., loss_max=1.,
                    min_size=1, max_size=10)

    assert bounds.objval_min == bounds.loss_min == 0.
    assert bounds.objval_max == bounds.loss_max == 1.
    assert bounds.min_size == 1 and bounds.max_size == 10

    bounds = bounds.asdict()
    assert isinstance(bounds, dict)



def test_stats():

    incumbent = np.zeros(10)

    stats = Stats(incumbent)

    stats_dict = stats.asdict()
    assert isinstance(stats_dict, dict)

    stats = Stats(incumbent)

    for k in stats_dict.keys():
        val = getattr(stats, k)

        if k == 'incumbent':
            assert np.all(val == incumbent)
        elif k == 'bounds':
            assert isinstance(val, Bounds)
        else:
            assert hasattr(stats, k)
            assert isinstance(val, (int, float))
            assert (val == 0) or not np.isfinite(val)

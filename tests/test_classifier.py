"""RiskSLIMClassifier API behavior: what a user sees when building, printing, and saving it."""

import pickle

import numpy as np
import pytest

from riskslim import CoefficientSet, RiskSLIMClassifier


def make_binary_data():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(80, 3)).astype(float)
    y = (X[:, 0] + X[:, 1] + rng.random(80) > 1.2).astype(int)
    return X, y


def fit_classifier(max_size=2):
    X, y = make_binary_data()
    clf = RiskSLIMClassifier(max_size=max_size, verbose=False, max_runtime=10, cplex_randomseed=0)
    return clf.fit(X, y), X


@pytest.mark.parametrize('init_coef', [True, False])
def test_init_stores_parameters(init_coef):
    """Test RiskSLIMClassifier initialization."""
    variable_names = ['variable_' + str(i) for i in range(10)]

    coef_set = CoefficientSet(variable_names) if init_coef else None

    max_size=10

    rs = RiskSLIMClassifier(coef_set=coef_set, max_size=max_size)

    assert rs.max_size == max_size
    assert rs.optimizer is None
    assert rs.coef_set is coef_set
    assert rs.variable_names is None
    assert rs.c0_value == 1e-6
    assert not rs.fitted


@pytest.mark.parametrize('max_size, expected_row', [
    (0, 'NO VARIABLES: SCORE IS 0'),
    (2, 'ADD POINTS FROM ROWS 1 to'),
])
def test_print_shows_score_table(max_size, expected_row):
    clf, _ = fit_classifier(max_size)

    table = str(clf)

    assert table.startswith('+')
    assert expected_row in table
    assert repr(clf) == table


def test_saved_classifier_loads_prints_and_predicts_the_same(tmp_path):
    clf, X = fit_classifier()
    model_file = tmp_path / 'model.pkl'

    model_file.write_bytes(pickle.dumps(clf))
    loaded = pickle.loads(model_file.read_bytes())

    assert str(loaded) == str(clf)
    np.testing.assert_array_equal(loaded.predict(X), clf.predict(X))

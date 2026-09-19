"""RiskSLIMClassifier API behavior: what a user sees when building, printing, and saving it."""

import pickle

import numpy as np
import pytest

from riskslim import RiskSLIMClassifier


def make_binary_data():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(80, 3)).astype(float)
    y = (X[:, 0] + X[:, 1] + rng.random(80) > 1.2).astype(int)
    return X, y


def fit_classifier(max_size=2):
    X, y = make_binary_data()
    clf = RiskSLIMClassifier(max_size=max_size, verbose=False, max_runtime=10, cplex_randomseed=0)
    return clf.fit(X, y), X


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

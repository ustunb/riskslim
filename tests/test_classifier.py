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
    clf = RiskSLIMClassifier(max_size=max_size, verbose=False, max_runtime=10, cplex_randomseed=0)
    return clf.fit(*make_binary_data())


@pytest.mark.parametrize('init_coef', [True, False])
def test_init_stores_parameters(init_coef):
    coef_set = CoefficientSet([f'variable_{i}' for i in range(10)]) if init_coef else None

    rs = RiskSLIMClassifier(coef_set=coef_set, max_size=10)

    assert rs.max_size == 10
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
    clf = fit_classifier(max_size)

    table = str(clf)

    assert table.startswith('+')
    assert expected_row in table
    assert repr(clf) == table


def test_saved_classifier_loads_prints_and_predicts_the_same(tmp_path):
    clf = fit_classifier()
    X, _ = make_binary_data()
    model_file = tmp_path / 'model.pkl'

    model_file.write_bytes(pickle.dumps(clf))
    loaded = pickle.loads(model_file.read_bytes())

    assert str(loaded) == str(clf)
    np.testing.assert_array_equal(loaded.predict(X), clf.predict(X))


def test_report_shows_training_data_and_labelled_test_sample():
    X, y = make_binary_data()
    labels = np.where(y == 1, 'yes', 'no')
    clf = RiskSLIMClassifier(max_size=2, verbose=False, max_runtime=10, cplex_randomseed=0)
    clf.fit(X, labels)

    data = clf.report(X[:40], labels[:40]).data

    assert data['samples'] == ['Training', 'Test']
    rows = {row['key']: row['values'] for row in data['summary']['rows']}
    assert rows['n'] == ['80', '40']
    assert rows['outcome_rate'] == ['62.5%', '62.5%']
    assert 'run_time' in rows


def test_intercept_bound_counts_only_max_size_items():
    # 3 binary items with |coef| <= 5: one nonzero item scores at most 5, so the intercept needs at most 5 + 1
    clf = fit_classifier(max_size=1)

    assert clf.coef_set_.ub[0] == 6
    assert clf.coef_set_.lb[0] == -6

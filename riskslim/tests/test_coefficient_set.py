"""Test coefficient sets."""

import pytest
import numpy as np
from riskslim.coefficient_set import CoefficientSet, _CoefficientElement
from riskslim.bounds import get_score_bounds
from riskslim.data import ClassificationDataset


@pytest.mark.parametrize('lb', [-5, [-5]*11])
@pytest.mark.parametrize('ub', [5, [5]*11])
def test_coefficientset_init(lb, ub):

    variable_names = ["var_" + str(i) for i in range(10)]
    variable_names.insert(0, '(Intercept)')

    cs = CoefficientSet(variable_names, lb=lb, ub=ub)

    assert cs._default_print_flag is True
    assert cs._initialized is True

    assert cs._variable_names == variable_names


    if isinstance(lb, list):
        assert isinstance(cs.lb, np.ndarray)
        assert len(lb) == len(cs.lb)

    if isinstance(ub, list):
        assert isinstance(cs.ub, np.ndarray)
        assert len(ub) == len(cs.ub)

    assert len(cs._coef_elements) == len(variable_names)


# @pytest.mark.parametrize('has_intercept', [
#     True, pytest.param(False, marks=pytest.mark.xfail(raises=ValueError))
# ])
@pytest.mark.parametrize('max_size', [10, None])
def test_coefficientset_update_intercept_bounds(
        generated_normal_data, max_size
    ):
    # Dataset
    X = generated_normal_data['X'][0]
    y = generated_normal_data['y']
    variable_names = generated_normal_data['variable_names']
    data = ClassificationDataset(X, y, variable_names=variable_names, outcome_name='outcome')

    cs = CoefficientSet(data.variable_names)

    cs.update_intercept_bounds(data.X, data.y, 1, max_size=max_size)


def test_coefficientset_tabulate():

    variable_names = ["var_" + str(i) for i in range(10)]
    variable_names.insert(0, '(Intercept)')

    cs = CoefficientSet(variable_names)
    table = cs.tabulate()

    assert isinstance(table, str)

    for key in ["variable_name", "vtype", "sign", "lb", "ub", "c0"]:
        assert key  in table


def test_coefficientset_properties():
    """Test properties and dunder methods."""

    variable_names = ["var_" + str(i) for i in range(10)]
    variable_names.insert(0, '(Intercept)')

    cs = CoefficientSet(variable_names)
    assert cs.variable_names == variable_names

    # Index
    cs.index('var_0')

    with pytest.raises(ValueError):
        cs.index('')

    # Penalize
    indices = cs.penalized_indices()
    assert isinstance(indices, np.ndarray)
    assert len(indices) == len(variable_names)

    # Setting
    cs = CoefficientSet(variable_names)
    cs.ub = [0.] * len(variable_names)
    assert cs._ub == [0] * len(variable_names)
    cs._ub = [0.] * len(variable_names)
    assert all(cs._ub == cs.ub)

    # Get items
    cs = CoefficientSet(variable_names)
    assert cs[0] == cs['(Intercept)']
    assert cs[1] == cs['var_0']

    with pytest.raises(KeyError):
        # Expects int or string index
        cs[0.]

    # Set items
    cs[0] = _CoefficientElement('(Intercept)', ub=-1.)
    assert cs[0]._ub == -1
    cs['(Intercept)'] =  _CoefficientElement('(Intercept)', ub=-2.)
    assert cs[0]._ub == -2

    with pytest.raises(KeyError):
        # Expects int or string
        cs[0.] = None

    # Dunders
    assert len(cs) == len(variable_names)
    assert str(cs) == cs.tabulate()


@pytest.mark.parametrize('lb',
    [-1, -1., '-1', [-1], [-1]*11, np.array([-1]), np.array([-1]*11),
     pytest.param(None, marks=pytest.mark.xfail(raises=ValueError))]
)
def test_coefficientset_expand_values(lb):
    variable_names = ["var_" + str(i) for i in range(10)]
    variable_names.insert(0, '(Intercept)')

    cs = CoefficientSet(variable_names)
    values = cs._expand_values(lb)
    assert len(values) == 11

    if isinstance(lb, (list, np.ndarray)):
        with pytest.raises(ValueError):
            cs = CoefficientSet(variable_names)
            values = cs._expand_values(lb[1:])


def test_coefficientelement():
    """Test coefficient element."""
    test = _CoefficientElement('test')
    assert test._is_integer(1)
    assert not test._is_integer('1')

    test.vtype = 'C'

    test.ub = [1.]
    test.ub = 1.

    test.lb = [.1]
    test.lb = .1

    test.c0 = np.nan
    test.c0 = 1e-3

    assert test.c0 == 1e-3
    assert test._ub == 1.
    assert test.lb == .1
    assert test.sign == 1

    # Flip sign
    test = _CoefficientElement('test')

    test.ub = [-.1]
    test.ub = -.1

    test.lb = [-1]
    test.lb = -1

    assert test.sign == -1

    test = _CoefficientElement('test')
    test.sign = 1
    test.sign = -1

    assert str(test) == repr(test)


@pytest.mark.parametrize('use_L0', [True, False])
def test_get_score_bounds(use_L0):

    Z_min = np.ones(10)
    Z_max = np.ones(10) + 9
    rho_lb = np.repeat(-5, 10)
    rho_ub = np.repeat(-5, 10)

    L0_reg_ind = None
    max_size = None
    if use_L0:
        L0_reg_ind = np.ones(10).astype(int)
        L0_max = 1

    s_min, s_max = get_score_bounds(
        Z_min, Z_max, rho_lb, rho_ub, L0_reg_ind=L0_reg_ind, max_size=max_size
    )

    assert s_min <= s_max

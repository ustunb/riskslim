"""Test utility functions."""

import os
import tempfile
from itertools import cycle
import logging
import pytest
import numpy as np
from riskslim.utils import (
    print_model, setup_logging, print_log,
    validate_settings, is_integer, cast_to_integer
)


@pytest.mark.parametrize('variable_names',
    [
        ['(Intercept)', *['var_' + str(i) for i in range(9)]],
        ['var_' + str(i) for i in range(10)]
    ]
)
@pytest.mark.parametrize('outcome_name', ['Outcome',None])
def test_print_model(variable_names, outcome_name):

    rho = np.random.rand(10)

    print_model(rho, variable_names, outcome_name)


@pytest.mark.parametrize('log_to_console', [True, False])
@pytest.mark.parametrize('log_file', [True, False])
def test_setup_logging(log_to_console, log_file):

    temp = None
    if log_file is True:
        temp = tempfile.NamedTemporaryFile(suffix='.log', mode='w')
        log_file = temp.name
    else:
        log_file = None

    logger = logging.getLogger()
    logger = setup_logging(logger, log_to_console=log_to_console, log_file=log_file)

    if temp is not None:
        temp.close()

    assert True


def test_print_log():

    print_log('msg', True)
    print_log('msg', False)


@pytest.mark.parametrize('settings', [None, {'b':2}])
def test_validate_settings(settings):

    defaults = {'a': 0, 'b': 1}
    validated_settings = validate_settings(settings, defaults)

    if settings is None:
        assert validated_settings['a'] == defaults['a']
        assert validated_settings['b'] == defaults['b']
    else:
        assert validated_settings['a'] == defaults['a']
        assert validated_settings['b'] == settings['b']


def test_is_integer():
    assert is_integer(np.array([0]))
    assert not is_integer(np.array([0.1]))


def test_cast_to_integer():

    int_arr = cast_to_integer(np.array([0.01, 1.01]))
    assert int_arr.dtype == np.float64
    assert int_arr[0] == 0.
    assert int_arr[1] == 1.

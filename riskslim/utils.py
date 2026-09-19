import logging
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
import time
import numpy as np
import prettytable as pt
from .bounds import Bounds
from .defaults import INTERCEPT_NAME


# MODEL PRINTING
def print_model(rho, variable_names, outcome_name, show_omitted_variables=False, return_only=False):

    rho_values = np.copy(rho)
    rho_names = list(variable_names)

    if INTERCEPT_NAME in rho_names:
        intercept_ind = variable_names.index(INTERCEPT_NAME)
        intercept_val = int(rho[intercept_ind])
        rho_values = np.delete(rho_values, intercept_ind)
        rho_names.remove(INTERCEPT_NAME)
    else:
        intercept_val = 0

    if outcome_name is None:
        predict_string = "Pr(Y = +1) = 1.0/(1.0 + exp(-(%d + score))" % intercept_val
    else:
        predict_string = "Pr(%s = +1) = 1.0/(1.0 + exp(-(%d + score))" % (outcome_name.upper(), intercept_val)

    if not show_omitted_variables:
        selected_ind = np.flatnonzero(rho_values)
        rho_values = rho_values[selected_ind]
        rho_names = [rho_names[i] for i in selected_ind]

        #sort by most positive to most negative
        sort_ind = np.argsort(-np.array(rho_values))
        rho_values = [rho_values[j] for j in sort_ind]
        rho_names = [rho_names[j] for j in sort_ind]
        rho_values = np.array(rho_values)

    rho_values_string = [str(int(i)) + " points" for i in rho_values]
    n_variable_rows = len(rho_values)
    if n_variable_rows > 0:
        total_string = "ADD POINTS FROM ROWS %d to %d" % (1, n_variable_rows)
    else:
        total_string = "NO VARIABLES: SCORE IS 0"

    max_name_col_length = max(map(len, [predict_string, total_string, *rho_names])) + 2
    max_value_col_length = max(7, max(map(len, rho_values_string), default=0) + len("points")) + 2

    m = pt.PrettyTable()
    m.field_names = ["Variable", "Points", "Tally"]

    m.add_row([predict_string, "", ""])
    m.add_row(['=' * max_name_col_length, "=" * max_value_col_length, "========="])

    for name, value_string in zip(rho_names, rho_values_string):
        m.add_row([name, value_string, "+ ....."])

    m.add_row(['=' * max_name_col_length, "=" * max_value_col_length, "========="])
    m.add_row([total_string, "SCORE", "= ....."])
    m.header = False
    m.align["Variable"] = "l"
    m.align["Points"] = "r"
    m.align["Tally"] = "r"

    if not return_only:
        print(m)

    return m


# LOGGING
def setup_logging(logger, log_to_console = True, log_file = None):
    """
    Sets up logging to console and file on disk
    See https://docs.python.org/2/howto/logging-cookbook.html for details on how to customize

    Parameters
    ----------
    log_to_console  set to True to disable logging in console
    log_file        path to file for loggin

    Returns
    -------
    Logger object that prints formatted messages to log_file and console
    """

    # quick return if no logging to console or file
    if log_to_console is False and log_file is None:
        logger.disabled = True
        return logger

    log_format = logging.Formatter(fmt='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%m-%d-%Y %I:%M:%S %p')

    # log to file
    if log_file is not None:
        fh = logging.FileHandler(filename=log_file)
        #fh.setLevel(logging.DEBUG)
        fh.setFormatter(log_format)
        logger.addHandler(fh)

    if log_to_console:
        ch = logging.StreamHandler()
        #ch.setLevel(logging.DEBUG)
        ch.setFormatter(log_format)
        logger.addHandler(ch)

    return logger


def print_log(msg, print_flag = True):
    """

    Parameters
    ----------
    msg
    print_flag

    Returns
    -------

    """
    if print_flag:
        if isinstance(msg, str):
            print('%s | %s' % (time.strftime("%m/%d/%y @ %I:%M %p", time.localtime()), msg))
        else:
            print('%s | %r' % (time.strftime("%m/%d/%y @ %I:%M %p", time.localtime()), msg))
        sys.stdout.flush()

#
def check_cplex():
    try:
        __import__("cplex")
    except:
        raise ImportError("CPLEX must be installed.")
    return True

# File IO
def open_file(file_name):
    """
    open a file using the System viewer
    :param file_name: path of the file
    :return: None
    """
    f = Path(file_name)
    assert f.exists(), 'file not found: %s' % str(f)
    cmd = 'open "%s"' % str(f)
    os.system(cmd)


# Settings
def validate_settings(settings=None, defaults=None, raise_key_error=True):

    if settings is None:
        settings = dict()
    else:
        assert isinstance(settings, dict)
        settings = dict(settings)

    if defaults is not None:
        assert isinstance(defaults, dict)
        _settings = defaults.copy()
        for k, v in settings.items():
            # Raise error if key isn't valid
            if raise_key_error and k not in defaults:
                raise ValueError(f"Invalid setting: {k}")
            else:
                _settings[k] = v

        settings = _settings

    return settings

# Data Types
def is_integer(x):
    """
    checks if numpy array is an integer vector

    Parameters
    ----------
    x

    Returns
    -------

    """
    return np.array_equal(x, np.require(x, dtype=np.int_))


def cast_to_integer(x):
    """
    casts numpy array to integer vector

    Parameters
    ----------
    x

    Returns
    -------

    """
    original_type = x.dtype
    return np.require(np.rint(x), dtype=original_type)


@dataclass
class Stats:
    """Data class for tracking statistics."""
    incumbent: np.ndarray
    upperbound: float = np.inf
    bounds: Bounds = field(default_factory=Bounds)
    lowerbound: float = 0.0
    relative_gap: float = np.inf
    nodes_processed: int = 0
    nodes_remaining: int = 0
    # Time
    start_time: float = np.nan
    total_run_time: float = 0.0
    total_cut_time: float = 0.0
    total_polish_time: float = 0.0
    total_round_time: float = 0.0
    total_round_then_polish_time: float = 0.0
    # Cuts
    cut_callback_times_called: int = 0
    heuristic_callback_times_called: int = 0
    total_cut_callback_time: float = 0.0
    total_heuristic_callback_time: float = 0.0
    # Number of times solutions were updates
    n_incumbent_updates: int = 0
    n_heuristic_updates: int = 0
    n_cuts: int = 0
    n_polished: int = 0
    n_rounded: int = 0
    n_rounded_then_polished: int = 0
    # Total # of bound updates
    n_update_bounds_calls: int = 0
    n_bound_updates: int = 0
    n_bound_updates_loss_min: int = 0
    n_bound_updates_loss_max: int = 0
    n_bound_updates_min_size: int = 0
    n_bound_updates_max_size: int = 0
    n_bound_updates_objval_min: int = 0
    n_bound_updates_objval_max: int = 0

    def asdict(self):
        return self.__dict__

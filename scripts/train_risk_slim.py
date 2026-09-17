#!/usr/bin/python

"""
This file is to train a RiskSLIM model in a batch computing environment
It parses command line arguments, and can be called as:

python train_risk_slim.py --data="${data_file}" --results="${results_file}"

where:

data_file       csv file containing the training data
results_file    file name for the save file; needs to be unique and not already exist on disk

Use "python train_risk_slim.py --help" for a description of additional arguments.

Copyright (C) 2017 Berk Ustun
"""
import os
import sys
import time
import argparse
import logging
import pickle
import json
import numpy as np

# add the source directory to search path to avoid module import errors if riskslim has not been installed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from riskslim.helper_functions import load_data_from_csv, setup_logging
from riskslim.coefficient_set import CoefficientSet
from riskslim.lattice_cpa import run_lattice_cpa, DEFAULT_LCPA_SETTINGS

# uncomment for debugging

# TODO: run the following when building
# with open(settings_json, 'w') as outfile:
#     json.dump(DEFAULT_LCPA_SETTINGS, outfile, sort_keys = False, indent=4)

def setup_parser():
    """
    Create an argparse Parser object for RiskSLIM command line arguments.
    This object determines all command line arguments, handles input
    validation and default values.

    See https://docs.python.org/3/library/argparse.html for configuration
    """

    #parser helper functions
    def is_positive_integer(value):
        parsed_value = int(value)
        if parsed_value <= 0:
            raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
        return parsed_value

    def is_positive_float(value):
        parsed_value = float(value)
        if parsed_value <= 0.0:
            raise argparse.ArgumentTypeError("%s must be a positive value" % value)
        return parsed_value

    def is_negative_one_or_positive_integer(value):
        parsed_value = int(value)
        if not (parsed_value == -1 or parsed_value >= 1):
            raise argparse.ArgumentTypeError("%s is an invalid value (must be -1 or >=1)" % value)
        else:
            return parsed_value

    def is_file_on_disk(file_name):
        if not os.path.isfile(file_name):
            raise argparse.ArgumentTypeError("the file %s does not exist!" % file_name)
        else:
            return file_name

    def is_file_not_on_disk(file_name):
        if os.path.isfile(file_name):
            raise argparse.ArgumentTypeError("the file %s already exists on disk" % file_name)
        else:
            return file_name

    def is_valid_fold(value):
        parsed_value = int(value)
        if parsed_value < 0:
            raise argparse.ArgumentTypeError("%s must be a positive integer" % value)
        return parsed_value

    parser = argparse.ArgumentParser(
        prog='train_risk_slim',
        description='Train a RiskSLIM classifier from the command shell',
        epilog='Copyright (C) 2017 Berk Ustun',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data',
                        type=str,
                        required=True,
                        help='csv file with training data')

    parser.add_argument('--results',
                        type=str,
                        required=True,
                        help='name of results file (must not already exist)')

    parser.add_argument('--cvindices',
                        type=is_file_on_disk,
                        help='csv file with indices for K-fold CV')

    parser.add_argument('--fold',
                        type=is_valid_fold,
                        default=0,
                        help='index of test fold; set as 0 to use all data for training')

    parser.add_argument('--weights',
                        type=is_file_on_disk,
                        help='csv file with non-negative weights for each point')

    parser.add_argument('--settings',
                        type=is_file_on_disk,
                        help='JSON file with additional settings for LCPA')

    parser.add_argument('--timelimit',
                        type=is_negative_one_or_positive_integer,
                        default=300,
                        help='time limit on training (in seconds); set as -1 for no time limit')

    parser.add_argument('--max_size',
                        type = is_negative_one_or_positive_integer,
                        default=-1,
                        help='maximum number of non-zero coefficients; set as -1 for no limit')

    parser.add_argument('--max_coef',
                        type=is_positive_integer,
                        default=5,
                        help='value of upper and lower bounds for any coefficient')

    parser.add_argument('--max_offset',
                        type=is_negative_one_or_positive_integer,
                        default=-1,
                        help='value of upper and lower bound on offset parameter; set as -1 to use a conservative value')

    parser.add_argument('--c0_value',
                        type=is_positive_float,
                        default=1e-6,
                        help='l0 regularization parameter; set as a positive number between 0.00 and log(2)')

    parser.add_argument('--w_pos',
                        type=is_positive_float,
                        default=1.00,
                        help='w_pos')

    parser.add_argument('--log',
                        type=str,
                        help='name of the log file')

    parser.add_argument('--silent',
                        action='store_true',
                        help='flag to suppress logging to stderr')

    return parser

if __name__ == '__main__':

    parser = setup_parser()
    parsed = parser.parse_args()
    parsed_dict = vars(parsed)
    parsed_string = [key + ' : ' + str(parsed_dict[key]) + '\n' for key in parsed_dict]
    parsed_string.sort()

    # setup logging
    logger = logging.getLogger()
    logger = setup_logging(logger, log_to_console =(not parsed.silent), log_file = parsed.log)
    logger.setLevel(logging.INFO)
    logger.info("running 'train_risk_slim.py'")
    logger.info("working directory: %r" % os.getcwd())
    logger.info("parsed the following variables:\n-%s" % '-'.join(parsed_string))

    # check results_file does not exist
    if os.path.isfile(parsed.results):
        logger.error("results file %s already exists)" % parsed.results)
        logger.error("either delete %s or choose a different name" % parsed.results)
        sys.exit(1)

    # check settings_json exists / or use default settings
    settings = dict(DEFAULT_LCPA_SETTINGS)
    if parsed.settings is not None:
        with open(parsed.settings) as json_file:
            loaded_settings = json.load(json_file)
            loaded_settings = {str(key): loaded_settings[key] for key in loaded_settings if key in settings}
            settings.update(loaded_settings)

    #overwrite parameters specified by the user
    settings['max_runtime'] = float('inf') if parsed.timelimit == -1 else parsed.timelimit
    settings['c0_value'] = parsed.c0_value
    settings['w_pos'] = parsed.w_pos

    # check if sample weights file was specified, if not set as None
    logger.info("loading data and sample weights")

    data = load_data_from_csv(dataset_csv_file = parsed.data,
                              sample_weights_csv_file = parsed.weights,
                              fold_csv_file = parsed.cvindices,
                              fold_num = parsed.fold)
    N, P = data['X'].shape

    # initialize coefficient set and offset parameter
    logger.info("creating coefficient set and constraints")
    max_coefficient = parsed.max_coef
    max_model_size = parsed.max_size if parsed.max_size >= 0 else float('inf')
    max_offset = parsed.max_offset if parsed.max_offset >= 0 else float('inf')

    coef_set = CoefficientSet(variable_names = data['variable_names'],
                              lb = -max_coefficient,
                              ub = max_coefficient,
                              sign = 0)
    coef_set.update_intercept_bounds(X = data['X'], y = data['y'], max_offset = max_offset, max_L0_value = max_model_size)

    #print coefficient set
    if not parsed.silent:
        print(coef_set)

    constraints = {
        'min_size': 0,
        'max_size': max_model_size,
        'coef_set': coef_set,
    }

    # fit RiskSLIM model using Lattice Cutting Plane Algorithm
    model_info, mip_info, lcpa_info = run_lattice_cpa(data, constraints, settings)

    # save output to disk
    results = {
        "date": time.strftime("%d/%m/%y", time.localtime()),
        "data_file": parsed.data,
        "fold_file": parsed.cvindices,
        "fold_num": parsed.settings,
        "results_file": parsed.results,
    }
    results.update(model_info)

    coef_set = results.pop('coef_set')
    results['coef_set_ub'] = coef_set.ub
    results['coef_set_lb'] = coef_set.lb
    results['coef_set_signs'] = coef_set.sign
    results['coef_set_c0'] = coef_set.c0

    logger.info("saving results...")
    with open(parsed.results, 'wb') as outfile:
        pickle.dump(results, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("saved results as pickle file: %r" % parsed.results)
    logger.info('''to access results, use this snippet:

                \t\t\t    import pickle
                \t\t\t    f = open(results_file, 'rb')
                \t\t\t    results = pickle.load(f)
                '''
                )
    logger.info("finished training")
    logger.info("quitting\n\n")
    sys.exit(0)
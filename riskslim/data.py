"""Data structures."""
import warnings

import numpy as np
import pandas as pd
from riskslim.defaults import INTERCEPT_NAME, OUTCOME_NAME
from riskslim.utils import is_integer


class ClassificationDataset:
    """Binary classification data.

    Attributes
    ----------
    X : 2d-array
        Observations (rows) and features (columns).
        With an addtional column of 1s for the INTERCEPT_NAME.
    y : 2d-array
        Class labels (+1, -1) with shape (n_rows, 1).
    variable_names : list of str, optional, default: None
        Names of each features. Only needed if coefficients is not passed on
        initalization. None defaults to generic variable names.
    outcome_name : str, optional, default: None
        Name of the output class.
    sample_weights : 2d array, optional, default: None
        Sample weights with shape (n_features, 1). Must all be positive.
    """
    def __init__(self, X, y, variable_names = None, outcome_name=None, sample_weights=None):

        # todo copy
        # convert y \in 0,1 to y \in -1,1 in this function?
        # output warnings if X is not binary
        # **issue warning if any column of X are constant**

        if variable_names is None:
            variable_names = default_variable_names(n_variables = X.shape[1])

        # store entries
        self._X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        self._variable_names = [INTERCEPT_NAME] + list(variable_names)

        #todo: convert to binary
        self._y = np.array(y)
        self._classes, _ = np.unique(y, return_inverse=True)
        self._outcome_name = outcome_name

        self._sample_weights = sample_weights
        assert self.__check_rep__()

        self._Z = (self._X * self._y).astype(np.float64)

        # Infer variable types
        self._variable_types = np.zeros(self.X.shape[1], dtype="str")
        self._variable_types[:] = "C"
        self._variable_types[np.all(self.X == np.require(self.X, dtype=np.int_), axis=0)] = "I"
        self._variable_types[np.all(self.X == np.require(self.X, dtype=np.bool_), axis=0)] = "B"
        self._integer_data = not np.any(self._variable_types == "C")


    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def Z(self):
        return self._Z

    @property
    def classes(self):
        return self._classes

    @property
    def sample_weights(self):
        return self._sample_weights

    @property
    def variable_names(self):
        return self._variable_names

    @property
    def outcome_name(self):
        return self._outcome_name

    @property
    def n(self):
        return self._X.shape[0]

    @property
    def d(self):
        return self._X.shape[1]

    @property
    def variable_types(self):
        return self._variable_types

    @property
    def df(self):
        """Pandas dataframe."""
        df = pd.DataFrame(self._X[:, 1:], columns=self._variable_names[1:])
        df.insert(loc = 0, column = self._outcome_name, value = self._y)
        return df

    def __check_rep__(self):
        """Ensures that data contains training data that is suitable for binary
            classification problems throws AssertionError if not.

            Parameters
            ----------
            X : 2d-array
                Observations (rows) and features (columns).
                With an addtional column of 1s for the INTERCEPT_NAME.
            y : 2d-array
                Class labels (+1, -1) with shape (n_rows, 1).
            variable_names : list of str
                Names of each features.
            outcome_name : str, optional, default: None
                Name of the output class.
            sample_weights : 2d array.
                Sample weights with shape (n_features, 1). Must all be positive.

            Returns
            -------
            True if data passes checks
            """
        X = self._X
        y = self._y
        variable_names = self._variable_names
        outcome_name = self._outcome_name
        sample_weights =  self._sample_weights

        assert isinstance(X, np.ndarray), "X should be numpy array"
        assert isinstance(y, np.ndarray), "y should be numpy.ndarray"
        assert isinstance(variable_names, list), "variable_names should be a list"
        assert isinstance(outcome_name, str), "outcome_name should be a str"

        # sizes and uniqueness
        n, d = X.shape
        assert n > 0, 'X matrix must have at least 1 row'
        assert d > 0, 'X matrix must have at least 1 column'
        assert len(y) == n, 'dimension mismatch. y must contain as many entries as X.'
        assert len(variable_names) == d, 'len(variable_names) should be same as # of cols in X'
        assert len(list(set(variable_names))) == len(variable_names), 'variable_names is not unique'

        # feature matrix
        assert np.isfinite(X).all(), 'X should consist of finite values'

        classes = np.unique(y)
        assert len(classes) == 2, 'y should contain two classes'
        assert np.isin(classes,(0,1)).all() or np.isin(classes, (-1,1)).all(), 'y should consist of 0,1 or +1,-1 values'

        if sample_weights is not None:
            assert isinstance(sample_weights, np.ndarray), 'sample_weights should be an array'
            assert len(sample_weights) == n, 'sample_weights should contain N elements'
            assert np.greater(sample_weights, 0.0).all(), 'sample_weights[i] > 0 for all i '

        return True

    def __str__(self):
        return str(self.df)

    def __repr__(self):
        """Preview of dataframe."""
        with pd.option_context('display.max_rows', 6):
            with pd.option_context('display.max_columns', 6):
                return str(self.df)

    def check_data(self):
        if np.all(self._variable_types[1:] == "B"):
            warnings.warn("X is recommended to be all binary.")

        # Constant warning
        idx = np.flatnonzero(self.X == self.X[0], axis=0)
        constant_variables = [self.variable_names[j] for j in idx if j > 0]
        if len(constant_variables):
            warnings.warn("Constant variable other than intercept found in X.")



def default_variable_names(n_variables, prefix = 'x'):
    """
    Parameters
    ----------
    n_features: # of columns in X matrix
    include_intercept = set to True to set the first variable name to INTERCEPT_NAME
    Returns
    list of unique variable names
    -------
    """
    assert is_integer(n_variables) and n_variables > 0
    assert isinstance(prefix, str) and len(prefix) > 0
    n_padding = np.floor(np.log10(n_variables + 1)).astype(int) + 1
    fmt = ':0{}d'.format(n_padding)
    namer = prefix + '{' + fmt + '}'
    names = [namer.format(j) for j in range(n_variables)]
    return names


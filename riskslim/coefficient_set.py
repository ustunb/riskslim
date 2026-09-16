import numpy as np
import warnings
from prettytable import PrettyTable
from .bounds import get_score_bounds
from .defaults import INTERCEPT_NAME

class CoefficientSet:
    """
    Class to represent and specify constraints on coefficients of input variables
    Constraints include upper and lower bounds, variable type, and regularization.
    CoefficientSet is a set of CoefficientElement
    """
    _default_print_flag = True
    _initialized = False
    _variable_names = None

    def __init__(self, variable_names, lb=-5., ub=5., c0=float('nan'), vtype='I', print_flag=True):

        # Set variables using setter methods
        self.variable_names = list(variable_names)
        self.print_flag = print_flag

        # Ensure bounds are floats
        if isinstance(ub, int):
            ub = float(ub)
        elif isinstance(ub, list):
            ub = np.array(ub, dtype=np.float64)

        if isinstance(lb, int):
            lb = float(lb)
        elif isinstance(lb, list):
            lb = np.array(lb, dtype=np.float64)

        ub = self._expand_values(value = ub)
        lb = self._expand_values(value = lb)
        c0 = self._expand_values(value = c0)
        vtype = self._expand_values(value = vtype)

        self._coef_elements = dict()
        for name in variable_names:
            idx = variable_names.index(name)
            self._coef_elements[name] = _CoefficientElement(name = name, ub = ub[idx], lb = lb[idx], c0 = c0[idx], vtype = vtype[idx])

        self._check_rep()
        self._initialized = True

    ### methods ###
    def update_intercept_bounds(self, X, y, max_offset, max_size = None):
        """
        uses data to set the lower and upper bound on the offset to a conservative value
        the value is guaranteed to avoid a loss in performance

        optimal_offset = max_abs_score + 1
        where max_abs_score is the largest absolute score that can be achieved using the coefficients in coef_set
        with the training data. note:
        when offset >= optimal_offset, then we predict y = +1 for every example
        when offset <= optimal_offset, then we predict y = -1 for every example
        thus, any feasible model should do better.


        Parameters
        ----------
        X
        y
        max_offset
        max_size

        Returns
        -------
        None

        """
        if INTERCEPT_NAME not in self._coef_elements:
            raise ValueError(f"coef_set must contain a variable for the offset called {INTERCEPT_NAME}")

        e = self._coef_elements[INTERCEPT_NAME]

        # get idx of intercept/variables
        names = self.variable_names
        variable_names = list(names)
        variable_names.remove(INTERCEPT_NAME)
        variable_idx = np.array([names.index(n) for n in variable_names])

        # get max # of non-zero coefficients given model size limit
        penalized_idx = [self._coef_elements[n].penalized for n in variable_names]
        trivial_max_size = len(penalized_idx)

        if max_size is None:
            max_size = trivial_max_size

        if max_size > 0:
            max_size = min(trivial_max_size, max_size)

        # update intercept bounds
        Z = X * y
        Z_min = np.min(Z, axis = 0)
        Z_max = np.max(Z, axis = 0)

        # get regularized indices
        L0_reg_ind = self.penalized_indices()

        # get smallest / largest score
        s_min, s_max = get_score_bounds(Z_min = Z_min[variable_idx],
                                        Z_max = Z_max[variable_idx],
                                        rho_lb = self.lb[variable_idx],
                                        rho_ub = self.ub[variable_idx],
                                        L0_reg_ind = L0_reg_ind[variable_idx],
                                        max_size = max_size)

        # set intercept
        conservative_offset = max(abs(s_min), abs(s_max)) + 1
        if max_offset is None:
            max_offset = conservative_offset
        else:
            max_offset = min(max_offset, conservative_offset)
        e.ub = max_offset
        e.lb = -max_offset

    def tabulate(self):
        t = PrettyTable()
        t.align = "r"
        t.add_column("variable_name", self._variable_names)
        t.add_column("vtype", self.vtype)
        t.add_column("sign", self.sign)
        t.add_column("lb", self.lb)
        t.add_column("ub", self.ub)
        t.add_column("c0", self.c0)
        return str(t)

    ### properties ####
    @property
    def variable_names(self):
        return self._variable_names

    @variable_names.setter
    def variable_names(self, names):
        assert isinstance(names, list), 'variable_names must be a list'
        for name in names:
            assert isinstance(name, str), 'variable_names must be a list of strings'
        assert len(names) == len(set(names)), 'variable_names contain elements with unique names'
        if self._variable_names is not None:
            assert len(names) == len(self), 'variable_names must contain at least %d elements' % len(self)
        self._variable_names = list(names)

    def index(self, name):
        assert isinstance(name, str)
        if name in self._variable_names:
            return self._variable_names.index(name)
        else:
            raise ValueError('no variable named %s in coefficient set' % name)

    def penalized_indices(self):
        return np.array(list(map(lambda v: self._coef_elements[v].penalized, self._variable_names)))

    #### flags ####
    @property
    def print_flag(self):
        return self._default_print_flag

    @print_flag.setter
    def print_flag(self, flag):
        self._default_print_flag = bool(flag)

    @property
    def max_coef(self):
        idx = [j for j, n in enumerate(self.variable_names) if n != INTERCEPT_NAME]
        value = np.maximum(np.abs(self.ub[idx]), np.abs(self.lb[idx])).max()
        return value

    #### coefficient element access ####
    def __getattr__(self, name):

        # todo: remove
        if name == 'C_0j':
            name = 'c0'

        vals = [getattr(self._coef_elements[v], name) for v in self._variable_names]
        if name in ['ub', 'lb', 'c0', 'sign', 'vtype']:
            return np.array(vals)
        else:
            return list(vals)

    def __setattr__(self, name, value):
        if self._initialized:
            assert all(map(lambda e: hasattr(e, name), self._coef_elements.values()))
            attr_values = self._expand_values(value)
            for e, v in zip(self._coef_elements.values(), attr_values):
                setattr(e, name, v)
            self._check_rep()
        else:
            object.__setattr__(self, name, value)

    def __getitem__(self, key):

        if isinstance(key, int):
            assert 0 <= int(key) <= len(self._variable_names)
            return self._coef_elements[self._variable_names[key]]
        elif isinstance(key, str):
            return self._coef_elements[key]
        else:
            raise KeyError('invalid key')

    def __setitem__(self, key, value):

        if isinstance(key, int):
            assert 0 <= int(key) <= len(self._variable_names)
            key = self._variable_names[key]
        elif isinstance(key, str):
            assert isinstance(key, str)
            assert key in self._variable_names
            assert value.name == key
        else:
            raise KeyError('invalid key')

        assert isinstance(value, _CoefficientElement)
        self._coef_elements[key] = value

    #### built-ins ####
    def __len__(self):
        return len(self._variable_names)

    def __str__(self):
        return self.tabulate()

    def __repr__(self):
        return self.tabulate()

    def _check_rep(self):

        #if self._check_flag:

        assert len(self._variable_names) == len(set(self._variable_names))

        for name in self._variable_names:
            assert isinstance(name, str)
            assert len(name) >= 1
            assert self._coef_elements[name]._check_rep()

        #if self._correct_flag:

        for name in self._variable_names:
            e = self._coef_elements[name]
            if name in {'Intercept', '(Intercept)', 'intercept', '(intercept)'}:
                if e.c0 > 0 or np.isnan(e.c0):
                    if self._default_print_flag:
                        warnings.warn("setting c0_value = 0.0 for %s to ensure that intercept is not penalized" % name)
                    e._c0 = 0.0

        return True

    def _expand_values(self, value):

        if isinstance(value, np.ndarray):
            if value.size == len(self._variable_names):
                value_array = value
            elif value.size == 1:
                value_array = np.repeat(value, len(self._variable_names))
            else:
                raise ValueError("length mismatch; need either 1 or %d values" % len(self._variable_names))

        elif isinstance(value, list):
            if len(value) == len(self._variable_names):
                value_array = value
            elif len(value) == 1:
                value_array = [value] * len(self._variable_names)
            else:
                raise ValueError("length mismatch; need either 1 or %d values" % len(self._variable_names))

        elif isinstance(value, str):
            value_array = [str(value)] * len(self._variable_names)

        elif isinstance(value, int):
            value_array = [int(value)] * len(self._variable_names)

        elif isinstance(value, float):
            value_array = [float(value)] * len(self._variable_names)

        else:
            raise ValueError("unknown variable type %s")

        return(value_array)


class _CoefficientElement(object):
    """
    Class to represent and specify constraints on a single input variable
    Constraints include upper and lower bounds, variable type, and regularization.
    """

    _default_ub = 5.
    _default_lb = -5.
    _default_c0 = float('nan')
    _default_vtype = 'I'
    _valid_vtypes = ['I', 'C']

    def _is_integer(self, x):
        return np.array_equal(x, np.require(x, dtype = np.int_))

    def __init__(self, name, **kwargs):
        self._name = str(name)
        self._ub = kwargs.get('ub', _CoefficientElement._default_ub)
        self._lb = kwargs.get('lb', _CoefficientElement._default_lb)
        self._c0 = kwargs.get('c0', _CoefficientElement._default_c0)
        self._vtype = kwargs.get('vtype', _CoefficientElement._default_vtype)
        assert self._check_rep()

    @property
    def name(self):
        return self._name

    @property
    def vtype(self):
        return self._vtype

    @vtype.setter
    def vtype(self, value):
        assert isinstance(value, str)
        assert value in self._valid_vtypes
        self._vtype = str(value)

    @property
    def ub(self):
        return self._ub

    @ub.setter
    def ub(self, value):
        if hasattr(value, '__len__'):
            assert len(value) == 1
            value = value[0]
        assert value >= self._lb
        self._ub = float(value)


    @property
    def lb(self):
        return self._lb


    @lb.setter
    def lb(self, value):
        if hasattr(value, '__len__'):
            assert len(value) == 1
            value = value[0]
        assert value <= self._ub
        self._lb = float(value)


    @property
    def c0(self):
        return self._c0

    @c0.setter
    def c0(self, value):
        if np.isnan(value):
            self._c0 = float('nan')
        else:
            assert np.isfinite(value), 'L0 penalty for %s must either be NaN or a finite positive number' % self._name
            assert value >= 0.0, 'L0 penalty for %s must either be NaN or a finite positive number' % self._name
            self._c0 = float(value)

    @property
    def penalized(self):
        return np.greater(self._c0, 0.0) or np.isnan(self._c0)


    @property
    def sign(self):
        if np.greater(self._ub, 0.0) and np.greater_equal(self._lb, 0.0):
            return 1
        elif np.less_equal(self._ub, 0.0) and np.less(self._lb, 0.0):
            return -1
        else:
            return 0

    @sign.setter
    def sign(self, value):
        if np.greater(value, 0.0):
            self._lb = 0.0
        elif np.less(value, 0.0):
            self._ub = 0.0

    def _check_rep(self):

        #name
        assert isinstance(self._name, str)
        assert len(self._name) >= 1

        #bounds
        assert np.isfinite(self.ub)
        assert np.isfinite(self.lb)
        assert self.ub >= self.lb

        # value
        assert self._vtype in self._valid_vtypes
        assert np.isnan(self.c0) or (np.isfinite(self.c0) and np.greater_equal(self.c0, 0.0))

        return True

    def __repr__(self):
        return self.tabulate()

    def __str__(self):
        return self.tabulate()

    def tabulate(self):
        s = ['-' * 60,
             'variable: %s' % self._name,
             '-' * 60,
             '%s: %1.1f' % ('ub', self._ub),
             '%s: %1.1f' % ('lb', self._lb),
             '%s: %1.2g' % ('c0', self._c0),
             '%s: %1.0f' % ('sign', self.sign),
             '%s: %s' % ('vtype', self._vtype)]
        t = '\n' + '\n'.join(s) + '\n'
        return t



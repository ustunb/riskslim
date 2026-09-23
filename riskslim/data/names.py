"""
Names and Rule Parsing Module
=============================

Provides the Names dataclass for column names and rule parsing utilities.
"""

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# =============================================================================
# Names Dataclass
# =============================================================================


@dataclass
class Names:
    """Column names for features (X), label (y), and groups (G).

    :param X: Feature names.
    :param y: Label name (default 'y', always required).
    :param G: Group attribute names.
    """

    DEFAULT_Y = 'y'

    X: list[str] = field(default_factory=list)
    y: str = DEFAULT_Y
    G: list[str] = field(default_factory=list)

    @staticmethod
    def is_valid_name(s) -> bool:
        """Check if s is a valid variable name (non-empty string).

        :param s: Value to check.
        :returns: True if s is a non-empty string.
        """
        return isinstance(s, str) and len(s.strip()) > 0

    @staticmethod
    def generate_feature_names(d: int) -> list[str]:
        """Generate default feature names x01, x02, ... for d features."""
        return [f"x{j:02d}" for j in range(1, d + 1)]

    def __post_init__(self):
        valid_y = Names.is_valid_name(self.y)
        assert valid_y, f"Invalid y name: {self.y!r}"
        assert self._validate_names(self.X, "X")
        assert self._validate_names(self.G, "G")
        assert self.check_disjoint()

    def _validate_names(self, names: list[str], label: str) -> bool:
        """Validate a list of names.

        :param names: List of names to validate.
        :param label: Label for error messages (e.g., "X" or "G").
        :returns: True if valid.
        """
        all_valid = all(Names.is_valid_name(n) for n in names)
        all_unique = len(names) == len(set(names))
        assert all_valid, f"Invalid {label} names: {[n for n in names if not Names.is_valid_name(n)]}"
        assert all_unique, f"Duplicate {label} names: {list({n for n in names if names.count(n) > 1})}"
        return True

    @property
    def all(self) -> list[str]:
        """All names in order: [y] + G + X."""
        return [self.y] + self.G + self.X

    def check_disjoint(self) -> bool:
        """Assert X, G, y are disjoint. Returns True."""
        overlap = set(self.X) & set(self.G)
        assert not overlap, f"X and G overlap: {overlap}"
        assert self.y not in self.X, f"y '{self.y}' in X"
        assert self.y not in self.G, f"y '{self.y}' in G"
        return True

    def validate_columns(self, df: pd.DataFrame, check_extra: bool = True) -> bool:
        """Assert names match DataFrame columns. Returns True.

        :param df: DataFrame to validate against.
        :param check_extra: If True, asserts no extra columns in df.
        :returns: True if valid.
        """
        expected = set(self.all)
        actual = set(df.columns)
        all_present = expected <= actual
        no_extra = actual <= expected
        assert all_present, f"Missing columns: {expected - actual}"
        assert not check_extra or no_extra, f"Extra columns: {actual - expected}"
        return True


# =============================================================================
# Constants (for rule parsing)
# =============================================================================

SEPARATOR = '_'

OPERATORS = ('is', 'isnot', 'geq', 'lt', 'leq', 'gt', 'eq', 'neq', 'in', 'notin')

COMPLEMENTS = {
    'is': 'isnot', 'isnot': 'is',
    'geq': 'lt', 'lt': 'geq',
    'leq': 'gt', 'gt': 'leq',
    'eq': 'neq', 'neq': 'eq',
    'in': 'notin', 'notin': 'in',
}

# Regex for parsing rule names like 'Age_geq_30' or 'work_hours_lt_40'
# Uses alternation for operators to handle feature names with underscores
_RULE_PATTERN = re.compile(
    r'^(.+)_(' + '|'.join(OPERATORS) + r')_(.+)$'
)


# Delimiters for bin/interval notation in rule names
VALID_DELIMITERS_START = ('[', '(')
VALID_DELIMITERS_END = (')', ']')


# =============================================================================
# Rule Name Parsing
# =============================================================================


class RuleName:
    """Parser for rule names like 'Age_geq_30'.

    Provides class methods for parsing rule names into components.

    Example
    -------
    >>> RuleName.parse('Age_geq_30')
    ('Age', 'geq', '30')
    >>> RuleName.parse_feature('Age_geq_30')
    'Age'
    >>> RuleName.parse_complement('Age_geq_30')
    'Age_lt_30'
    """

    @staticmethod
    def parse(rule_name: str) -> tuple[str, str, str]:
        """Parse a rule name into (feature, operator, value) components.

        :param rule_name: Rule name like 'Age_geq_30' or 'Job_is_Manager'.
        :returns: Tuple of (feature_name, operator, value_string).
        :raises ValueError: If rule_name doesn't match expected pattern.
        """
        match = _RULE_PATTERN.match(rule_name)
        if not match:
            raise ValueError(f"Cannot parse rule name: {rule_name!r}")
        return match.groups()

    @staticmethod
    def parse_feature(rule_name: str) -> str:
        """Extract the feature name from a rule name.

        :param rule_name: Rule name like 'Age_geq_30'.
        :returns: The feature name (e.g., 'Age').
        """
        return RuleName.parse(rule_name)[0]

    @staticmethod
    def parse_operator(rule_name: str) -> str:
        """Extract the operator from a rule name.

        :param rule_name: Rule name like 'Age_geq_30'.
        :returns: The operator (e.g., 'geq').
        """
        return RuleName.parse(rule_name)[1]

    @staticmethod
    def parse_value(rule_name: str) -> int | float | str | set | list:
        """Extract and parse the value from a rule name with correct type.

        :param rule_name: Rule name like 'Age_geq_30' or 'Job_is_Manager'.
        :returns: Parsed value with appropriate type:
            - int: for integer thresholds ('Age_geq_30' -> 30)
            - float: for float thresholds ('Age_geq_30.5' -> 30.5)
            - str: for single categorical values ('Job_is_Manager' -> 'Manager')
            - set: for multiple categorical values ('Job_in_{A,B}' -> {'A', 'B'})
            - list: for bin intervals ('Age_in_[20,30]' -> [20, 30])
        """
        _, _, value_str = RuleName.parse(rule_name)

        is_set = value_str.startswith('{') and value_str.endswith('}')
        is_bin = value_str[0] in VALID_DELIMITERS_START and value_str[-1] in VALID_DELIMITERS_END

        if is_set:
            items = value_str[1:-1].split(',')
            out = items[0] if len(items) == 1 else set(items)
        elif is_bin:
            b = Bin.from_string(value_str)
            out = [b.start, b.end]
        else:
            try:
                out = float(value_str) if '.' in value_str else int(value_str)
            except ValueError:
                out = value_str

        return out

    @staticmethod
    def parse_complement(rule_name: str) -> str:
        """Get the complement rule name.

        :param rule_name: Rule name like 'Age_geq_30' or 'Male'.
        :returns: Complement rule name with operator swapped.

        Example
        -------
        >>> RuleName.parse_complement('Age_geq_30')
        'Age_lt_30'
        >>> RuleName.parse_complement('Male')
        'NOT_Male'
        """
        match = _RULE_PATTERN.match(rule_name)
        if match:
            feature, op, value = match.groups()
            comp_op = COMPLEMENTS[op]
            out = f'{feature}{SEPARATOR}{comp_op}{SEPARATOR}{value}'
        elif rule_name.startswith('NOT_'):
            out = rule_name[4:]
        else:
            out = f'NOT_{rule_name}'
        return out

    @staticmethod
    def parse_categories(category_string: str) -> str | set:
        """Parse a category string into a single value or set of values.

        :param category_string: Either a single category name or set notation like '{A,B,C}'.
        :returns: Single string if one category, set if multiple.

        Example
        -------
        >>> RuleName.parse_categories('Manager')
        'Manager'
        >>> RuleName.parse_categories('{A,B,C}')
        {'A', 'B', 'C'}
        """
        category_string = category_string.strip()
        is_set = category_string.startswith('{') and category_string.endswith('}')

        if is_set:
            items = [item.strip() for item in category_string[1:-1].split(',')]
            out = items[0] if len(items) == 1 else set(items)
        else:
            out = category_string
        return out


# =============================================================================
# Bin Class
# =============================================================================


@dataclass
class Bin:
    """Interval over integers or floats with automatic operator detection.

    :param start: Lower bound of the interval.
    :param end: Upper bound of the interval.
    :param closure: Bracket notation for interval closure: '[]', '[)', '(]', or '()'.
    :param integer: If True, normalize to closed form.
    """

    start: float
    end: float
    closure: str = '[)'
    integer: bool = False

    # Computed fields
    symbol: str = field(init=False)
    is_left_unbounded: bool = field(init=False)
    is_right_unbounded: bool = field(init=False)
    one_sided: bool = field(init=False)
    is_point: bool = field(init=False)

    def __check_rep__(self):
        assert self.start <= self.end
        assert self.closure in ('[]', '[)', '(]', '()')
        assert not (self.start == float('-inf') and self.end == float('inf'))
        assert self.integer or self.start != self.end or self.closure == '[]'
        return True

    @staticmethod
    def _normalize(start, end, closure, integer):
        assert start <= end and closure in ('[]', '[)', '(]', '()')
        if integer:
            if not np.isinf(start) and not float(start).is_integer():
                start, closure = int(np.ceil(start)), '[' + closure[1]
            if not np.isinf(end) and not float(end).is_integer():
                end, closure = int(np.floor(end)), closure[0] + ']'
            if closure[0] == '(' and start != float('-inf'):
                start, closure = start + 1, '[' + closure[1]
            if closure[1] == ')' and end != float('inf'):
                end, closure = end - 1, closure[0] + ']'
            start = start if np.isinf(start) else int(start)
            end = end if np.isinf(end) else int(end)
        else:
            start, end = float(start), float(end)
        return start, end, closure

    def __post_init__(self):
        self.start, self.end, self.closure = self._normalize(self.start, self.end, self.closure, self.integer)
        assert self.__check_rep__()

        self.is_left_unbounded = self.start == float('-inf')
        self.is_right_unbounded = self.end == float('inf')
        self.one_sided = self.is_left_unbounded or self.is_right_unbounded
        self.is_point = self.integer and self.start == self.end

        if self.is_left_unbounded:
            closed = self.integer or self.closure[1] == ']'
            self.symbol = 'leq' if closed else 'lt'
        elif self.is_right_unbounded:
            closed = self.integer or self.closure[0] == '['
            self.symbol = 'geq' if closed else 'gt'
        else:
            self.symbol = 'eq' if self.is_point else 'in'

    def __repr__(self):
        if self.is_point:
            out = repr(self.start)
        else:
            d = ('[', ']') if self.integer else (self.closure[0], self.closure[1])
            out = f'{d[0]}{self.start!r},{self.end!r}{d[1]}'
        return out

    def __len__(self):
        if self.one_sided:
            raise OverflowError("Unbounded interval")
        out = int(self.end - self.start + 1) if self.integer else int(self.end - self.start)
        return out

    def length(self):
        if self.one_sided:
            out = float('inf')
        elif self.integer:
            out = int(self.end - self.start + 1)
        else:
            out = self.end - self.start
        return out

    def __eq__(self, other):
        return (isinstance(other, Bin) and self.start == other.start and
                self.end == other.end and (self.integer or self.closure == other.closure))

    def __contains__(self, value):
        value = float(value)
        if self.integer:
            out = self.start <= value <= self.end
        elif self.start < value < self.end:
            out = True
        else:
            at_start = value == self.start and self.closure[0] == '['
            at_end = value == self.end and self.closure[1] == ']'
            out = at_start or at_end
        return out

    def rule_string(self, for_complement=False):
        op = COMPLEMENTS[self.symbol] if for_complement else self.symbol
        if self.one_sided:
            threshold = self.end if self.is_left_unbounded else self.start
            thresh = f'{threshold:d}' if self.integer else f'{threshold:g}'
            out = f'{SEPARATOR}{op}{SEPARATOR}{thresh}'
        else:
            out = f'{SEPARATOR}{op}{SEPARATOR}{repr(self)}'
        return out

    _PARSE_PATTERN = re.compile(r'([\[\(])\s*([-+]?(?:\d+\.?\d*|\d*\.?\d+)|[-+]?[iI][nN][fF])\s*,\s*([-+]?(?:\d+\.?\d*|\d*\.?\d+)|[-+]?[iI][nN][fF])\s*([\]\)])')

    @staticmethod
    def from_string(s, integer=None):
        match = Bin._PARSE_PATTERN.match(s.replace(' ', ''))
        if not match:
            raise ValueError(f'invalid bin string: {s}')
        start_delim, start_str, end_str, end_delim = match.groups()
        start, end = float(start_str), float(end_str)
        if integer is None:
            looks_like_float = '.' in start_str or '.' in end_str
            both_inf = np.isinf(start) and np.isinf(end)
            if looks_like_float or both_inf:
                integer = False
            else:
                start_is_int = np.isinf(start) or float(start).is_integer()
                end_is_int = np.isinf(end) or float(end).is_integer()
                integer = start_is_int and end_is_int
        return Bin(start, end, start_delim + end_delim, integer)


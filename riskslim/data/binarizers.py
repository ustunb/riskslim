"""
Binarizers Module
=================

Binarizer classes for transforming raw features into binary variables.

Terminology
-----------
features
    Raw input columns (Age, Sex, Income).
variables
    Binary rules after binarization (Age_geq_30, Sex_is_Male).
"""

from abc import ABC, abstractmethod
from functools import partial
from itertools import combinations
from typing import cast
import warnings
import numpy as np

from .names import SEPARATOR, COMPLEMENTS, Bin


# =============================================================================
# Helpers
# =============================================================================


def _compare(values: np.ndarray, operator: str, threshold) -> np.ndarray:
    """Apply comparison operator to values."""
    match operator:
        case 'geq':
            return values >= threshold
        case 'lt':
            return values < threshold
        case 'gt':
            return values > threshold
        case 'leq':
            return values <= threshold
        case 'eq':
            return values == threshold
        case 'neq':
            return values != threshold
        case _:
            raise ValueError(f"Invalid operator: {operator}")


# =============================================================================
# Base Class
# =============================================================================


class Binarizer(ABC):
    """Abstract base class for all binarizers.

    :cvar include_complements: Whether to include complement rules.
    """

    include_complements: bool = False

    @abstractmethod
    def apply(self, values: np.ndarray, name: str) -> dict[str, np.ndarray]:
        """Transform column into binary rules.

        :param values: Feature values to transform.
        :param name: Feature name (used for rule naming).
        :return: Mapping of rule_name to boolean array.
        """

    def _rule_name(self, feature: str, op: str, value) -> str:
        """Build rule name like 'Age_geq_30'."""
        return f'{feature}{SEPARATOR}{op}{SEPARATOR}{value}'

    def _add_complement(self, rules: dict, rule_values: np.ndarray, rule_name: str, comp_name: str) -> dict:
        """Add complement rule if include_complements is True."""
        if self.include_complements:
            rules[comp_name] = ~rule_values
        return rules

# =============================================================================
# Binarizer Implementations
# =============================================================================


class BooleanBinarizer(Binarizer):
    """For boolean features (exactly 2 distinct values).

    Numeric (0/1): Creates 'Male', 'NotMale'
    String ("M"/"F"): Creates 'Gender_is_M', 'Gender_is_F'

    :param include_complements: Include complement rules (default False).
    """

    def __init__(self, include_complements: bool = False):
        self.include_complements = include_complements

    def apply(self, values: np.ndarray, name: str) -> dict[str, np.ndarray]:
        values = np.asarray(values)
        distinct, counts = np.unique(values, return_counts=True)
        if len(distinct) != 2:
            raise ValueError(f'{name}: expected 2 distinct values, got {len(distinct)}')

        if np.issubdtype(values.dtype, np.number):
            bool_values = (values == distinct[1])
            rule_name = name
            comp_name = f'Not{name}'
        else:
            # Use more common value as primary
            primary_idx = np.argmax(counts)
            primary_val = distinct[primary_idx]
            other_val = distinct[1 - primary_idx]
            bool_values = (values == primary_val)
            rule_name = self._rule_name(name, 'is', primary_val)
            comp_name = self._rule_name(name, 'is', other_val)

        rules = {rule_name: bool_values}
        out = self._add_complement(rules, bool_values, rule_name, comp_name)
        return out


class CategoricalBinarizer(Binarizer):
    """For categorical features (>2 distinct string values).

    Creates:
    - Single: 'Job_is_manager', 'Job_isnot_manager'
    - Subset: 'Job_in_{a,b}', 'Job_notin_{a,b}'

    :param categories: Reverse mapping from output values to input values.
        Format: {output_value: [input_values]}.
        Example: {-1: ['<=50K'], 1: ['>50K']}.
    :param max_subset_size: Maximum subset size for 'in' rules (default 1).
    :param min_support: Minimum fraction of samples for a rule (default 0.0).
    :param include_complements: Include complement rules (default False).
    """

    def __init__(self,
                 categories: dict | None = None,
                 max_subset_size: int = 1,
                 min_support: float = 0.0,
                 include_complements: bool = False
                 ):
        self.categories = categories
        self.max_subset_size = max_subset_size
        self.min_support = min_support
        self.include_complements = include_complements
        if categories is not None:
            self._reverse_map = {k: v for v, keys in categories.items() for k in keys}

    def apply(self, values: np.ndarray, name: str) -> dict[str, np.ndarray]:
        unique_values = np.unique(values)

        # Apply categories mapping if provided
        if self.categories is not None:
            mapping_keys = np.array(list(self._reverse_map.keys()))
            if len(unmapped := np.setdiff1d(unique_values, mapping_keys)):
                warnings.warn(
                    f"{name}: values not in categories will pass through unchanged: {unmapped.tolist()}"
                )
            if len(unused := np.setdiff1d(mapping_keys, unique_values)):
                warnings.warn(
                    f"{name}: category inputs not found in data: {unused.tolist()}"
                )
            values = np.array([self._reverse_map.get(v, v) for v in values])
            unique_values = np.unique(values)

        min_count = np.ceil(self.min_support * len(values))
        max_size = min(self.max_subset_size, len(unique_values) - 1)

        rules = {}
        for size in range(1, max_size + 1):
            for subset in combinations(unique_values, size):
                bool_values = np.isin(values, subset)
                if np.sum(bool_values) >= min_count:
                    if size == 1:
                        rule_name = self._rule_name(name, 'is', subset[0])
                        comp_name = self._rule_name(name, 'isnot', subset[0])
                    else:
                        subset_str = '{' + ','.join(str(c) for c in subset) + '}'
                        rule_name = self._rule_name(name, 'in', subset_str)
                        comp_name = self._rule_name(name, 'notin', subset_str)
                    rules[rule_name] = bool_values
                    self._add_complement(rules, bool_values, rule_name, comp_name)

        return rules


def _format_threshold(t) -> str:
    """Name a threshold by its own value: 5.0 -> '5', 5.5 -> '5.5'."""
    return str(int(t)) if float(t).is_integer() else f'{t:.6g}'


class NumericBinarizer(Binarizer):
    """For numeric features. Creates threshold rules.

    Supports multiple input styles:
    - operator='geq', thresholds=[30, 50]  (simple threshold list)
    - rules=[('leq', 25), ('geq', 55)]  (operator-value tuples)
    - rules=[('leq', 24), (25, 54), ('geq', 55)]  (with range tuples)
    - bins=['[30,inf)', '[50,inf)']  (interval notation)

    Output: 'Age_geq_30', 'Age_lt_30'

    :param operator: Operator for thresholds: 'geq' (default), 'lt', 'gt', 'leq', 'eq', 'neq'.
    :param thresholds: Explicit threshold values.
    :param rules: List of rule tuples. Each rule is either (operator, value) or
        (low, high) for a range rule (low <= x <= high).
    :param bins: Interval notation strings like '[30,inf)'.
    :param n_bins: Number of bins for auto-generation (default 5).
    :param strategy: Binning strategy: 'quantile', 'uniform', 'all'.
    :param preprocessor: Preprocessor to apply before binarization (e.g., Ordinal).
    :param include_complements: Include complement rules (default False).
    """

    def __init__(self, operator: str = 'geq', thresholds: list | None = None,
                 rules: list | None = None, bins: list | None = None, n_bins: int | None = 5,
                 strategy: str | None = 'quantile', preprocessor=None,
                 include_complements: bool = False):
        self.operator = operator
        self.thresholds = thresholds
        self.rules = rules
        self.bins = bins
        self.n_bins = n_bins
        self.strategy = strategy
        self.preprocessor = preprocessor
        self.include_complements = include_complements

        if operator not in ('geq', 'lt', 'gt', 'leq', 'eq', 'neq'):
            raise ValueError(f"operator must be 'geq', 'lt', 'gt', 'leq', 'eq', or 'neq', got {operator}")

    def apply(self, values: np.ndarray, name: str) -> dict[str, np.ndarray]:
        values = np.asarray(values)

        if self.preprocessor is not None:
            values = self.preprocessor.apply(values)

        distinct = np.unique(values)
        is_int = np.all(np.mod(values, 1) == 0)

        if self.rules is not None:
            rules = self._transform_from_rules(values, name, is_int)
        elif self.bins is not None:
            rules = self._transform_from_bins(values, name, is_int)
        else:
            rules = self._transform_from_thresholds(values, name, distinct)

        return rules

    def _transform_from_thresholds(self, values: np.ndarray, name: str, distinct: np.ndarray) -> dict:
        """Transform using thresholds and operator."""
        data_driven = self.thresholds is None
        if self.thresholds is not None:
            thresholds = sorted(self.thresholds)
        elif self.strategy == 'all':
            thresholds = list(distinct[1:])
        elif self.strategy == 'quantile':
            percentiles = np.linspace(0, 100, cast(int, self.n_bins) + 1)[1:-1]
            thresholds = list(np.unique(np.percentile(values, percentiles)))
        else:  # uniform
            thresholds = list(np.linspace(distinct[0], distinct[-1], cast(int, self.n_bins) + 1)[1:-1])

        rules = {}
        comp_op = COMPLEMENTS[self.operator]

        for t in thresholds:
            t_str = _format_threshold(t)

            bool_values = _compare(values, self.operator, t)
            # A data-driven cut on a skewed feature (e.g. a quantile landing on a
            # zero-inflated counter's minimum) can produce an all-True/all-False
            # column: information-free, and it inflates nominal d / the Rashomon
            # set. Drop such degenerate cuts; explicit expert thresholds are kept
            # verbatim (the author's choice, even at a boundary).
            if data_driven and (bool_values.all() or not bool_values.any()):
                continue
            rule_name = self._rule_name(name, self.operator, t_str)
            comp_name = self._rule_name(name, comp_op, t_str)
            rules[rule_name] = bool_values
            self._add_complement(rules, bool_values, rule_name, comp_name)

        return rules

    def _transform_from_rules(self, values: np.ndarray, name: str, is_int) -> dict:
        """Transform using rules list."""
        rules = {}

        for rule in cast(list, self.rules):
            if len(rule) != 2:
                raise ValueError(f"Rule must be (operator, value) or (low, high), got {rule}")

            first, second = rule
            is_range = isinstance(first, (int, float)) and isinstance(second, (int, float))

            if is_range:
                low, high = first, second
                bool_values = (values >= low) & (values <= high)
                low_str = _format_threshold(low)
                high_str = _format_threshold(high)
                rule_name = self._rule_name(name, 'in', f'[{low_str},{high_str}]')
                comp_name = self._rule_name(name, 'notin', f'[{low_str},{high_str}]')
            else:
                op, t = first, second
                if op not in ('geq', 'lt', 'gt', 'leq', 'eq', 'neq'):
                    raise ValueError(f"Invalid operator in rule: {op}")

                t_str = _format_threshold(t)
                bool_values = _compare(values, op, t)
                comp_op = COMPLEMENTS[op]
                rule_name = self._rule_name(name, op, t_str)
                comp_name = self._rule_name(name, comp_op, t_str)

            rules[rule_name] = bool_values
            self._add_complement(rules, bool_values, rule_name, comp_name)

        return rules

    def _transform_from_bins(self, values: np.ndarray, name: str, is_int) -> dict:
        """Transform using bin notation."""
        rules = {}
        for bin_str in cast(list, self.bins):
            b = Bin.from_string(bin_str, integer=is_int)
            bool_values = np.array([v in b for v in values])
            rule_name = name + b.rule_string()
            comp_name = name + b.rule_string(for_complement=True)
            rules[rule_name] = bool_values
            self._add_complement(rules, bool_values, rule_name, comp_name)

        return rules


# =============================================================================
# Preprocessors
# =============================================================================


class Ordinal:
    """Maps raw values to ordered indices.

    Use as a preprocessor for NumericBinarizer to handle ordinal data.

    :param levels: Mapping of level names to raw values. Keys are level names
        (in order from lowest to highest), values are lists of raw values.
        Example: {'Low': ['a', 'b'], 'Medium': ['c'], 'High': ['d', 'e']}
    """

    def __init__(self, levels: dict):
        self.levels = levels
        self.ordering = list(levels.keys())
        self._reverse_mapping = {k: v for v, keys in levels.items() for k in keys}
        self._level_to_idx = {level: i for i, level in enumerate(self.ordering)}

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Transform raw values to ordered indices (0, 1, 2, ...).

        :param values: Raw values to transform.
        :return: Array of numeric indices.
        :raises ValueError: If a value is not found in levels mapping.
        """
        values = np.asarray(values)
        out = np.empty(len(values), dtype=np.int64)
        for i, v in enumerate(values):
            if v not in self._reverse_mapping:
                raise ValueError(f"Value {v!r} not found in levels mapping")
            level_name = self._reverse_mapping[v]
            out[i] = self._level_to_idx[level_name]
        return out


# =============================================================================
# Wrappers
# =============================================================================


class Categorical:
    """Converts binarizer to categorical feature.

    Validates that the binarizer produces mutually exclusive, exhaustive rules
    (exactly one rule is True for each sample).

    :param binarizer: Binarizer to wrap.
    """

    def __init__(self, binarizer):
        self.binarizer = binarizer

    def apply(self, values: np.ndarray, name: str) -> dict:
        """Apply binarizer and validate categorical output.

        :param values: Feature values to transform.
        :param name: Feature name.
        :return: Mapping of rule_name to boolean array.
        :raises ValueError: If rules are not mutually exclusive and exhaustive.
        """
        rules = self.binarizer.apply(values, name)
        if not rules:
            raise ValueError(f"{name}: binarizer produced no rules")
        column_totals = np.vstack(list(rules.values())).sum(axis = 0)
        bad_rows = np.not_equal(column_totals, 1)
        if np.any(bad_rows):
            raise ValueError(
                f"{name}: rules are not mutually exclusive and exhaustive. "
                f"{np.count_nonzero(bad_rows)} samples have sum != 1 (first few indices: {np.flatnonzero(bad_rows)[:5].tolist()})"
            )
        return rules


# =============================================================================
# Convenience Subclasses
# =============================================================================


#: Convenience factory for explicit threshold rules (e.g., ThresholdBinarizer('geq', [30, 45, 60]))
ThresholdBinarizer = partial(NumericBinarizer, strategy=None, bins=None, n_bins=None)


#: Convenience factory for bin/interval rules (e.g., BinBinarizer(['[30,inf)', '(-inf, 20]']))
BinBinarizer = partial(NumericBinarizer, strategy=None, thresholds=None, n_bins=None, operator='geq')


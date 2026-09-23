"""
Pipeline Module
===============

Provides the Pipeline class for chainable post-processing operations on binarized rules.

Adding a New Pipeline Operation
-------------------------------
1. Create a class inheriting from PipelineOp::

    class MyNewOp(PipelineOp):
        is_data_dependent = True  # Set to True if operation needs X

        def __init__(self, my_param):
            self.my_param = my_param

        def apply(self, rules: dict, X: np.ndarray | None = None) -> dict:
            # Modify rules dict and return it
            return filtered_rules

2. Add a convenience method to Pipeline::

    def my_new_op(self, my_param):
        return self._chain(MyNewOp(my_param))
"""

import warnings
from collections.abc import Callable
from functools import reduce

import numpy as np

from .names import RuleName


# =============================================================================
# Helpers
# =============================================================================


def _drop_trivial(rules: dict) -> dict:
    """Remove trivial (all-true or all-false) rules.

    :param rules: Mapping of rule_name to boolean array.
    :return: Filtered rules dict.
    """
    def is_nontrivial(values):
        if len(values) <= 1 or values.dtype != np.bool_:
            return True
        return not (np.all(values) or not np.any(values))

    return {name: values for name, values in rules.items() if is_nontrivial(values)}


def _drop_duplicates(rules: dict, warn: bool = False, error: bool = False) -> dict:
    """Remove duplicate rules, keeping the first occurrence of each mask.

    :param rules: Mapping of rule_name to boolean array.
    :param warn: If True, emit a UserWarning listing dropped duplicates.
    :param error: If True, raise ValueError on duplicates instead of dropping.
        Takes precedence over ``warn``.
    :return: Filtered rules dict with duplicates removed.
    """
    if len(rules) < 2:
        return rules

    rule_names = list(rules.keys())
    rule_values = list(rules.values())
    _, inverse = np.unique(np.vstack(rule_values), axis=0, return_inverse=True)

    seen = {}
    duplicates = []
    for i, group_id in enumerate(inverse):
        if group_id in seen:
            duplicates.append((rule_names[i], rule_names[seen[group_id]]))
        else:
            seen[group_id] = i

    if duplicates:
        if error:
            raise ValueError(f"Duplicate rules found: {duplicates}")
        if warn:
            warnings.warn(f"Duplicate rules found, keeping first: {duplicates}")
        for dup_name, _ in duplicates:
            rules.pop(dup_name, None)

    return rules


class EmptyRulesWarning(UserWarning):
    """Warning raised when a pipeline operation filters out all rules."""


def _check_empty_rules(rules: dict, op_name: str) -> None:
    """Warn if rules dict is empty after an operation.

    :param rules: Rules dictionary to check.
    :param op_name: Name of the operation for the warning message.
    """
    if len(rules) == 0:
        warnings.warn(f"{op_name} filtered out all rules. Result is empty.", EmptyRulesWarning)


def _compute_stat(values: np.ndarray, stat: str | Callable) -> float:
    """Compute a statistic for rule values.

    :param values: Boolean array of rule values.
    :param stat: Statistic name ('support', 'gini') or callable.
    :return: Computed statistic value.
    :raises ValueError: If stat is not recognized.
    """
    support = float(np.mean(values))
    match stat:
        case 'support':
            return support
        case 'gini':
            return 2 * support * (1 - support)
        case _ if callable(stat):
            return stat(values)
        case _:
            raise ValueError(f"Unknown statistic: {stat}. Use 'support' or 'gini'.")


# =============================================================================
# Pipeline Operations
# =============================================================================


class PipelineOp:
    """Base class for pipeline operations.

    Subclasses must implement ``apply()`` which takes a rules dict and optionally
    data X, and returns a modified rules dict.

    :cvar is_data_dependent: If True, the operation requires X data to compute.
    """

    is_data_dependent: bool = False

    def apply(self, rules: dict, X: np.ndarray | None = None) -> dict:
        """Apply operation to rules dict.

        :param rules: Mapping of rule_name to boolean array.
        :param X: Original feature data (required if is_data_dependent=True).
        :return: Modified rules dict.
        """
        raise NotImplementedError("Subclasses must implement apply()")


class AddComplementsOp(PipelineOp):
    """Add complement rules (logical NOT) for each rule.

    This operation is idempotent - applying it twice has no effect because
    it only adds complements that are missing.
    """

    def apply(self, rules: dict, X: np.ndarray | None = None) -> dict:
        complements = {
            RuleName.parse_complement(name): ~values
            for name, values in rules.items()
            if RuleName.parse_complement(name) not in rules
        }
        return {**rules, **complements}


class DropTrivialOp(PipelineOp):
    """Remove rules that are all-true or all-false.

    This operation is idempotent - applying it twice has no effect.
    """

    is_data_dependent = True

    def apply(self, rules: dict, X: np.ndarray | None = None) -> dict:
        out = _drop_trivial(rules)
        _check_empty_rules(out, 'DropTrivialOp')
        return out


class DropDuplicatesOp(PipelineOp):
    """Remove duplicate rules (identical boolean arrays).

    Always keeps the first occurrence of each mask. Use ``warn`` / ``error``
    to control notification when duplicates are found.

    This operation is idempotent - applying it twice has no effect.

    :param warn: If True, emit a UserWarning listing dropped duplicates.
    :param error: If True, raise ValueError on duplicates instead of dropping.
        Takes precedence over ``warn``.
    """

    is_data_dependent = True

    def __init__(self, warn: bool = False, error: bool = False):
        self.warn = warn
        self.error = error

    def apply(self, rules: dict, X: np.ndarray | None = None) -> dict:
        out = _drop_duplicates(rules, warn=self.warn, error=self.error)
        _check_empty_rules(out, 'DropDuplicatesOp')
        return out


class RequireOp(PipelineOp):
    """Filter rules by requiring a statistic to be within bounds.

    This operation is idempotent - applying it twice has no effect.

    :param stat: Statistic to filter by: 'support' or 'gini'.
    :param min: Minimum value. Rules below this are dropped.
    :param max: Maximum value. Rules above this are dropped.

    Note
    ----
    - Support = mean(rule values)
    - Gini = 2 * p * (1 - p) where p = support

    Example
    -------
    >>> RequireOp(stat='support', min=0.1, max=0.9)
    >>> RequireOp(stat='gini', min=0.2)
    """

    is_data_dependent = True

    def __init__(self, stat: str, min: float | None = None, max: float | None = None):
        if stat not in ('support', 'gini'):
            raise ValueError(f"Unknown stat: {stat}. Use 'support' or 'gini'.")
        self.stat = stat
        self.min = min
        self.max = max

    def apply(self, rules: dict, X: np.ndarray | None = None) -> dict:
        def in_bounds(values):
            stat_value = _compute_stat(values, self.stat)
            if self.min is not None and stat_value < self.min:
                return False
            if self.max is not None and stat_value > self.max:
                return False
            return True

        out = {name: values for name, values in rules.items() if in_bounds(values)}
        _check_empty_rules(out, 'RequireOp')
        return out


class TopNOp(PipelineOp):
    """Keep only top N rules by a given statistic.

    This operation is idempotent - applying it twice with the same N has no
    effect because the rules are already the top N.

    :param stat: Statistic to rank by: 'support', 'gini', or custom callable.
    :param n: Number of rules to keep.
    :param ascending: If True, keep lowest N instead of highest N.

    Example
    -------
    >>> TopNOp(stat='support', n=10)  # Top 10 by support
    >>> TopNOp(stat='gini', n=5)      # Top 5 by gini
    """

    is_data_dependent = True

    def __init__(self, stat: str, n: int, ascending: bool = False):
        self.stat = stat
        self.n = n
        self.ascending = ascending

    def apply(self, rules: dict, X: np.ndarray | None = None) -> dict:
        # If n >= number of rules, return all rules unchanged (idempotent)
        if len(rules) <= self.n:
            return rules

        # Compute statistic for each rule
        stats = {name: _compute_stat(values, self.stat) for name, values in rules.items()}

        # Sort and take top N
        sorted_names = sorted(stats.keys(), key=lambda x: stats[x],
                             reverse=not self.ascending)
        top_names = sorted_names[:self.n]

        out = {name: rules[name] for name in top_names}
        _check_empty_rules(out, 'TopNOp')
        return out


# =============================================================================
# Pipeline Class
# =============================================================================


class Pipeline:
    """Wraps a binarizer with chainable post-processing operations.

    A Pipeline stores a base binarizer and a sequence of operations to apply
    after transformation. The pipeline is lazy - operations are stored and
    applied when transform() is called.

    :param binarizer: The base binarizer to wrap.
    :param ops: Operations to apply after binarization.

    Example
    -------
    >>> from riskslim.data import ThresholdBinarizer
    >>> pipeline = (
    ...     ThresholdBinarizer('geq', [30, 45, 60])
    ...     .add_complements()
    ...     .require(stat='support', min=0.05)
    ...     .drop_trivial()
    ... )
    >>> # Use pipeline in Processor
    >>> processor = Processor({'Age': pipeline})

    Note
    ----
    Pipeline implements the same interface as Binarizer, so it can be used
    anywhere a Binarizer is expected.
    """

    def __init__(self, binarizer, ops: list | None = None):
        self.binarizer = binarizer
        self.ops = ops or []

    @property
    def is_data_dependent(self) -> bool:
        """True if any operation requires data to compute."""
        return any(op.is_data_dependent for op in self.ops)

    def _chain(self, op: PipelineOp) -> 'Pipeline':
        """Return new Pipeline with operation appended."""
        return Pipeline(self.binarizer, self.ops + [op])

    # =========================================================================
    # Chainable methods
    # =========================================================================

    def add_complements(self) -> 'Pipeline':
        """Add complement rules (logical NOT) for each rule."""
        return self._chain(AddComplementsOp())

    def drop_trivial(self) -> 'Pipeline':
        """Remove rules that are all-true or all-false."""
        return self._chain(DropTrivialOp())

    def drop_duplicates(self, warn: bool = False, error: bool = False) -> 'Pipeline':
        """Remove duplicate rules, keeping the first occurrence of each mask.

        :param warn: If True, emit a UserWarning listing dropped duplicates.
        :param error: If True, raise ValueError on duplicates instead of dropping.
            Takes precedence over ``warn``.
        """
        return self._chain(DropDuplicatesOp(warn=warn, error=error))

    def require(self, stat: str, min: float | None = None, max: float | None = None) -> 'Pipeline':
        """Filter rules by requiring a statistic to be within bounds.

        :param stat: Statistic to filter by: 'support' or 'gini'.
        :param min: Minimum value. Rules below this are dropped.
        :param max: Maximum value. Rules above this are dropped.

        Example
        -------
        >>> pipeline.require(stat='support', min=0.1, max=0.9)
        """
        return self._chain(RequireOp(stat=stat, min=min, max=max))

    def top_n(self, stat: str, n: int, ascending: bool = False) -> 'Pipeline':
        """Keep only top N rules by a given statistic.

        :param stat: Statistic to rank by: 'support', 'gini', or custom callable.
        :param n: Number of rules to keep.
        :param ascending: If True, keep lowest N instead of highest N.
        """
        return self._chain(TopNOp(stat=stat, n=n, ascending=ascending))

    def bottom_n(self, stat: str, n: int) -> 'Pipeline':
        """Keep only bottom N rules by a given statistic.

        Convenience method equivalent to top_n(stat, n, ascending=True).

        :param stat: Statistic to rank by: 'support', 'gini', or custom callable.
        :param n: Number of rules to keep.
        """
        return self._chain(TopNOp(stat=stat, n=n, ascending=True))

    # =========================================================================
    # Apply interface (matches Binarizer)
    # =========================================================================

    def apply(self, values: np.ndarray, name: str) -> dict:
        """Apply binarizer and all pipeline operations.

        :param values: Feature values to transform.
        :param name: Feature name (used for rule naming).
        :return: Mapping of rule_name to boolean array, after all operations applied.
        """
        rules = self.binarizer.apply(values, name)
        return reduce(
            lambda r, op: op.apply(r, values if op.is_data_dependent else None),
            self.ops,
            rules
        )

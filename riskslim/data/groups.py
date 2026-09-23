"""
Group attribute utilities for classification datasets.

This module provides:
- GroupInfo: Lightweight metadata/inspection for group attributes
- EncodingType: Enum for encoding types (onehot, intersectional)
- GroupAttributeEncoder: Encoder for group attributes to dummy variables
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from itertools import product
from typing import Literal, overload
from .names import SEPARATOR

INTERSECT_SEPARATOR = '&'   # Between groups: "Sex_Male&Race_White"


from riskslim.data.array_utils import EncodingType


@dataclass
class GroupInfo:
    """Lightweight object for group attribute analysis.

    Provides access to group metadata without encoding overhead.
    Works with ALL G columns from _df_out (even excluded ones).

    :param df: DataFrame containing G columns for analysis.

    ----
    Representation Invariants:
    - df is a DataFrame with 0+ rows and 0+ columns
    - Each column has >= 2 unique values (if columns exist)
    - All values are non-null
    """
    df: pd.DataFrame = field(repr=False)

    def __post_init__(self):
        assert self.__check_rep__()

    def __check_rep__(self):
        """Check representation invariants."""
        assert isinstance(self.df,
                          pd.DataFrame), f"df must be DataFrame, got {type(self.df).__name__}"
        for k in self.names:
            assert not self.df[k].isna().any(), f"Column '{k}' has null values"
            m = self.df[k].nunique()
            assert m >= 2, f"Column '{k}' must have ≥ 2 unique values, got {m}"
        return True

    def __repr__(self):
        return f"GroupInfo(n={len(self.df)}, columns={self.names}, n_groups={self.n_groups})"

    @property
    def names(self) -> list:
        return self.df.columns.tolist()

    @property
    def empty(self) -> bool:
        return len(self.names) == 0

    @property
    def labels(self) -> dict:
        """Unique values per column: {'sex': ['F', 'M'], 'race': ['A', 'B', 'C']}"""
        return {k: sorted(self.df[k].unique().tolist()) for k in self.names}

    @property
    def counts(self) -> dict:
        """Value counts per column: {'sex': {'M': 12, 'F': 8}, 'race': {'A': 5, 'B': 10, 'C': 5}}
        """
        return {k: self.df[k].value_counts().to_dict() for k in self.names}

    @property
    def combinations(self) -> list:
        """
        List of tuples containing intersectional group combinations
        Each tuple contains one value from each group column, in column order.
        Returns empty list if there are no group columns.
        Example: [('F', 'A'), ('F', 'B'), ('F', 'C'), ('M', 'A'), ('M', 'B'), ('M', 'C')]
        """
        return self.get_combinations(columns=None)

    @property
    def n_groups(self) -> int:
        """Number of intersectional groups. Returns 0 if there are no group columns."""
        return len(self.combinations)

    def to_indices(self, columns: list | None = None) -> np.ndarray:
        """Integer group ID (0 to n_groups-1) per sample.

        :param columns: Column names to use for grouping. If None, uses all columns.
        :returns: Array of integers representing group membership.
        Example:

            # All groups (for analysis)
            indices = data.groups.to_indices()

            # Included groups only (for training consistency)
            indices = data.groups.to_indices(columns=data.names.G)
        """
        cols = self.names if columns is None else list(columns)
        if len(cols) == 0:
            out = np.zeros(len(self.df), dtype=np.int_)
        else:
            assert set(cols).issubset(self.names), f"Columns not found: {set(cols) - set(self.names)}"
            combos = self.get_combinations(columns=cols)
            combo_to_idx = {combo: i for i, combo in enumerate(combos)}
            tuples = zip(*[self.df[k].values for k in cols])
            out = np.array([combo_to_idx[t] for t in tuples], dtype=np.int_)
        return out

    def get_combinations(self, columns: list | None = None) -> list:
        """Get intersectional combinations for specified columns.

        :param columns: Column names to use. If None, uses all columns.
        :returns: List of tuples representing all group combinations.

        Example::

            # All combinations
            data.groups.get_combinations()  # [('F', 'A'), ('F', 'B'), ...]

            # Included columns only
            data.groups.get_combinations(columns=data.names.G)
        """
        cols = self.names if columns is None else list(columns)
        out = []
        if len(cols) > 0:
            assert set(cols).issubset(
                    self.names), f"Columns not found: {set(cols) - set(self.names)}"
            label_lists = [sorted(self.df[k].unique().tolist()) for k in cols]
            out = list(product(*label_lists))
        return out


@dataclass
class GroupAttributeEncoder:
    """
    Encoder that converts a DataFrame of group attributes into numerical values.
    Supports two encoding types:
    - ONEHOT: Each label becomes a separate binary column
    - INTERSECTIONAL: Each unique combination of labels becomes a column

    :param df: DataFrame where each column is a categorical group attribute.
    :param encoding_type: EncodingType.ONEHOT or EncodingType.INTERSECTIONAL.
    :param labels: Optional dict {column: labels} to specify label order.

    ----
    Usage: used in BinaryClassificationDataset to encode group attributes for analysis.
    Example usage:
        encoder = GroupAttributeEncoder(df[['Sex', 'Race']], encoding_type=EncodingType.INTERSECTIONAL)
        dummies = encoder.to_dummies(df[['Sex', 'Race']])
        indices = encoder.to_indices(df[['Sex', 'Race']])

    Used in training.py to encode group attributes for training.
    """
    df: pd.DataFrame = field(repr=False)
    encoding_type: EncodingType = EncodingType.ONEHOT
    labels: dict = field(default_factory=dict)
    # Computed post-init
    names: list = field(init=False, repr=False)
    groups: list = field(init=False, repr=False)
    dummy_names: list = field(init=False, repr=False, default_factory=list)
    dtypes: dict = field(init=False, repr=False, default_factory=dict)
    indexer: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        self.encoding_type = EncodingType(self.encoding_type)
        self.names = self.df.columns.tolist()

        # Build labels: use provided or infer from df
        # Note: extra keys in self.labels (not in df columns) are silently ignored
        labels_out = {}
        for name in self.names:
            if name in self.labels:
                col_labels = self.labels[name]
                assert self.df[name].isin(col_labels).all(), f"df[{name}] includes labels not in provided labels"
            else:
                col_labels = self.df[name].unique()
                assert len(col_labels) >= 2, f"group attribute {name} must have at least 2 distinct labels"
            labels_out[name] = sorted(col_labels)
        self.labels = labels_out

        # Derived fields
        self.dtypes = {n: pd.CategoricalDtype(categories=lbls) for n, lbls in self.labels.items()}
        self.groups = list(product(*self.labels.values())) if self.labels else []

        if self.labels:
            cols = [[f"{n}{SEPARATOR}{v}" for v in self.labels[n]] for n in self.names]
            match self.encoding_type:
                case EncodingType.ONEHOT:
                    self.dummy_names = [name for col in cols for name in col]
                case EncodingType.INTERSECTIONAL:
                    self.dummy_names = [INTERSECT_SEPARATOR.join(c) for c in product(*cols)]
                    self.indexer = {g: i for i, g in enumerate(self.groups)}

    @overload
    def to_dummies(self, df: pd.DataFrame, return_names: Literal[False] = ...) -> np.ndarray: ...
    @overload
    def to_dummies(self, df: pd.DataFrame, return_names: Literal[True]) -> tuple[np.ndarray, list[str]]: ...
    def to_dummies(self, df, return_names=False):
        """
        Convert DataFrame to dummy variables.

        :param df: DataFrame with group attributes.
        :param return_names: If True, return (values, names) tuple.
        :returns: Dummy values array, or (values, names) if return_names=True.
        """
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(self.names)
        match self.encoding_type:
            case EncodingType.ONEHOT:
                indices = pd.get_dummies(df[self.names].astype(self.dtypes), prefix=self.names, prefix_sep=SEPARATOR)
                values = indices.values.astype(np.int_)
            case EncodingType.INTERSECTIONAL:
                indices = self.to_indices(df)
                values = np.array([np.isin(indices, k) for k in self.indexer.values()]).T.astype(np.int_)
        out = (values, self.dummy_names) if return_names else values
        return out

    def to_indices(self, df):
        """
        Convert DataFrame to integer indices representing intersectional groups.

        :param df: DataFrame with group attributes.
        :returns: Array of integers from 0 to len(self.groups)-1.
        """
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(self.names)
        if len(self.names) == 0:
            out = np.zeros(len(df), dtype=np.int_)
        else:
            group_to_index = {g: i for i, g in enumerate(self.groups)}
            tuples = zip(*[df[col].values for col in self.names])
            out = np.array([group_to_index[t] for t in tuples], dtype=np.int_)
        return out

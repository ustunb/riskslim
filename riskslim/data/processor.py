"""
Processor Module
================

Provides the Processor class for managing feature->binarizer mappings
and marker classes for special feature handling.
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .names import Names

if TYPE_CHECKING:
    from .data import BinaryClassificationDataset


# =============================================================================
# Marker Classes
# =============================================================================


class Label:
    """Marker for features that are outcome/label columns.

    Features marked with Label are excluded from binarization and used
    as the target variable. Supports encoding raw values to target classes.

    :param recode: Target label values as tuple (neg_class, pos_class).
        If None, preserve original values. If provided (e.g., (-1, 1) or (0, 1)),
        recode labels so that the smaller unique value maps to neg_class
        and the larger unique value maps to pos_class. Default is (-1, 1).
    :param binarizer: Optional binarizer for categorical labels. When provided,
        the binarizer's categories/mapping keys define valid label values.
        Use with recode dict for explicit mapping from categories to targets.

    Example
    -------
    >>> # Recode to (-1, 1) (default)
    >>> Label()
    >>>
    >>> # Recode to (0, 1)
    >>> Label(recode=(0, 1))
    >>>
    >>> # Preserve original values
    >>> Label(recode=None)
    >>>
    >>> # Categorical binarizer (categories are label values)
    >>> Label(binarizer=CategoricalBinarizer(categories=['<=50K', '>50K']))
    >>>
    >>> # Explicit mapping with binarizer
    >>> Label(binarizer=CategoricalBinarizer(), recode={'<=50K': -1, '>50K': 1})
    """

    def __init__(self, recode: tuple | dict | None = (-1, 1), binarizer=None):
        self.recode = recode
        self.binarizer = binarizer
        self._include = True

    @property
    def include(self) -> bool:
        """Whether label is included in output. Always True for Label."""
        return self._include

    @include.setter
    def include(self, value: bool):
        if not value:
            raise ValueError("Cannot exclude label column from training data")
        self._include = True

    def apply(self, values: np.ndarray, name: str) -> np.ndarray:
        """Transform raw label values.

        :param values: Raw label values.
        :param name: Feature name (used for error messages).
        :return: Array of transformed label values.
        """
        values = np.asarray(values)

        if isinstance(self.recode, dict):
            try:
                result = np.array([self.recode[v] for v in values])
            except KeyError as e:
                valid_keys = set(self.recode.keys())
                actual_values = set(np.unique(values))
                missing = actual_values - valid_keys
                raise ValueError(
                    f"Label recode for '{name}' missing keys: {missing}. "
                    f"Recode covers: {valid_keys}"
                ) from e
        elif self.recode is not None:
            unique_vals = sorted(np.unique(values))
            if len(unique_vals) > 2:
                raise ValueError(
                    f"Label for '{name}': binary labels expected but got "
                    f"{len(unique_vals)} unique values: {unique_vals}"
                )
            if values.dtype == np.bool_:
                unique_vals = [False, True]
            neg_class, pos_class = self.recode
            if len(unique_vals) == 1:
                val = unique_vals[0]
                if val in (0, False, -1):
                    result = np.full(len(values), neg_class, dtype=np.int_)
                else:
                    result = np.full(len(values), pos_class, dtype=np.int_)
            else:
                mapping = {unique_vals[0]: neg_class, unique_vals[1]: pos_class}
                result = np.array([mapping[v] for v in values], dtype=np.int_)
        else:
            unique_vals = set(np.unique(values))
            if unique_vals <= {-1, 1} or unique_vals <= {0, 1}:
                result = values.astype(np.int_)
            else:
                raise ValueError(
                    f"Label for '{name}': values must be {{-1, 1}} or {{0, 1}}. "
                    f"Got unique values: {np.unique(values)}"
                )

        return result


class Feature:
    """Marker for features that pass through unchanged (not binarized).

    Use Feature for columns that should be preserved in the output without
    transformation. These columns are added to _df_out but their inclusion
    in the final training data (_df) is controlled by the include flag.

    :param include: Whether to include in training data (X matrix).
        Default is True. Set to False for metadata columns.

    Example
    -------
    >>> # Include in X (default)
    >>> processor = Processor({
    ...     'outcome': Label(),
    ...     'age': NumericBinarizer(thresholds=[30, 50]),
    ...     'raw_score': Feature(),  # Include in training
    ... })
    >>>
    >>> # Exclude from X (metadata)
    >>> processor = Processor({
    ...     'outcome': Label(),
    ...     'age': NumericBinarizer(thresholds=[30, 50]),
    ...     'patient_id': Feature(include=False),  # Metadata, exclude from training
    ... })
    """

    def __init__(self, include: bool = True):
        self._include = include

    @property
    def include(self) -> bool:
        """Whether feature is included in X matrix."""
        return self._include

    @include.setter
    def include(self, value: bool):
        self._include = value

    def apply(self, values: np.ndarray, name: str) -> dict:
        """Pass through values unchanged.

        :param values: Feature values to pass through.
        :param name: Feature name (used as column name).
        :return: Dict mapping name to values (same format as binarizers).
        """
        out = {name: np.asarray(values)}
        return out


class GroupAttribute:
    """Marker for features that become group attributes (excluded from X).

    Can be used with a binarizer wrapper or with a groups dict for collapsing.

    :param binarizer: Binarizer to wrap. If provided, uses its configuration
        for transformation. For CategoricalBinarizer with mapping, applies
        the mapping to values.
    :param include: Whether to also include in X matrix (default False).
    :param groups: Mapping of output group names to input values.
        Format: {output: [inputs]}. E.g., {'Young': ['0-18', '19-30']}.

    Example
    -------
    >>> # Simple passthrough
    >>> GroupAttribute()
    >>>
    >>> # With include in X
    >>> GroupAttribute(include=True)
    >>>
    >>> # With groups (collapse categories)
    >>> GroupAttribute(groups={'Young': ['0-18', '19-30'], 'Old': ['60+']})
    >>>
    >>> # With binarizer (wrapper pattern)
    >>> GroupAttribute(binarizer=CategoricalBinarizer())
    >>> GroupAttribute(binarizer=CategoricalBinarizer(mapping={'A': 'X', 'B': 'X'}))
    >>>
    >>> # Binarizer with include in X
    >>> GroupAttribute(binarizer=BooleanBinarizer(), include=True)
    """

    def __init__(self, binarizer=None, include: bool = False, groups: dict | None = None):
        self.binarizer = binarizer
        self._include = include
        self.groups = groups

    @property
    def include(self) -> bool:
        """Whether group attribute is also included in X matrix."""
        return self._include

    @include.setter
    def include(self, value: bool):
        self._include = value

    def apply(self, values: np.ndarray, name: str) -> np.ndarray:
        """Transform values into group labels.

        :param values: Feature values to transform.
        :param name: Feature name (used for labeling).
        :return: 1D array of group labels (not dict like Binarizer).
        """
        if self.binarizer is not None and hasattr(self.binarizer, "categories") and self.binarizer.categories:
            mapping = {old: new for new, olds in self.binarizer.categories.items() for old in olds}
            return np.array([mapping.get(v, v) for v in values])

        if self.groups is not None:
            mapping = {old: new for new, olds in self.groups.items() for old in olds}
            return np.array([mapping.get(v, v) for v in values])

        return values


# =============================================================================
# Processor Class
# =============================================================================


class Processor:
    """Stores feature -> binarizer mapping. Handles transform and group extraction.

    :param mapping: Initial mapping of feature names to Binarizer or
        marker (Label, GroupAttribute, Feature) instances.

    Example
    -------
    >>> from riskslim.data import NumericBinarizer, BooleanBinarizer
    >>> from riskslim.data import Label, GroupAttribute, Feature
    >>> processor = Processor(mapping={
    ...     'Age': NumericBinarizer(thresholds=[30, 50]),
    ...     'Income': Label(),
    ...     'Sex': GroupAttribute(BooleanBinarizer()),
    ...     'Race': GroupAttribute(),
    ... })
    >>> df_out = processor.apply(df_raw)
    >>> X = df_out[processor.outputs.X].values
    >>> y = df_out[processor.outputs.y].values
    >>> G = df_out[processor.outputs.G]
    """

    def __init__(self, mapping: dict):
        # Auto-wrap plain Binarizers in Pipeline
        from .binarizers import Binarizer
        from .pipeline import Pipeline
        mapping = {
            f: Pipeline(b) if isinstance(b, Binarizer) and not isinstance(b, Pipeline) else b
            for f, b in mapping.items()
        }

        y_names = [f for f, b in mapping.items() if isinstance(b, Label)]
        G_names = [f for f, b in mapping.items() if isinstance(b, GroupAttribute)]
        # Feature(include=False) goes to carryover, Feature(include=True) goes to X
        C_names = [f for f, b in mapping.items() if isinstance(b, Feature) and not b.include]
        # X includes binarizers AND Feature(include=True)
        X_names = [f for f, b in mapping.items()
                   if not isinstance(b, (Label, GroupAttribute))
                   and not (isinstance(b, Feature) and not b.include)]

        if not y_names:
            raise ValueError("Processor requires an outcome (Label)")
        if not X_names:
            raise ValueError("Processor requires at least one feature")

        self._inputs = Names(X=X_names, y=y_names[0], G=G_names)
        self._inputs_carryover = C_names
        self._outputs = Names()
        self._outputs_carryover = []
        self._mapping = mapping
        self._slices = {}
        self._dataset: BinaryClassificationDataset | None = None

        # _include tracks "dropped" state - whether feature is included in training data
        # This is separate from GroupAttribute.include which controls X inclusion
        self._include = {
            feature: handler.include if isinstance(handler, Feature) else True
            for feature, handler in mapping.items()
        }

    @classmethod
    def from_names(cls, names: Names, *, label_recode=None) -> 'Processor':
        """Create Processor with default mapping from Names object.

        :param names: Names object with X, y, and G attributes.
        :param label_recode: Recode parameter for Label (default None preserves values).
        :returns: Processor with Label for y, GroupAttribute for G, Feature for X.
        """
        mapping: dict[str, Label | GroupAttribute | Feature] = {names.y: Label(recode=label_recode)}
        mapping.update({g: GroupAttribute() for g in names.G})
        mapping.update({x: Feature() for x in names.X})
        return cls(mapping=mapping)

    def __eq__(self, other):
        if not isinstance(other, Processor):
            return False
        return self._inputs == other._inputs and self._outputs == other._outputs

    def __getitem__(self, feature: str):
        return self._mapping[feature]

    def __setitem__(self, feature: str, binarizer):
        # Auto-wrap plain Binarizers in Pipeline
        from .binarizers import Binarizer
        from .pipeline import Pipeline
        if isinstance(binarizer, Binarizer) and not isinstance(binarizer, Pipeline):
            binarizer = Pipeline(binarizer)

        old_handler = self._mapping.get(feature)

        if isinstance(old_handler, Label) and not isinstance(binarizer, Label):
            raise NotImplementedError(
                "Label reassignment not yet supported. "
                "Demote to Feature(include=False) first."
            )

        if old_handler is not binarizer:
            self._mapping[feature] = binarizer

            old_is_label = isinstance(old_handler, Label)
            old_is_group = isinstance(old_handler, GroupAttribute)
            old_is_feature = isinstance(old_handler, Feature)
            new_is_label = isinstance(binarizer, Label)
            new_is_group = isinstance(binarizer, GroupAttribute)
            new_is_feature = isinstance(binarizer, Feature)

            if old_is_label != new_is_label or old_is_group != new_is_group or old_is_feature != new_is_feature:
                y_names = [f for f, b in self._mapping.items() if isinstance(b, Label)]
                self._inputs = Names(
                    X=[f for f, b in self._mapping.items()
                       if not isinstance(b, (Label, GroupAttribute))
                       and not (isinstance(b, Feature) and not b.include)],
                    y=y_names[0],
                    G=[f for f, b in self._mapping.items() if isinstance(b, GroupAttribute)],
                )
                self._inputs_carryover = [f for f, b in self._mapping.items()
                                          if isinstance(b, Feature) and not b.include]

            if isinstance(binarizer, Feature):
                self._include[feature] = binarizer.include
            elif isinstance(binarizer, Label):
                self._include[feature] = True
            elif isinstance(binarizer, GroupAttribute):
                self._include[feature] = binarizer.include
            else:
                self._include[feature] = True

            self._slices.clear()
            if self._dataset is not None:
                self._dataset._df_out = self.apply(self._dataset._df_raw)
                self._dataset._rebuild_from_df_out()

    def update(self, mapping: dict) -> None:
        """Batch update multiple feature->handler mappings.

        :param mapping: Dict of feature names to binarizer/marker instances.
        """
        for feature, handler in mapping.items():
            self[feature] = handler

    def apply(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Apply binarizers to raw data.

        :param df_raw: Raw feature matrix with named columns.
        :return: DataFrame with transformed columns (X, y, G combined).
            Use processor.outputs.X/y/G to extract specific parts.
        """
        # Auto-include unspecified columns as Feature(include=False)
        for col in df_raw.columns:
            if col not in self._mapping:
                self._mapping[col] = Feature(include=False)
                self._inputs_carryover.append(col)
                self._include[col] = False

        self._slices.clear()
        output_X_names = []
        output_G_names = []
        columns = {}

        for feature in self._inputs.X:
            binarizer = self._mapping[feature]
            rules = binarizer.apply(df_raw[feature].values, feature)
            rule_names = list(rules.keys())
            self._slices[feature] = rule_names
            output_X_names.extend(rule_names)
            columns.update(rules)

        feature = self._inputs.y
        self._slices[feature] = [feature]
        output_y_name = feature
        marker = self._mapping[feature]
        columns[feature] = marker.apply(df_raw[feature].values, feature)

        for feature in self._inputs.G:
            marker = self._mapping[feature]
            self._slices[feature] = [feature]
            output_G_names.append(feature)
            columns[feature] = marker.apply(df_raw[feature].values, feature)

            if marker.include:
                if marker.binarizer is None:
                    raise ValueError(
                        f"GroupAttribute for '{feature}' requires a binarizer when include=True"
                    )
                rules = marker.binarizer.apply(df_raw[feature].values, feature)
                rule_names = list(rules.keys())
                self._slices[feature] = [feature] + rule_names
                output_X_names.extend(rule_names)
                columns.update(rules)

        output_C_names = []
        for feature in self._inputs_carryover:
            marker = self._mapping[feature]
            result = marker.apply(df_raw[feature].values, feature)
            result_names = list(result.keys())
            self._slices[feature] = result_names
            output_C_names.extend(result_names)
            columns.update(result)

        self._outputs = Names(X=output_X_names, y=output_y_name, G=output_G_names)
        self._outputs_carryover = output_C_names

        all_names = output_X_names + [output_y_name] + output_G_names + output_C_names
        if not all_names:
            out = pd.DataFrame(index=range(len(df_raw)))
            return out

        out = pd.DataFrame({name: columns[name] for name in all_names})
        return out

    @property
    def inputs(self) -> Names:
        """Input feature names (X, y, G from mapping)."""
        return self._inputs

    @property
    def outputs(self) -> Names:
        """Output column names after apply() (X, y, G)."""
        return self._outputs

    def get_output_names(self) -> Names:
        """Return a new Names object from outputs (for assignment to dataset)."""
        return Names(X=self._outputs.X, y=self._outputs.y, G=self._outputs.G)

    @property
    def included(self) -> Names:
        """Names filtered to only included columns."""
        return Names(
            X=[x for x in self._outputs.X if x in self.included_columns],
            y=self._outputs.y,
            G=self.included_G_names,
        )

    @property
    def mapping(self) -> dict:
        """Feature name to binarizer mapping."""
        return self._mapping

    @property
    def slices(self) -> dict:
        """Mapping of input feature names to output column names."""
        out = self._slices.copy()
        return out

    def __contains__(self, feature: str) -> bool:
        out = feature in self._mapping
        return out

    def set_include(self, feature: str | list[str], value: bool) -> tuple[bool, bool]:
        """Set whether a feature's output columns are included in training data.

        Works for X features and G (group attribute) features.

        :param feature: Input feature name or list of names.
        :param value: True to include, False to exclude.
        :returns: Tuple of (changed, g_affected) where changed is True if any
            include state was modified, and g_affected is True if any G column was affected.
        :raises KeyError: If feature not in processor.
        :raises ValueError: If trying to exclude the label column.
        """
        if isinstance(feature, list):
            results = [self.set_include(f, value) for f in feature]
            changed = any(r[0] for r in results)
            g_affected = any(r[1] for r in results)
            return changed, g_affected
        if feature not in self._mapping:
            raise KeyError(f"Feature '{feature}' not in processor")
        if isinstance(self._mapping[feature], Label) and not value:
            raise ValueError("Cannot exclude label column from training data")
        changed = self._include[feature] != value
        self._include[feature] = value
        g_affected = isinstance(self._mapping[feature], GroupAttribute)
        return changed, g_affected

    def is_included(self, feature: str) -> bool:
        """Check if a feature is included in training data.

        :param feature: Input feature name.
        :return: True if included, False otherwise.
        :raises KeyError: If feature not in processor.
        """
        if feature not in self._mapping:
            raise KeyError(f"Feature '{feature}' not in processor")
        out = self._include[feature]
        return out

    @property
    def included_columns(self) -> list:
        """Output column names where include=True.

        Returns the subset of output columns (from apply()) that should be
        included in training data. Excludes columns from features where
        set_include(feature, False) was called.

        :return: List of output column names to include.
        """
        result = []
        for feature, output_cols in self._slices.items():
            if self._include.get(feature, True):
                result.extend(output_cols)
        return result

    @property
    def included_G_names(self) -> list:
        """G column names where include=True.

        Returns the subset of G input columns that should be included in
        training data. Used by Dataset to filter G columns.

        Note: The GroupAttribute.include flag controls whether a G column is
        ALSO binarized and added to X. The Processor._include flag (set via
        set_include()) controls whether the G column is "dropped" from training.

        :return: List of included G input names.
        """
        out = [g for g in self._inputs.G if self._include.get(g, True)]
        return out

    def __repr__(self) -> str:
        """Return header line plus full tabulate output."""
        n_in = len(self._mapping)
        n_out = len(self._outputs.X) + 1 + len(self._outputs.G)
        header = f"Processor({n_in} inputs → {n_out} outputs)"
        table = self.tabulate(view="out", include_dropped=True, return_df=False)
        out = f"{header}\n\n{table}"
        return out

    def tabulate(
        self,
        view: str = "out",
        include_dropped: bool = True,
        return_df: bool = False,
    ) -> pd.DataFrame | str:
        """Display column mappings as a table.

        :param view: 'out' shows raw→out mapping, 'raw' shows df→df_raw mapping.
        :param include_dropped: If False, hide excluded/dropped rows.
        :param return_df: If True, return DataFrame instead of string.
        :return: DataFrame if return_df=True, else formatted string.
        """
        rows = []
        x_index_lookup = {name: i for i, name in enumerate(self._outputs.X)}

        order = [self._inputs.y] + self._inputs.G + self._inputs.X + self._inputs_carryover

        for raw_name in order:
            handler = self._mapping[raw_name]
            is_dropped = not self._include.get(raw_name, True)

            if not include_dropped and is_dropped:
                continue

            if isinstance(handler, Label):
                marker_name = "Label"
                processor_name = "-"
            elif isinstance(handler, GroupAttribute):
                marker_name = "GroupAttribute"
                if handler.binarizer is not None:
                    processor_name = type(handler.binarizer).__name__
                else:
                    processor_name = "-"
            elif isinstance(handler, Feature):
                marker_name = "Feature"
                processor_name = "-"
            else:
                marker_name = "-"
                processor_name = type(handler).__name__

            out_names = self._slices.get(raw_name, [raw_name])

            for out_name in out_names:
                if out_name in x_index_lookup:
                    idx = x_index_lookup[out_name]
                else:
                    idx = "-"

                if view == "out":
                    row = {
                        "raw": raw_name,
                        "idx": idx,
                        "out": out_name,
                        "marker": marker_name,
                        "processor": processor_name,
                        "_dropped": is_dropped,
                    }
                else:
                    row = {
                        "df": out_name,
                        "df_raw": raw_name,
                        "marker": marker_name,
                        "processor": processor_name,
                        "_dropped": is_dropped,
                    }
                rows.append(row)

        if not rows:
            return pd.DataFrame() if return_df else ""

        df = pd.DataFrame(rows)

        if return_df:
            df = df.drop(columns=["_dropped"])
            out = df
            return out

        dropped_mask = df["_dropped"].values
        df = df.drop(columns=["_dropped"])

        lines = df.to_string(index=False).split("\n")
        header_line = lines[0]
        data_lines = lines[1:]

        grey = "\033[90m"
        reset = "\033[0m"

        formatted_lines = [header_line]
        for i, line in enumerate(data_lines):
            if i < len(dropped_mask) and dropped_mask[i]:
                formatted_lines.append(f"{grey}{line}{reset}")
            else:
                formatted_lines.append(line)

        out = "\n".join(formatted_lines)
        return out

    def feature_groups(self) -> dict:
        """Get feature index mapping for input columns.

        :return: Dict mapping raw column names to tuples of (index, output_name) pairs.

        Example
        -------
        >>> processor.feature_groups()
        {'Age': ((0, 'Age_geq_25'), (1, 'Age_geq_55')), ...}
        """
        x_index_lookup = {name: i for i, name in enumerate(self._outputs.X)}
        result = {}

        for raw_name in self._inputs.X:
            out_names = self._slices.get(raw_name, [])
            pairs = tuple((x_index_lookup[out_name], out_name) for out_name in out_names if out_name in x_index_lookup)
            if pairs:
                result[raw_name] = pairs
        return result

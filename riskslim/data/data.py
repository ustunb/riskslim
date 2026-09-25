"""
This file contains:
- Helper classes to represent and manipulate datasets for binary classification
- Helper functions to represent and manipulate categorical group attributes
"""
# pyright: reportArgumentType=false, reportCallIssue=false, reportAssignmentType=false, reportReturnType=false, reportOperatorIssue=false

import numpy as np
import pandas as pd
from functools import cached_property
from pathlib import Path
import warnings
from copy import deepcopy
from itertools import combinations
from dataclasses import dataclass, field
from types import SimpleNamespace
from imblearn.over_sampling import RandomOverSampler
from .cv import CVFolds
from .groups import GroupInfo, GroupAttributeEncoder, EncodingType
from .names import Names
from .processor import Processor
from riskslim.data.array_utils import is_boolean_array, DillSerializableMixin

ENCODING_FALSE_VALUES = {'sym': -1, 'bool': 0}
MISSING_VALUE = '?'


class BinaryClassificationDataset(DillSerializableMixin):
    """Class to represent/manipulate a dataset for a binary classification task.

    ----
    Internal Representation:
    - _df_raw: Raw DataFrame before binarization (always present)
    - _df_out: Intermediate DataFrame after transformation (always present)
    - _df: Final training DataFrame, subset of _df_out filtered by include mask
    - _transformer: Processor for rebinarization (always present)
        Note: Also accessible via `processor` property (alias for `transformer`).

    Column Flow:
        df_raw -> transformer.apply() -> _df_out -> include mask -> _df

    Invariants:
    - _n >= 1
    - _d >= 1
    - _df: DataFrame with _n rows, columns [y_name, G_names..., X_names...]
    - _df_out: DataFrame with _n rows, columns are superset of _df columns
    - X (derived from _df): 2D float64 array, shape (_n, _d), all values finite
    - y (derived from _df): 1D array, length _n, values in _classes
    - _classes: tuple of exactly 2 ints, sorted ascending (immutable after init)
    - len(_names.X) == _d
    - X, G, y names are disjoint (no overlapping names)
    - _cv.n == _n
    - _df_raw: DataFrame with _n rows (always present)
    - _df_out: DataFrame with _n rows (always present)
    - _transformer: Processor (always present)
    - _splits: None, or SimpleNamespace where:
        - sum of sample.n == _n
        - sample masks are disjoint and cover all indices
    """

    SAMPLE_TYPES = ('training', 'validation', 'test')
    DEFAULT_CLASSES = (-1, 1)

    def __init__(self, X = None, y = None, *, df_raw = None,
                 processor = None, recode_labels = None,
                 group_encoding = None, classes = None, **kwargs):
        """
        Initialize a binary classification dataset.

        Supports two initialization patterns:

        Pattern 1: df_raw + processor (recommended)
            >>> BinaryClassificationDataset(df_raw=df_raw, processor=processor)

        Pattern 2: X, y arrays (backwards compat, infer trivial processor)
            >>> BinaryClassificationDataset(X=X, y=y, X_names=names, group_df=group_df)

        :param X: Feature matrix (n_samples, n_features). Pattern 2 only.
        :param y: Target vector. Pattern 2 only.
        :param df_raw: Raw DataFrame before binarization. Pattern 1.
        :param processor: Processor spec. Pattern 1 only.
        :param recode_labels: Target label values as tuple (neg_class, pos_class).
            If None (default), preserve original values. If provided (e.g., (-1, 1) or (0, 1)),
            recode labels so that the smaller unique value maps to neg_class
            and the larger unique value maps to pos_class. Pattern 2 only.
        :param group_encoding: Group encoding type (default INTERSECTIONAL).
        :param classes: Class labels tuple (neg_class, pos_class). If None, inferred from y.
        :param X_names: Feature names. Pattern 2 only.
        :param y_name: Target name. Pattern 2 only.
        :param group_df: Group attribute data. Pattern 2 only.
        :param cv: CVFolds object. If not provided, creates one with default params.
        :param n_folds: Tuple of outer fold counts (default (1, 3, 4, 5)).
        :param n_folds_inner: Tuple of inner fold counts (default ()).
        :param n_reps: Number of replicates per fold count (default 3).
        :param stratify_by: Array to stratify folds by, or None.
        :param seed: Random seed for CV fold generation (default 2025).
        """
        if df_raw is not None and processor is not None:
            self._init_from_df(df_raw, processor,
                                               group_encoding=group_encoding,
                                               classes=classes, **kwargs)
        elif X is not None and y is not None:
            self.init_from_array(X, y, df_raw = df_raw, processor = processor,
                                recode_labels = recode_labels, **kwargs)
        else:
            raise ValueError(
                    "Invalid initialization pattern. Use one of:\n"
                    "  Pattern 1: df_raw + processor\n"
                    "  Pattern 2: X + y"
                    )
        assert self.__check_rep__()

    def init_from_array(self, X, y, *, df_raw = None, processor = None,
                       recode_labels = None, **kwargs):
        """Pattern 4: Initialize from X, y arrays (backwards compatible).
        :param X: Feature matrix (n_samples, n_features).
        :param y: Target vector.
        :param df_raw: Optional raw DataFrame for rebinarization.
        :param processor: Optional processor for rebinarization.
        :param recode_labels: Target label values as tuple (neg_class, pos_class).
            If None, preserve original values. If provided, recode labels.
        :param kwargs: Additional parameters (X_names, y_name, group_df, cv params).
        """
        # Convert and validate X
        X = np.atleast_2d(np.array(X, np.float64))
        assert X.ndim == 2, f"X must be 2D, got {X.ndim}D"
        assert np.isfinite(X).all(), "X must have finite values"

        # Convert and validate y
        y = np.array(y)
        assert y.ndim == 1, f"y must be 1D, got {y.ndim}D"

        # Apply recode_labels if specified
        if recode_labels is not None:
            neg_class, pos_class = recode_labels
            unique_vals = sorted(np.unique(y))
            assert len(unique_vals) <= 2, f"y must have at most 2 unique values for recoding, got {len(unique_vals)}"
            if len(unique_vals) == 1:
                val = unique_vals[0]
                new_class = neg_class if val in (0, -1, False) else pos_class
                y = np.full(len(y), new_class, dtype = np.int_)
            else:
                mapping = {unique_vals[0]: neg_class, unique_vals[1]: pos_class}
                y = np.array([mapping[v] for v in y], dtype = np.int_)

        # Validate dimensions
        n, d = len(y), X.shape[1]
        assert n >= 1, f"n must be >= 1, got {n}"
        assert d >= 1, f"d must be >= 1, got {d}"
        assert X.shape[0] == n, f"X rows ({X.shape[0]}) ≠ y length ({n})"

        # Get group_df and names from kwargs
        group_df = kwargs.pop('group_df', pd.DataFrame(index = range(n)))
        assert isinstance(group_df, pd.DataFrame), f"group_df must be a pd.DataFrame, got {type(group_df)}"
        assert len(group_df) == n, f"group_df rows ({len(group_df)}) ≠ n ({n})"

        names = Names(
                X = kwargs.pop('X_names', Names.generate_feature_names(d)),
                y = kwargs.pop('y_name', Names.DEFAULT_Y),
                G = list(group_df.columns),
                )

        if df_raw is None:
            df_raw = pd.DataFrame(X, columns = names.X)
            df_raw[names.y] = y
            for g in names.G:
                df_raw[g] = group_df[g].values

        if processor is None:
            processor = Processor.from_names(names)

        self._init_from_df(df_raw, processor, **kwargs)

    def _init_from_df(self, df_raw, processor, *,
                                       group_encoding=None, classes=None, **kwargs):
        """Pattern 1: Initialize from raw DataFrame + processor.

        :param df_raw: Raw DataFrame before binarization.
        :param processor: Processor spec defining column roles and binarizers.
        :param group_encoding: Group encoding type (default INTERSECTIONAL).
        :param classes: Class labels tuple. If None, inferred from y.
        :param kwargs: Additional parameters (cv params).
        """
        df_raw = pd.DataFrame(df_raw)
        self._n = len(df_raw)
        assert self._n >= 1, f"n must be >= 1, got {self._n}"

        # Apply processor
        df_out = processor.apply(df_raw)
        self._names = processor.get_output_names()
        assert self.d >= 1, f"d must be >= 1, got {self.d}"
        y = df_out[self._names.y].values
        self._classes = self._infer_classes(y)

        # Group encoding type
        self._group_encoding = group_encoding if group_encoding is not None else EncodingType.INTERSECTIONAL
        self._splits = None

        # Cross-validation
        cvfolds = kwargs.get('cv')
        if isinstance(cvfolds, CVFolds):
            self.cv = cvfolds
        else:
            self.cv = CVFolds(
                    n = self._n,
                    n_folds = kwargs.get('n_folds', (1, 3, 4, 5)),
                    n_folds_inner = kwargs.get('n_folds_inner', ()),
                    n_reps = kwargs.get('n_reps', 3),
                    stratify_by = kwargs.get('stratify_by'),
                    seed = kwargs.get('seed', 2025),
                    )

        # Store raw data and processor
        self._df_raw = df_raw
        self._processor = processor
        processor._dataset = self
        self._df_out = df_out
        self._rebuild_from_df_out(reset=False)

        # Update classes if different from inferred
        if classes is not None:
            self.update_classes(classes)

    def _infer_classes(self, y):
        """Infer class labels from y values.
        :param y: Target vector.
        :returns: Tuple of (negative_class, positive_class).
        """
        inferred = tuple(sorted(np.unique(y).tolist()))
        assert len(inferred) in (1, 2), \
            f"y must have ≤ 2 distinct classes, got {len(inferred)}: {inferred}"
        if len(inferred) == 2:
            return inferred
        # Single class: use default classes if the value is in defaults
        assert inferred[0] in self.DEFAULT_CLASSES, \
            f"y has single class {inferred[0]} not in DEFAULT_CLASSES {self.DEFAULT_CLASSES}"
        return self.DEFAULT_CLASSES

    def drop(self, to_drop):
        """Drop one or more features from training data.

        For both X features and G (group attributes): Sets include=False in
        transformer. Column remains in _df_out but is excluded from _df.
        This is reversible via include().

        :param to_drop: Feature name(s) to drop. Can be string or list of strings.
        :raises KeyError: If feature not in processor.
        :raises ValueError: If attempting to drop the label column.
        """
        changed, g_affected = self._processor.set_include(to_drop, False)
        if changed:
            if g_affected:
                self._invalidate_group_encoder_cache()
            self._rebuild_from_df_out()

    def include(self, to_include):
        """Re-include dropped X or G features in training data.

        :param to_include: Feature name(s) to include. Can be string or list of strings.
        :raises KeyError: If feature not in processor.
        """
        changed, g_affected = self._processor.set_include(to_include, True)
        if changed:
            if g_affected:
                self._invalidate_group_encoder_cache()
            self._rebuild_from_df_out()

    def reset(self, flag=True):
        """Reset to state before split."""
        if flag:
            self._splits = None
            assert self.__check_rep__()

    #### built-ins ####
    def __eq__(self, other):
        out = ( isinstance(other, BinaryClassificationDataset) and
                self._df.equals(other._df) and
                self._df_raw.equals(other._df_raw) and
                self._processor == other._processor and
                self._group_encoding == other._group_encoding and
                self._classes == other._classes and
                self.cv == other.cv
        )
        return out

    def __len__(self):
        return len(self._df)

    def __repr__(self):
        return f'BinaryClassificationDataset<n={self.n}, d={self.d}>'

    def __copy__(self):
        return BinaryClassificationDataset(
                df_raw = self._df_raw.copy(),
                processor = deepcopy(self._processor),
                cv = self.cv,
                group_encoding = self._group_encoding,
                classes = self._classes,
                )

    def __check_rep__(self):
        """Verify representation invariants."""
        names = self._names

        # Core dimensions
        assert self._n >= 1, f"n must be >= 1, got {self._n}"
        assert self.d >= 1, f"d must be >= 1, got {self.d}"

        # _df: DataFrame with _n rows, expected columns
        assert len(self._df) == self._n, \
            f"_df rows ({len(self._df)}) must match n ({self._n})"
        expected_cols = [names.y] + names.G + names.X
        assert list(self._df.columns) == expected_cols, \
            f"_df columns {list(self._df.columns)} must match {expected_cols}"

        # X (derived from _df): 2D float64, shape (_n, d), all finite
        X = self.X
        assert X.ndim == 2, f"X must be 2D, got {X.ndim}D"
        assert X.shape == (self._n, self.d), f"X shape {X.shape} != ({self._n}, {self.d})"
        assert X.dtype == np.float64, f"X dtype must be float64, got {X.dtype}"
        assert np.isfinite(X).all(), "X must have finite values"

        # y (derived from _df): 1D, length _n, values in _classes
        y = self.y
        assert y.ndim == 1, f"y must be 1D, got {y.ndim}D"
        assert len(y) == self._n, f"y length ({len(y)}) must match n ({self._n})"
        assert np.isin(y, self._classes).all(), f"y values must be in {self._classes}"

        # classes: tuple of exactly 2 ints, sorted ascending (immutable)
        assert isinstance(self._classes, tuple), \
            f"classes must be tuple, got {type(self._classes)}"
        assert len(self._classes) == 2, \
            f"classes must have 2 elements, got {len(self._classes)}"
        assert self._classes[0] < self._classes[1], \
            f"classes must be sorted ascending, got {self._classes}"

        # cv: n must match
        assert self._cv.n == self._n, f"cv.n ({self._cv.n}) must match n ({self._n})"

        # _df_raw: always present, DataFrame with _n rows
        assert len(self._df_raw) == self._n, \
            f"_df_raw rows ({len(self._df_raw)}) must match n ({self._n})"

        # _df_out: always present, DataFrame with _n rows
        assert len(self._df_out) == self._n, \
            f"_df_out rows ({len(self._df_out)}) must match n ({self._n})"

        p = self._processor

        # Include mask invariant
        # _df columns must be subset of _df_out columns (filtered by include mask)
        assert set(self._df.columns).issubset(self._df_out.columns), \
            f"_df columns must be subset of _df_out columns. Extra in _df: {set(self._df.columns) - set(self._df_out.columns)}"

        # Verify processor inputs are subset of _df_raw columns
        processor_inputs = set(p.inputs.X + [p.inputs.y] + p.inputs.G)
        processor_inputs.update(p._inputs_carryover)
        assert processor_inputs <= set(self._df_raw.columns), \
            f"Processor inputs must be subset of _df_raw columns. Missing: {processor_inputs - set(self._df_raw.columns)}"

        # splits: None or valid split structure
        if self._splits is not None:
            samples = list(vars(self._splits).values())
            n_total = sum(s.n for s in samples)
            assert n_total == self._n, f"sum of split sizes ({n_total}) must equal n ({self._n})"
            # Masks are disjoint and cover all indices
            combined_mask = np.zeros(self._n, dtype = bool)
            for s in samples:
                assert not np.any(combined_mask & s._mask), "split masks must be disjoint"
                combined_mask |= s._mask
            assert combined_mask.all(), "split masks must cover all indices"

        return True

    #### io functions ####
    @staticmethod
    def read_csv(data_file, recode_labels = (-1, 1), helper_file = None, **kwargs):
        """
        loads raw data from CSV
        :param data_file: Path to the data_file
        :param recode_labels: Target label values as tuple (neg_class, pos_class).
            If None, preserve original values. If provided (e.g., (-1, 1) or (0, 1)),
            recode labels so that the smaller unique value maps to neg_class
            and the larger unique value maps to pos_class. Default is (-1, 1).
        :param helper_file: Path to the helper_file or None. If None, NAME_data.csv implies
            NAME_helper.csv; other data file names imply no helper file.
        :return: BinaryClassificationDataset
        """
        data_path = Path(data_file)
        if helper_file is None and data_path.name.endswith('_data.csv'):
            helper_file = data_path.with_name(data_path.name[:-len('_data.csv')] + '_helper.csv')
        files = {
            'data': data_path,
            'helper': Path(helper_file).with_suffix('.csv') if helper_file is not None else None,
            }
        assert files[
            'data'].is_file(), f"could not find dataset file: {files['data']}"

        # read helper file
        if files['helper'] is not None and files['helper'].is_file():
            hf = pd.read_csv(files['helper'], sep = ',')
            hf['is_variable'] = ~(hf['is_outcome'].astype(bool) | hf[
                'is_group_attribute'].astype(bool))
            assert hf[['is_outcome', 'is_group_attribute', 'is_variable']].isin([0, 1]).to_numpy().all()
        else:
            warnings.warn(
                    f"did not find helper file: {files['helper']} inferring data types from disk")
            # if no helper file is found, we assume:
            # df has d+1 columns:
            # - column 0 is the outcome
            # - column 1...k are group attributes if they have non-numeric values (must be contiguous)
            # - column k+1,...,d are features
            df = pd.read_csv(files['data'], sep = ',')
            headers = df.columns.tolist()
            d = len(headers)
            assert d >= 2, "expected at least 2 columns (outcome and 1 feature)"
            is_group_attribute = np.zeros(d - 1, dtype = np.int8)
            is_variable = np.zeros(d - 1, dtype = np.int8)
            for j, name in enumerate(headers[1:]):
                dt = df[name].dtype
                if dt == object or getattr(dt, 'name', '') in ('category',
                                                               'string'):
                    is_group_attribute[j] = 1
                else:
                    is_variable[j] = 1
            hf = pd.DataFrame({
                'header': headers,
                'is_outcome': [1] + [0] * (d - 1),
                'is_group_attribute': [0] + is_group_attribute.tolist(),
                'is_variable': [0] + is_variable.tolist()
                })
        assert sum(hf['is_outcome']) == 1, 'helper file should specify 1 outcome'
        assert sum(hf['is_variable']) >= 1, 'helper file should specify at least 1 variable'
        if sum(hf['is_group_attribute']) < 1:
            warnings.warn('dataset does not contain group attributes')

        # parse names
        names = Names(
            y = hf.query('is_outcome == 1')['header'].iloc[0],
            G = hf.query('is_group_attribute == 1')['header'].tolist(),
            X = hf.query('is_variable == 1')['header'].tolist(),
            )
        # specify expected data types
        dtypes = ({names.y: np.int_}
                  | dict.fromkeys(names.G, "category")
                  | dict.fromkeys(names.X, np.float64))

        # load data with correct dtypes
        df = pd.read_csv(files['data'], sep = ',', dtype = dtypes)
        assert set(df.columns.to_list()) == set(hf['header'].to_list()), 'helper file should contain metadata for every column in the data file'
        return BinaryClassificationDataset(
                df_raw = df,
                processor = Processor.from_names(names, label_recode=recode_labels),
                **kwargs,
                )

    #### properties of the full dataset ####
    @property
    def df(self):
        """DataFrame containing y, G, X."""
        return self._df.copy()

    @property
    def names(self):
        return self._names

    @property
    def n(self):
        """Number of examples in full dataset."""
        return self._n

    @property
    def d(self):
        """Number of features in full dataset."""
        return len(self._names.X)

    @property
    def classes(self):
        """Class labels (neg_class, pos_class)."""
        return self._classes

    @property
    def X(self):
        """Feature matrix."""
        return self._df[self._names.X].values

    @property
    def G(self) -> pd.DataFrame:
        """DataFrame of group attributes (included columns only).

        Returns a DataFrame containing only the G columns that are currently
        included (not dropped). Use data.groups for analysis of all G columns.
        """
        names = self._names.G
        if names:
            out = self._df[names].copy()
        else:
            out = pd.DataFrame(index = range(self._n))
        return out

    @property
    def y(self):
        """Label vector."""
        return self._df[self._names.y].values

    #### group encoder ####
    @property
    def group_encoding(self):
        """Current encoding type for group attributes."""
        return self._group_encoding

    @group_encoding.setter
    def group_encoding(self, value):
        """Set encoding type for group attributes.
        :param value: EncodingType.ONEHOT or INTERSECTIONAL
        """
        updated = EncodingType(value)
        if updated != self._group_encoding:
            self._group_encoding = updated
            self._invalidate_group_encoder_cache()

    @property
    def Gx(self):
        """Encoded group attributes as numpy array.
        Returns the group attributes encoded using the stored encoding type.
        For empty G, returns array with shape (n, 0).

        :returns: Encoded group attributes, shape (n, n_encoded_cols).
        """
        return self.get_Gx(encoding_type = None)

    def get_Gx(self, encoding_type = None):
        """Get encoded group attributes with optional encoding type override.
        :param encoding_type: Encoding type to use. If None, uses stored
            encoding type (self._group_encoding). Does not modify stored state.
        :returns: Encoded group attributes, shape (n, n_encoded_cols).
        """
        if self.G.empty:
            out = np.empty((self._n, 0), dtype = np.int_)
        elif encoding_type is None:
            out = self.group_encoder.to_dummies(self.G)
        else:
            encoder = GroupAttributeEncoder(
                    df = self.G,
                    encoding_type = encoding_type
                    )
            out = encoder.to_dummies(self.G)
        return out

    def _invalidate_group_encoder_cache(self):
        """Invalidate cached group_encoder property.
        Called when G columns are dropped/included or encoding type changes.
        """
        self.__dict__.pop('group_encoder', None)

    @cached_property
    def groups(self):
        """GroupInfo object for analysis (includes ALL G columns).

        Returns a GroupInfo object containing all G columns from the
        transformer, including those that have been dropped/excluded.
        This allows fairness analysis on groups that were excluded
        from training.

        This property is cached since it uses ALL G columns from _df_out
        which never change after initialization.

        :returns: GroupInfo object with group metadata and utilities.
        """
        # Get ALL G input names from transformer (not just included ones)
        names = self._processor.inputs.G
        df = self._df_out[names] if names else pd.DataFrame(
                index = range(self._n))
        return GroupInfo(df = df)

    @cached_property
    def group_encoder(self):
        """GroupAttributeEncoder for included G columns.
        Creates encoder from current included G columns.
        Uses the stored group_encoding type. 
        This property is cached and invalidated when G columns are
        dropped/included or when group_encoding changes.

        :returns: GroupAttributeEncoder for included G columns.
        """
        return GroupAttributeEncoder(
                df = self.G,
                encoding_type = self._group_encoding
                )

    @property
    def boolean(self):
        """True if all features are boolean."""
        return is_boolean_array(self.X)

    #### processing API ####
    @property
    def df_raw(self):
        """Raw (unbinarized) data as DataFrame, or None if not available."""
        return self._df_raw

    @property
    def processor(self):
        """Processor for feature binarization."""
        return self._processor

    @processor.setter
    def processor(self, value: Processor):
        """Set processor and attach this dataset for reactive updates."""
        self._processor = value
        if value is not None:
            value._dataset = self
            self._df_out = value.apply(self._df_raw)
            self._rebuild_from_df_out()

    def _rebuild_from_df_out(self, reset=True):
        """Rebuild _df from _df_out. Called after processor changes."""
        names = self._processor.included
        df_out = self._df_out
        self._df = pd.concat([
            df_out[[names.y]],
            df_out[names.G],
            pd.DataFrame(df_out[names.X].values.astype(np.float64), columns=names.X),
        ], axis=1)
        self._names = names
        self.reset(reset)

    #### resampling ###
    def oversample(self, by_group = False, equalize_counts = True, **kwargs):
        """
        #TODO:
        """
        ros = RandomOverSampler(**kwargs)
        X, y, G = self.X, self.y, self.G
        if G.empty or (not by_group):
            X_res, y_res = ros.fit_resample(X, y)
            G_res = pd.DataFrame(index = range(len(y_res)))
        else:
            cols = self.names.G
            group_indices = self.groups.to_indices(columns = cols)
            if equalize_counts:
                m = len(cols)
                _, profile_idx = np.unique(np.column_stack((group_indices, y)), axis = 0, return_inverse = True)
                D, _ = ros.fit_resample(np.column_stack((G, X, y)), profile_idx)
                X_res, y_res = D[:, m:m + self.d], D[:, -1]
                G_res = pd.DataFrame(D[:, :m], columns = cols)
            else:
                Xr, yr, Gr = [], [], []
                for k, g in enumerate(self.groups.get_combinations(columns = cols)):
                    idx = group_indices == k
                    Xg, yg = ros.fit_resample(X[idx], y[idx])
                    Xr.append(Xg)
                    yr.append(yg)
                    Gr.append(np.tile(g, (len(yg), 1)))
                X_res, y_res = np.vstack(Xr), np.concatenate(yr)
                G_res = pd.DataFrame(np.vstack(Gr), columns = cols)
        return BinaryClassificationDataset(X = X_res, y = y_res, group_df = G_res,
                                           X_names = self.names.X, y_name = self.names.y)

    def update_classes(self, values):
        """Remap y values to new class labels.
        :param values: New class labels as (negative_class, positive_class).
        """
        yn = self._names.y
        assert isinstance(values, (np.ndarray, list, tuple)) and len(values) == 2
        values = tuple(int(v) for v in sorted(values))
        if self._n > 0 and values != self._classes:
            y = self._df[yn].values
            neg_idx = y == self._classes[0]
            pos_idx = y == self._classes[1]
            assert (neg_idx | pos_idx).all(), "y contains values not in current classes"
            new_y = np.where(neg_idx, values[0], values[1])
            self._df[yn] = new_y
            self._df_out[yn] = new_y
            self._classes = values
            self.reset()
            assert self.__check_rep__()

    #### cross validation ####
    def split(self, fold_id = None, *, n_folds = None, rep_id = 1, **holdout):
        """Split dataset into training and holdout sets.

        :param fold_id: CV fold identifier (e.g., "K05N01"). If None, uses n_folds/rep_id.
        :param n_folds: Number of folds. Required if fold_id is None.
        :param rep_id: Replicate ID (default 1).
        :param **holdout: fold_num_<sample_type> -> fold number (e.g., fold_num_test=1).
        :raises ValueError: If holdout names are not fold_num_<SAMPLE_TYPE>.
        """
        # Parse holdout: fold_num_<name> -> <name>
        parsed = {}
        for key, val in holdout.items():
            if not key.startswith('fold_num_'):
                raise ValueError(
                        f"Invalid holdout key '{key}'. Must start with 'fold_num_'")
            name = key[len('fold_num_'):]
            if name not in self.SAMPLE_TYPES:
                raise ValueError(
                        f"Invalid sample type '{name}'. Must be in {self.SAMPLE_TYPES}")
            parsed[name] = val

        if fold_id is None:
            fold_id = CVFolds.get_fold_id(n_folds, rep_id)
        masks = self.cv.get_split_masks(fold_id, **parsed)
        self._splits = SimpleNamespace(**{
            name: BinaryClassificationSample(self, mask)
            for name, mask in masks.items()
            })
        assert self.__check_rep__()

    @property
    def splits(self) -> SimpleNamespace | None:
        """Splits namespace (training, validation, test). None before split() called."""
        return self._splits

    @property
    def cv(self):
        """CVFolds object for cross-validation."""
        return self._cv

    @cv.setter
    def cv(self, cv: CVFolds):
        assert cv.n == self._n, f"cv.n ({cv.n}) must match n ({self._n})"
        self._cv = cv


@dataclass(frozen = True, eq = False)
class BinaryClassificationSample:
    """An immutable view into a parent dataset.
    ----
    Representation Invariants:
        _dataset: BinaryClassificationDataset, parent dataset
        _mask: 1D bool array, length == _dataset._n
    All other properties are views into the parent dataset.
    """
    _dataset: 'BinaryClassificationDataset' = field(repr = False)
    _mask: np.ndarray = field(repr = False)

    def __post_init__(self):
        assert self.__check_rep__()

    def __len__(self):
        return self.n

    def __eq__(self, other):
        return (isinstance(other, BinaryClassificationSample) and
                np.array_equal(self.X, other.X) and
                np.array_equal(self.y, other.y) and
                self.G.columns.tolist() == other.G.columns.tolist() and
                np.array_equal(self.G.values, other.G.values))

    def __check_rep__(self):
        """Returns True if object satisfies representation invariants."""
        assert len(self._mask) == self._dataset._n, "mask length must match dataset.n"
        assert np.sum(self._mask) == self.X.shape[0], "mask sum must match n"
        assert np.isfinite(self.X).all(), "X must have finite values"
        assert np.isin(self.y, self.classes).all(), f"y values must be in {self.classes}"
        return True

    @property
    def df(self):
        return self._dataset._df.loc[self._mask].copy()

    @property
    def y(self):
        return self._dataset.y[self._mask]

    @property
    def X(self):
        return self._dataset.X[self._mask]

    @property
    def G(self):
        """DataFrame of group attributes (view into parent)."""
        return self._dataset.G.loc[self._mask]

    @property
    def Gx(self):
        """Matrix of encoded group attributes array. 
        Returns the same encoding as parent.Gx for this sample.
        Uses the parent's group_encoder to ensure consistent encoding between train/test splits.
        :returns: Array of shape (n, n_encoded_columns).
        """
        if self.G.empty:
            out = np.empty((self.n, 0), dtype = np.int_)
        else:
            out = self._dataset.group_encoder.to_dummies(self.G)
        return out

    @property
    def classes(self):
        return self._dataset._classes

    @property
    def n(self):
        return self.X.shape[0]

    @property
    def d(self):
        return self.X.shape[1]

    @property
    def rank(self):
        """Matrix rank of X."""
        return np.linalg.matrix_rank(self.X)

    @property
    def duplicates(self):
        """Returns indices of duplicate features."""
        out = [S for S in combinations(range(self.d), 2) if
               np.array_equal(self.X[:, S[0]], self.X[:, S[1]])]
        return out

    def collinear_subsets(self, m = 2):
        """Returns indices of features that sum to 1.

        :param m: Size of subsets.
        :returns: List of m-tuples of collinear feature indices.
        """
        assert 2 <= m <= self.d
        out = [S for S in combinations(range(self.d), m) if
               np.all(np.sum(self.X[:, S], axis = 1) == 1)]
        return out

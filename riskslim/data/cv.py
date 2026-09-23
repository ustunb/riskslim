"""
Helper functions to generate cross-validation indices for binary classification task
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, InitVar
from itertools import product
from typing import Literal, overload
import numpy as np
from sklearn.model_selection import KFold as KFoldGenerator
from sklearn.model_selection import StratifiedKFold as StratifiedKFoldGenerator


@dataclass
class FoldVector:
    """
    Fold assignments for standard or stratified K-fold CV.

    :param n_folds: Number of folds (>= 1).
    :param stratify_by: Array to stratify folds by. Uses StratifiedKFold if >= 2 unique
        values, otherwise KFold.
    :param seed: Random seed for reproducibility.

    ----
    Representation invariants:

    - ``index`` is a 1D numpy array of length ``n``
    - ``n >= 1``, ``n_folds <= n``
    - Fold values are consecutive integers from 1 to ``n_folds``
    - Fold counts are balanced: ``max(counts) - min(counts) <= 1``
    - Same ``(n_folds, stratify_by, seed)`` always produces identical ``index``
    """
    n_folds: int
    stratify_by: InitVar[np.ndarray]
    seed: int | None = None
    n: int = field(init=False)
    index: np.ndarray = field(init=False, repr=False)
    stratified: bool = field(init=False)
    imbalance: float = field(init=False)  # max proportion deviation across folds
    counts: np.ndarray = field(init=False, repr=False)  # cached fold counts

    def __post_init__(self, stratify_by):
        assert self.n_folds >= 1
        self.n = len(stratify_by)
        self.stratified = len(np.unique(stratify_by)) >= 2

        if self.n_folds == 1:
            # trivial: all samples in fold 1
            self.index = np.ones(self.n, dtype=np.int64)

        elif self.stratified:
            # stratified k-fold
            self.index = np.zeros(self.n, dtype=np.int64)
            gen = StratifiedKFoldGenerator(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            for k, (_, idx) in enumerate(gen.split(stratify_by, stratify_by)):
                self.index[idx] = k + 1

        else:
            # standard k-fold
            self.index = np.zeros(self.n, dtype=np.int64)
            gen = KFoldGenerator(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            for k, (_, idx) in enumerate(gen.split(np.empty(self.n))):
                self.index[idx] = k + 1

        # compute counts and imbalance
        self.counts = np.bincount(self.index, minlength=self.n_folds + 1)[1:]
        self.imbalance = float(self.counts.max() - self.counts.min()) / self.n

    @overload
    def tally(self, strata: None = ..., return_labels: bool = ...) -> np.ndarray: ...
    @overload
    def tally(self, strata: np.ndarray, return_labels: Literal[False] = ...) -> np.ndarray: ...
    @overload
    def tally(self, strata: np.ndarray, return_labels: Literal[True]) -> tuple[np.ndarray, np.ndarray]: ...
    def tally(self, strata=None, return_labels=False):
        """
        Count samples per fold, optionally stratified.

        :param strata: Stratification labels. If None, returns cached fold counts.
        :param return_labels: If True and strata provided, return (counts, labels).
        :returns: 1D array (n_folds,) if strata is None,
            2D array (n_folds, n_strata) if strata provided,
            or tuple (counts, labels) if return_labels=True.
        """
        if strata is None:
            out = self.counts
        else:
            labels = np.unique(strata)
            counts = np.zeros((self.n_folds, len(labels)), dtype=np.int64)
            for k in range(self.n_folds):
                mask = self.index == k + 1
                for j, lbl in enumerate(labels):
                    counts[k, j] = np.sum(strata[mask] == lbl)
            out = (counts, labels) if return_labels else counts
        return out

    def __repr__(self):
        fold_sizes = self.tally().tolist()
        return f"FoldVector(n_folds={self.n_folds}, n={self.n}, fold_sizes={fold_sizes}, stratified={self.stratified})"


class OuterFoldID:
    """Converter for outer fold IDs (e.g., "K05N01" = 5-fold CV, replicate 1)."""
    _pattern = re.compile(r"^K(?P<n_folds>\d{2})N(?P<rep_id>\d{2})$")

    @staticmethod
    def get(n_folds: int, rep_id: int = 1) -> str:
        return f"K{n_folds:02d}N{rep_id:02d}"

    @staticmethod
    def parse(s: str) -> tuple[int, int]:
        match = OuterFoldID._pattern.match(s.strip().upper())
        if match is None:
            raise ValueError(f"Invalid fold ID: {s}")
        return int(match.group('n_folds')), int(match.group('rep_id'))


class InnerFoldID:
    """Converter for inner fold IDs (e.g., "F02K03" = fold 2 held out, 3-fold inner CV)."""
    _pattern = re.compile(r"^F(?P<fold_idx>\d{2})K(?P<n_folds>\d{2})$")

    @staticmethod
    def get(fold_idx: int, n_folds: int) -> str:
        return f"F{fold_idx:02d}K{n_folds:02d}"

    @staticmethod
    def parse(s: str) -> tuple[int, int]:
        match = InnerFoldID._pattern.match(s.strip().upper())
        if match is None:
            raise ValueError(f"Invalid inner fold ID: {s}")
        return int(match.group('fold_idx')), int(match.group('n_folds'))


class CVFolds:
    """Registry to generate and store fold assignment vectors for cross-validation.

    :param n: Number of samples.
    :param n_folds: Tuple of fold counts for outer CV.
    :param n_folds_inner: Tuple of fold counts for inner/nested CV.
    :param n_reps: Number of replicates per fold count.
    :param seed: Random seed for reproducibility.
    :param stratify_by: Array to stratify folds by, or None for unstratified CV.

    ----
    Invariants:
        - n >= 1
        - n_folds: unique positive integers in 1..n (empty: no folds are built)
        - n_folds_inner: unique positive integers >= 1
        - n_reps >= 1
        - seed > 0
        - _strata is 1D array of length n
        - _registry keys are outer fold IDs (K##N##)
        - _inner[outer_id] keys are inner fold IDs (F##K##)
        - replicates with same fold_count (> 1) have different seeds and fold assignments
    """

    def __init__(
        self,
        n: int,
        n_folds: Sequence[int] = (1, 2, 3, 5, 10),
        n_folds_inner: Sequence[int] = (2, 3, 5),
        n_reps: int = 3,
        seed: int = 2338,
        stratify_by: np.ndarray | None = None,
    ):
        # Validate and store parameters
        self.n = int(n)
        assert self.n >= 1, f"n must be >= 1, got {self.n}"

        self.n_folds = tuple(n_folds)
        assert len(self.n_folds) == len(set(self.n_folds)), "n_folds must have unique values"
        for k in self.n_folds:
            assert 1 <= k <= self.n, f"n_folds values must be in 1..{self.n}, got {k}"

        self.n_folds_inner = tuple(n_folds_inner)
        assert len(self.n_folds_inner) == len(set(self.n_folds_inner)), "n_folds_inner must have unique values"
        for k in self.n_folds_inner:
            assert k >= 1, f"n_folds_inner values must be >= 1, got {k}"

        self.n_reps = int(n_reps)
        assert self.n_reps >= 1, f"n_reps must be >= 1, got {self.n_reps}"

        self.seed = int(seed)
        assert self.seed > 0, f"seed must be > 0, got {self.seed}"

        # Initialize strata
        if stratify_by is None:
            self._strata = np.ones(self.n, dtype=np.int64)
        else:
            self._validate_strata(stratify_by, self.n)
            self._strata = stratify_by
        self.stratified = len(np.unique(self._strata)) >= 2

        # Generate fold registry
        self._rng = np.random.default_rng(self.seed)
        self._registry, self._inner = self._generate(self.n_folds, self.n_folds_inner, self.n_reps)

        assert self.__check_rep__()

    def __check_rep__(self):
        """Verify representation invariants."""
        # Check replicates with same fold_count > 1 have different seeds and values
        for k in self.n_folds:
            if k > 1 and self.n_reps > 1:
                seeds = []
                indices = []
                for r in range(1, self.n_reps + 1):
                    fold_id = OuterFoldID.get(k, r)
                    fv = self._registry[fold_id]
                    seeds.append(fv.seed)
                    indices.append(tuple(fv.index))
                assert len(set(seeds)) == len(seeds), f"replicates for k={k} must have different seeds"
                assert len(set(indices)) == len(indices), f"replicates for k={k} must have different fold assignments"
        return True

    @staticmethod
    def _validate_strata(strata: np.ndarray, n: int) -> None:
        """
        Validate stratification array.

        :param strata: Array to stratify by.
        :param n: Expected length.
        :raises AssertionError: If strata is invalid.
        """
        assert isinstance(strata, np.ndarray), 'strata must be ndarray'
        assert strata.ndim == 1, 'strata must be 1D'
        assert len(strata) == n, f"len(strata) ({len(strata)}) != n ({n})"
        if np.issubdtype(strata.dtype, np.number):
            assert np.isfinite(strata).all(), 'strata must be finite'
        assert len(np.unique(strata)) >= 2, 'strata must have >= 2 classes'

    def _generate(self, n_folds, n_folds_inner, n_reps):
        """
        Generate database of fold assignments for all CV configurations.
        :param n_folds: List of outer fold counts.
        :param n_folds_inner: List of inner fold counts.
        :param n_reps: Number of replicates for each fold.
        :returns: Tuple of (outer_registry, inner_registry).
        """
        outer = {}
        inner = {}
        for k, r in product(n_folds, range(1, n_reps + 1)):
            fold_id = OuterFoldID.get(k, r)
            fold_seed = int(self._rng.integers(0, 2**31))
            fold = FoldVector(k, self._strata, seed = fold_seed)
            outer[fold_id] = fold
            inner[fold_id] = {}
            if not n_folds_inner:
                continue
            for f in range(1, k + 1):
                train_strata = self._strata[fold.index != f]
                for l in n_folds_inner:
                    inner_id = InnerFoldID.get(f, l)
                    inner_seed = int(self._rng.integers(0, 2**31))
                    inner_fold = FoldVector(l, train_strata, seed=inner_seed)
                    inner[fold_id][inner_id] = inner_fold
        return outer, inner

    def __repr__(self):
        return f"CVFolds(n={self.n}, n_entries={len(self._registry)}, stratified={self.stratified})"

    def __contains__(self, fold_id: str) -> bool:
        return fold_id in self._registry

    def __getitem__(self, fold_id: str) -> np.ndarray:
        return self._registry[fold_id].index

    @overload
    def get(self, n_folds: int, rep_id: int = ..., return_object: Literal[True] = ...) -> FoldVector: ...
    @overload
    def get(self, n_folds: int, rep_id: int, return_object: Literal[False]) -> np.ndarray: ...
    @overload
    def get(self, n_folds: int, *, return_object: Literal[False]) -> np.ndarray: ...
    def get(self, n_folds: int, rep_id: int = 1, return_object: bool = True) -> np.ndarray | FoldVector:
        """Get fold assignments by n_folds and rep_id.
        :param n_folds: number of folds
        :param rep_id: replicate id, default:1
        :param return_object: If True, return FoldVector object; if False, return array of fold assignments.
        :returns: 1D array of fold assignments, or FoldVector if return_object=True.
        :raises KeyError: If fold configuration not found.
        """
        fold_id = OuterFoldID.get(n_folds, rep_id)
        if fold_id not in self._registry:
            raise KeyError(f"n_folds={n_folds}, rep_id={rep_id} not found")
        fv = self._registry[fold_id]
        out = fv.index if not return_object else fv
        return out

    def __eq__(self, other):
        if not isinstance(other, CVFolds):
            return NotImplemented
        outer_match = (
            self._registry.keys() == other._registry.keys()
            and all(np.array_equal(self._registry[k].index, other._registry[k].index) for k in self._registry)
        )
        inner_match = (
            self._inner.keys() == other._inner.keys()
            and all(
                self._inner[k].keys() == other._inner[k].keys()
                and all(np.array_equal(self._inner[k][j].index, other._inner[k][j].index) for j in self._inner[k])
                for k in self._inner
            )
        )
        out = outer_match and inner_match
        return out

    def get_inner(self, outer_id: str, fold_idx: int, n_folds: int) -> FoldVector:
        """Get inner fold assignments.

        :param outer_id: Outer fold ID string (e.g., "K05N01").
        :param fold_idx: Which fold of outer was held out.
        :param n_folds: Number of inner folds.
        :returns: FoldVector for the inner fold configuration.
        :raises KeyError: If outer_id or inner configuration not found.
        """
        inner_id = InnerFoldID.get(fold_idx, n_folds)
        out = self._inner[outer_id][inner_id]
        return out

    def check_distributions(
        self,
        strata=None,
        check_outer=True,
        check_inner=False,
        return_violations=False,
        max_imbalance=None,
    ):
        """
        Check fold distributions, optionally against stratification.

        :param strata: Stratification labels, or None to use self._strata.
        :param check_outer: Check outer fold vectors.
        :param check_inner: Check inner fold vectors.
        :param return_violations: If True, return only tallies exceeding max_imbalance.
        :param max_imbalance: Threshold for violations. If None, computed as 1/n.
            Use -1.0 to return all tallies.
        :returns: Dict of fold_id -> tally matrix.
        """
        if strata is None:
            strata = self._strata

        tallies = {}

        # check outer folds
        if check_outer:
            threshold = max_imbalance if max_imbalance is not None else 1.0 / self.n
            for fold_id, fv in self._registry.items():
                tally = fv.tally(strata)
                if return_violations:
                    props = tally / tally.sum(axis=0)
                    imbalance = (props.max(axis=0) - props.min(axis=0)).max()
                    if imbalance > threshold:
                        tallies[fold_id] = tally
                else:
                    tallies[fold_id] = tally

        # check inner folds
        if check_inner:
            for outer_id, inner_dict in self._inner.items():
                outer_fv = self._registry[outer_id]
                for inner_id, fv in inner_dict.items():
                    fold_idx = InnerFoldID.parse(inner_id)[0]
                    train_strata = strata[outer_fv.index != fold_idx]
                    tally = fv.tally(train_strata)
                    full_id = f"{outer_id}_{inner_id}"
                    if return_violations:
                        threshold = max_imbalance if max_imbalance is not None else 1.0 / fv.n
                        props = tally / tally.sum(axis=0)
                        imbalance = (props.max(axis=0) - props.min(axis=0)).max()
                        if imbalance > threshold:
                            tallies[full_id] = tally
                    else:
                        tallies[full_id] = tally

        return tallies

    @staticmethod
    def get_fold_id(n_folds: int, rep_id: int = 1) -> str:
        """
        Create an outer fold ID string.

        :param n_folds: Number of folds.
        :param rep_id: Replicate ID (default 1).
        :returns: Fold ID string (e.g., "K05N01").
        """
        return OuterFoldID.get(n_folds, rep_id)

    @classmethod
    def register(
        cls,
        *,
        n: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        seed: int = 1,
    ) -> "CVFolds":
        """Build a CVFolds whose K=3 outer fold encodes pre-baked indices.

        Encoding: fold 1 = test, fold 2 = validation, fold 3 = training.
        Reproduce the upstream split via::

            dataset.split('K03N01', fold_num_test=1, fold_num_validation=2)

        :param n: Number of samples (must equal len(train_idx) + len(val_idx) + len(test_idx)).
        :param train_idx: 1D int array of training indices.
        :param val_idx: 1D int array of validation indices.
        :param test_idx: 1D int array of test indices.
        :param seed: Random seed (kept for invariant compliance; default 1).
        :returns: CVFolds with a single outer fold ``K03N01`` containing the pre-baked split.
        """
        # Validate inputs
        for name, arr in (("train_idx", train_idx), ("val_idx", val_idx), ("test_idx", test_idx)):
            assert isinstance(arr, np.ndarray), f"{name} must be np.ndarray, got {type(arr)}"
            assert arr.ndim == 1, f"{name} must be 1D, got {arr.ndim}D"
            assert np.issubdtype(arr.dtype, np.integer), f"{name} must be int dtype, got {arr.dtype}"

        n = int(n)
        assert n >= 1, f"n must be >= 1, got {n}"

        # Per-split uniqueness: catch duplicate indices within a single split
        # before set conversion silently dedupes them in the coverage check.
        for name, arr in (("train_idx", train_idx), ("val_idx", val_idx), ("test_idx", test_idx)):
            unique = np.unique(arr)
            assert len(arr) == len(unique), (
                f"{name} must contain unique indices, got {len(arr) - len(unique)} "
                f"duplicate(s); offending values: "
                f"{sorted(v for v in unique if int((arr == v).sum()) > 1)}"
            )
            assert arr.size == 0 or (arr.min() >= 0 and arr.max() < n), (
                f"{name} must have all entries in [0, {n}); got "
                f"min={int(arr.min()) if arr.size else None}, "
                f"max={int(arr.max()) if arr.size else None}"
            )

        train_set = set(train_idx.tolist())
        val_set = set(val_idx.tolist())
        test_set = set(test_idx.tolist())

        # Disjointness
        assert not (train_set & val_set), f"train and val overlap on {train_set & val_set}"
        assert not (train_set & test_set), f"train and test overlap on {train_set & test_set}"
        assert not (val_set & test_set), f"val and test overlap on {val_set & test_set}"

        # Coverage: train must equal complement of (val ∪ test) within [0, n)
        full = set(range(n))
        holdout = val_set | test_set
        expected_train = full - holdout
        assert train_set == expected_train, (
            f"train_idx must be the complement of val ∪ test in [0, n). "
            f"Missing from train: {expected_train - train_set}; "
            f"unexpected in train: {train_set - expected_train}"
        )
        assert train_set | val_set | test_set == full, (
            f"train ∪ val ∪ test must cover [0, {n}); "
            f"missing: {full - (train_set | val_set | test_set)}"
        )

        # Build a synthetic stratify_by with K=3 fold ids so the standard
        # constructor flow accepts it; we will overwrite the registry entry.
        synthetic_strata = np.empty(n, dtype=np.int64)
        synthetic_strata[test_idx] = 1
        synthetic_strata[val_idx] = 2
        synthetic_strata[train_idx] = 3

        instance = cls(
            n=n,
            n_folds=(3,),
            n_folds_inner=(),
            n_reps=1,
            seed=seed,
            stratify_by=synthetic_strata,
        )

        # Build the pre-baked FoldVector by overwriting the auto-generated one.
        # Use FoldVector with n_folds=1 (trivial) then overwrite .index.
        fold_id = OuterFoldID.get(3, 1)
        index = np.empty(n, dtype=np.int64)
        index[test_idx] = 1
        index[val_idx] = 2
        index[train_idx] = 3

        fv = instance._registry[fold_id]
        fv.index = index
        fv.n = n
        fv.n_folds = 3
        fv.counts = np.bincount(index, minlength=4)[1:]
        fv.imbalance = float(fv.counts.max() - fv.counts.min()) / n
        fv.stratified = True

        instance._inner[fold_id] = {}

        assert instance.__check_rep__()
        return instance

    def get_split_masks(self, fold_id: str, **holdout) -> dict[str, np.ndarray]:
        """
        Get boolean masks for train/holdout splits.

        :param fold_id: Outer fold ID (e.g., "K05N01").
        :param **holdout: Sample type -> fold number (e.g., test=1, validation=2).
        :returns: Dict mapping sample types to boolean masks. Always includes 'training'.
        :raises KeyError: If fold_id not found.
        :raises ValueError: If fold numbers invalid or overlap.

        Example::

            masks = cv.get_split_masks("K05N01", test=1, validation=2)
            # -> {'training': array([...]), 'test': array([...]), 'validation': array([...])}
        """
        folds = self[fold_id]
        n_folds = int(folds.max())

        # Validate fold numbers
        holdout_folds = []
        for name, fold_num in holdout.items():
            if fold_num < 1 or fold_num > n_folds:
                raise ValueError(f"{name} fold must be 1..{n_folds}, got {fold_num}")
            if fold_num in holdout_folds:
                raise ValueError(f"fold {fold_num} assigned to multiple sample types")
            holdout_folds.append(fold_num)

        # Compute masks
        masks = {'training': np.isin(folds, holdout_folds, invert=True)}
        for name, fold_num in holdout.items():
            masks[name] = (folds == fold_num)

        return masks

"""RiskSLIM Classifier."""

import copy

import numpy as np
from scipy.special import expit

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import check_scoring
from sklearn.model_selection import PredefinedSplit, check_cv, cross_validate
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

from .optimizer import RiskSLIMOptimizer
from .coefficient_set import CoefficientSet
from .data import BinaryClassificationDataset, CVFolds
from .defaults import DEFAULT_LCPA_SETTINGS, INTERCEPT_NAME, OUTCOME_NAME
from .report import ModelReport
from .utils import print_model


# fitted state that belongs to one fit and is dropped when fit runs again
STATE_FROM_PREVIOUS_FIT = ("calibrated_estimator_", "cv_", "cv_results_", "cv_calibrated_estimators_")


class RiskSLIMClassifier(ClassifierMixin, BaseEstimator):
    """RiskSLIM classifier

    A scikit-learn estimator: it can be cloned, grid-searched, cross-validated, calibrated,
    and pickled. Settings passed as keyword arguments are reported by ``get_params`` and
    carried over by ``clone``.

    Attributes
    ----------
    classes_ : 1d array of shape (2,)
        Class labels. ``classes_[1]`` is the positive class.
    coef_ : 1d array
        Solved coefficients, excluding the intercept.
    intercept_ : float
        Solved intercept.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : 1d array of str
        Feature names seen during fit, when X is a DataFrame with string column names.
    max_size_ : int
        Maximum model size used in the fit.
    coef_set_ : riskslim.coefficient_set.CoefficientSet
        Coefficient constraints used in the fit (a copy; ``coef_set`` is never modified).
    optimizer_ : riskslim.optimizer.RiskSLIMOptimizer
        The fitted optimizer (CPLEX model, bounds, stats, solution pool). Not pickled.
    solution_info_ : dict
        Solver statistics at the end of the fit (objective value, optimality gap, run time, ...).
    calibrated_estimator_ : sklearn.calibration.CalibratedClassifierCV
        Calibrator trained on all data. Set by ``recalibrate``.
    cv_ : sklearn cross-validation splitter
        Set by ``fit_cv``.
    cv_results_ : dict
        Cross-validation results. Set by ``fit_cv``.
    cv_calibrated_estimators_ : list of sklearn.calibration.CalibratedClassifierCV
        Calibrators trained per fold. Set by ``recalibrate`` after ``fit_cv``.
    """
    def __init__(self, max_coef = 5, max_size = None, coef_set = None,
                 variable_names = None, outcome_name = None, c0_value = 1e-6,
                 verbose = True, solver = "cplex", **kwargs):
        """
        Parameters
        ----------
        max_coef : float or 1d array, optional, default: 5
            Maximum absolute coefficient.
        max_size : int, optional, default: None
            Maximum number of regularized coefficients.
            None defaults to the number of input variables.
        coef_set: riskslim.coefficient_set.CoefficientSet
            Contraints (bounds) on coefficients of input variables.
            If None, this is constructed from max_coef.
        variable_names : list of str, optional, default: None
            Names of each feature.
            None defaults to the DataFrame column names, or to generic variable names.
        outcome_name : str, optional, default: None
            Name of the output class.
        c0_value : 1d array or float, optional, default: 1e-6
            L0-penalty for all parameters when an integer or for each parameter
            separately when an array.
        verbose : bool, optional, default: True
            Prints out log information if True, supresses if False.
        solver : str, optional, default: "cplex"
            MIP solver. Only "cplex" is available.
        **kwargs
            Settings for warmstart (keys: 'init_*'), cplex (keys: 'cplex_*'), and lattice CPA.
            Defaults are defined in ``defaults.DEFAULT_LCPA_SETTINGS``.
        """
        self.max_coef = max_coef
        self.max_size = max_size
        self.coef_set = coef_set
        self.variable_names = variable_names
        self.outcome_name = outcome_name
        self.c0_value = c0_value
        self.verbose = verbose
        self.solver = solver
        self._settings = kwargs

    def get_params(self, deep=True):
        """Get parameters, including settings passed as keyword arguments."""
        params = super().get_params(deep=deep)
        params.update(self._settings)
        return params

    def set_params(self, **params):
        """Set parameters, including settings passed as keyword arguments."""
        settings = {k: v for k, v in params.items() if k not in self._get_param_names()}
        self._settings.update(settings)
        others = {k: v for k, v in params.items() if k not in settings}
        return super().set_params(**others)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def __getstate__(self):
        # the optimizer holds the CPLEX model and closures, neither of which pickle
        state = dict(super().__getstate__())
        state.pop("optimizer_", None)
        return state

    @property
    def _rho(self):
        """Intercept followed by the coefficients, the order print_model and reports expect."""
        return np.concatenate([[self.intercept_], self.coef_])

    @property
    def _variable_names(self):
        """INTERCEPT_NAME followed by the feature names, aligned with ``_rho``."""
        return [INTERCEPT_NAME] + list(self._data.names.X)

    def __repr__(self, N_CHAR_MAX=700):
        if hasattr(self, "coef_"):
            table = print_model(self._rho, self._variable_names, self._data.names.y,
                                return_only=True)
            return str(table)
        return super().__repr__(N_CHAR_MAX=N_CHAR_MAX)

    @property
    def fitted(self):
        """Whether the model has been fit."""
        return hasattr(self, "coef_")

    @property
    def optimizer(self):
        """The fitted optimizer, or None before fit (alias of ``optimizer_``)."""
        return getattr(self, "optimizer_", None)

    def fit(self, X, y, **kwargs):
        """Fit RiskSLIM classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Observations (rows) and features (columns), without an intercept column.
        y : array-like of shape (n_samples,)
            Class labels; exactly two classes.
        **kwargs
            Settings that override those given at initialization for this fit only.
        """
        settings = {**self._settings, **kwargs}
        unknown = sorted(set(settings) - set(DEFAULT_LCPA_SETTINGS))
        if unknown:
            raise ValueError(f"Unknown RiskSLIM settings: {unknown}")

        for name in STATE_FROM_PREVIOUS_FIT:
            self.__dict__.pop(name, None)

        X, y = validate_data(self, X, y, dtype=np.float64)
        check_classification_targets(y)
        y_type = type_of_target(y, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )
        self.classes_, y_index = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError(
                f"RiskSLIMClassifier needs y with exactly 2 classes; got {len(self.classes_)} class(es)."
            )
        y_signed = 2 * y_index - 1

        variable_names = self.variable_names
        if variable_names is None and hasattr(self, "feature_names_in_"):
            variable_names = list(self.feature_names_in_)
        outcome_name = OUTCOME_NAME if self.outcome_name is None else self.outcome_name
        names = {} if variable_names is None else {"X_names": list(variable_names)}
        self._data = BinaryClassificationDataset(X=X, y=y_signed, y_name=outcome_name, n_folds=(), **names)

        if self.coef_set is None:
            self.coef_set_ = CoefficientSet(self._variable_names, lb = -self.max_coef, ub = self.max_coef)
        else:
            self.coef_set_ = copy.deepcopy(self.coef_set)
        # default: every coefficient, the intercept counted (data.d does not count it)
        self.max_size_ = self._data.d + 1 if self.max_size is None else self.max_size

        self.optimizer_ = RiskSLIMOptimizer(
            data = self._data,
            coef_set = self.coef_set_,
            max_size = self.max_size_,
            c0_value = self.c0_value,
            verbose = self.verbose,
            solver = self.solver,
            **settings,
        )
        self.optimizer_.optimize()

        coefficients = self.optimizer_.coefficients
        self.coef_ = coefficients[1:]
        self.intercept_ = coefficients[0]
        self.solution_info_ = {
            k: v.tolist() if isinstance(v, (np.ndarray, np.generic)) else v
            for k, v in self.optimizer_.solution_info.items()
        }
        return self

    def report(self, X_test=None, y_test=None, model_type=None, *, data=None, folds=None):
        """HTML report of the fitted model: the model, a summary table, ROC and calibration.

        Parameters
        ----------
        X_test, y_test : array-like, optional
            A held-out sample, shown next to the training sample. Ignored, with a warning, when
            ``data`` already has a test split.
        model_type : {"risk_score", "checklist"}, optional
            How to show the model. Inferred from the coefficients when None: a checklist when
            every nonzero coefficient is +1 or -1.
        data : riskslim.data.BinaryClassificationDataset, optional
            The dataset the model was fit on: its names label the page, and when it has splits
            (``data.split(...)``) each split is a sample; otherwise the training sample is the
            data passed to ``fit``.
        folds : list of RiskSLIMClassifier, optional
            Fitted per-fold models for the CV sample, scoring the test rows of ``fit_cv``. None
            uses ``cv_results_["estimator"]`` after ``fit_cv``.

        Returns
        -------
        report : riskslim.report.ModelReport
            Use ``report.save(path)`` to write an HTML file; notebooks display it inline.
        """
        return ModelReport(self, data=data, folds=folds, model_type=model_type,
                           X_test=X_test, y_test=y_test)

    def decision_function(self, X):
        """Risk score of each sample; > 0 predicts ``classes_[1]``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : 1d array of shape (n_samples,)
        """
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return X.dot(self.coef_) + self.intercept_

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : 1d array of shape (n_samples,)
        """
        check_is_fitted(self)
        if getattr(self, "calibrated_estimator_", None) is not None:
            return self.calibrated_estimator_.predict(X)
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def predict_proba(self, X):
        """Probability estimates, one column per class in ``classes_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : 2d array of shape (n_samples, 2)
        """
        check_is_fitted(self)
        if getattr(self, "calibrated_estimator_", None) is not None:
            return self.calibrated_estimator_.predict_proba(X)
        positive = expit(self.decision_function(X))
        return np.column_stack([1.0 - positive, positive])

    def predict_log_proba(self, X):
        """Log of the probability estimates, one column per class in ``classes_``."""
        return np.log(self.predict_proba(X))

    def recalibrate(self, X, y, method = "sigmoid"):
        """Recalibrate the fitted risk scores (Platt scaling or isotonic regression).

        After this call, ``predict`` and ``predict_proba`` use the calibrator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        method : {"sigmoid", "isotonic"}
            Calibration method.
        """
        check_is_fitted(self)
        self.calibrated_estimator_ = None
        calibrator = CalibratedClassifierCV(estimator = FrozenEstimator(self), method = method)
        self.calibrated_estimator_ = calibrator.fit(X, y)

        if getattr(self, "cv_results_", None) is not None:
            # one calibrator per fold, each on its own fold model
            X, y = np.asarray(X), np.ravel(y)
            self.cv_calibrated_estimators_ = []
            for fold_estimator, train in zip(self.cv_results_["estimator"], self.cv_results_["indices"]["train"]):
                calibrator = CalibratedClassifierCV(estimator = FrozenEstimator(fold_estimator), method = method)
                self.cv_calibrated_estimators_.append(calibrator.fit(X[train], y[train]))
        return self

    def fit_cv(self, X, y, k=5, scoring="roc_auc", n_jobs=1, *, data=None, fold_id=None, **kwargs):
        """Cross-validate RiskSLIM; stores ``cv_`` and ``cv_results_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        k : int, sklearn cross-validation generator or an iterable, default: 5
            Determines the cross-validation splitting strategy. With ``data`` and no ``fold_id``,
            only the fold count is used: the folds are ``data.cv`` with ``k`` folds, replicate 1.
        scoring : str or callable, default: "roc_auc"
            Strategy to evaluate the cross-validated model on each test fold.
        n_jobs : int, optional, default: 1
            Number of jobs to run in parallel. -1 defaults to max cores or threads.
        data : riskslim.data.BinaryClassificationDataset, optional
            Dataset whose rows are ``X`` and ``y``; its folds ``fold_id`` are the CV folds.
        fold_id : str, optional
            Fold id in ``data.cv``, e.g. ``"K05N01"`` (5 folds, replicate 1). None uses
            ``k`` folds, replicate 1.
        **kwargs
            Settings passed to each fold's ``fit``.
        """
        scoring = check_scoring(self, scoring)
        self.__dict__.pop("cv_calibrated_estimators_", None)  # calibrated the previous folds
        cv = k if data is None else dataset_folds(data, CVFolds.get_fold_id(k) if fold_id is None else fold_id,
                                                  n_samples=len(y))
        self.cv_ = check_cv(cv=cv, y=y, classifier=True)
        self.cv_results_ = cross_validate(
                self,
                X=X,
                y=y,
                cv=self.cv_,
                return_estimator=True,
                return_indices=True,
                scoring=scoring,
                params=kwargs,
                n_jobs=n_jobs
                )
        return self


def dataset_folds(data, fold_id, n_samples):
    """The folds ``fold_id`` of ``data`` as a CV splitter, read without splitting ``data``."""
    if not isinstance(data, BinaryClassificationDataset):
        raise TypeError(f"data must be a BinaryClassificationDataset; got {type(data).__name__}")
    if data.n != n_samples:
        raise ValueError(f"data has {data.n} rows but y has {n_samples}; pass data.X and data.y")
    if fold_id not in data.cv:
        raise ValueError(f"data.cv has no folds {fold_id!r}; build the dataset with those folds "
                         f"(e.g. n_folds=(5,) for 'K05N01')")
    return PredefinedSplit(np.asarray(data.cv[fold_id]))

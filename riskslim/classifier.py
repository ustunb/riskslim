"""RiskSLIM Classifier."""

import copy

import numpy as np
from scipy.special import expit

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_validate, check_cv
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

from .optimizer import RiskSLIMOptimizer
from .reporter import RiskScoreReporter
from .coefficient_set import CoefficientSet
from .data import ClassificationDataset
from .defaults import DEFAULT_LCPA_SETTINGS, OUTCOME_NAME


def is_settings_key(key):
    """True if ``key`` names an LCPA, warmstart ('init_*'), or CPLEX ('cplex_*') setting."""
    return key in DEFAULT_LCPA_SETTINGS or key.startswith(("init_", "cplex_"))


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
    reporter_ : riskslim.reporter.RiskScoreReporter
        Risk score table, derived metrics, and reports.
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
                 verbose = True,  **kwargs):
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
        self._settings = dict(kwargs)

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
        # the optimizer (CPLEX model, closures) and the reporter (PrettyTable) do not pickle;
        # the reporter is rebuilt on unpickling, the optimizer is not
        state = dict(super().__getstate__())
        state.pop("optimizer_", None)
        state.pop("reporter_", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if hasattr(self, "coef_"):
            self.reporter_ = RiskScoreReporter.from_model(estimator=self)

    def __repr__(self, N_CHAR_MAX=700):
        if hasattr(self, "reporter_"):
            return self.reporter_.__repr__()
        return super().__repr__(N_CHAR_MAX=N_CHAR_MAX)

    @property
    def fitted(self):
        """Whether the model has been fit."""
        return hasattr(self, "coef_")

    @property
    def optimizer(self):
        """The fitted optimizer, or None before fit (alias of ``optimizer_``)."""
        return getattr(self, "optimizer_", None)

    @property
    def reporter(self):
        """The fitted reporter, or None before fit (alias of ``reporter_``)."""
        return getattr(self, "reporter_", None)

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
        unknown = sorted(key for key in settings if not is_settings_key(key))
        if unknown:
            raise ValueError(f"Unknown RiskSLIM settings: {unknown}")

        X, y = validate_data(self, X, y, dtype=np.float64)
        check_classification_targets(y)
        y_type = type_of_target(y, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                f"RiskSLIMClassifier needs y with exactly 2 classes; got {len(classes)} class(es)."
            )
        self.classes_ = classes
        y_signed = np.where(y == classes[1], 1, -1)

        variable_names = self.variable_names
        if variable_names is None and hasattr(self, "feature_names_in_"):
            variable_names = list(self.feature_names_in_)
        outcome_name = OUTCOME_NAME if self.outcome_name is None else self.outcome_name
        self._data = ClassificationDataset(X, y_signed, variable_names, outcome_name)

        if self.coef_set is None:
            self.coef_set_ = CoefficientSet(self._data.variable_names, lb = -self.max_coef, ub = self.max_coef)
        else:
            self.coef_set_ = copy.deepcopy(self.coef_set)
        self.max_size_ = self._data.d if self.max_size is None else self.max_size

        self.optimizer_ = RiskSLIMOptimizer(
            data = self._data,
            coef_set = self.coef_set_,
            max_size = self.max_size_,
            c0_value = self.c0_value,
            verbose = self.verbose,
            **settings,
        )
        self.optimizer_.optimize(self._data.X, self._data.y)

        coefficients = self.optimizer_.coefficients
        self.coef_ = coefficients[1:]
        self.intercept_ = coefficients[0]
        self._variable_types = self._data.variable_types
        self.reporter_ = RiskScoreReporter.from_model(estimator=self)
        return self

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
            for fold_estimator, (train, _) in zip(self.cv_results_["estimator"], self.cv_.split(X, y)):
                calibrator = CalibratedClassifierCV(estimator = FrozenEstimator(fold_estimator), method = method)
                self.cv_calibrated_estimators_.append(calibrator.fit(X[train], y[train]))
        return self

    def fit_cv(self, X, y, k=5, scoring="roc_auc", n_jobs=1, **kwargs):
        """Cross-validate RiskSLIM; stores ``cv_`` and ``cv_results_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        k : int, sklearn cross-validation generator or an iterable, default: 5
            Determines the cross-validation splitting strategy.
        scoring : str or callable, default: "roc_auc"
            Strategy to evaluate the cross-validated model on each test fold.
        n_jobs : int, optional, default: 1
            Number of jobs to run in parallel. -1 defaults to max cores or threads.
        **kwargs
            Settings passed to each fold's ``fit``.
        """
        scoring = check_scoring(self, scoring)
        self.cv_ = check_cv(cv=k, y=y, classifier=True)
        self.cv_results_ = cross_validate(
                self,
                X=X,
                y=y,
                cv=self.cv_,
                return_estimator=True,
                scoring=scoring,
                params=kwargs,
                n_jobs=n_jobs
                )
        return self

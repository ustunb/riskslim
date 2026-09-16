"""RiskSLIM Classifier."""

from pathlib import Path
import numpy as np
from scipy.special import expit

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_validate, check_cv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import check_scoring

from .optimizer import RiskSLIMOptimizer
from .reporter import RiskScoreReporter
from .coefficient_set import CoefficientSet
from .data import ClassificationDataset


class RiskSLIMClassifier(BaseEstimator, ClassifierMixin):
    """RiskSLIM classifier

    Attributes
    ----------
    X : 2d array
        Observations (rows) and features (columns).
    y : 1d or 2d array
        Class labels.
    coef_set : riskslim.coefficient_set.CoefficientSet
        Constraints on coefficients of input variables
    bounds : riskslim.data.Bounds
        Lower and upper bounds on objective value, loss, and model size.
    stats : riskslim.data.Stats
        Cplex solution statistics.
    solution : cplex SolutionInterface
        Solved cplex solution.
    solution_info : dict
        Additional solution information.
    pool : riskslim.solution_pool.SolutionPool
        Pool of solutions and associated objective values.
    rho : 1d array
        Solved cofficients, includes intercept.
    coefs_ : 1d array
        Solved cofficients, excludes intercept. For compatibility with sci-kit learn.
    intercept_ : 1d array
        Intercept coefficient. For compatibility with sci-kit learn.
    fitted : bool
        Whether model has be fit.
    report : riskslim.risk_scores.RiskScoreReporter
        Risk scores, derived metrics, and reports.
    cv : sklearn.model_selection._split.KFold
        Cross-validation.
    cv_results : dict
        Cross-validation results.
    calibrated_estimator : sklearn.calibration.CalibratedClassifierCV
        Single calibrator trained on all data.
    calibrated_estimators_ : list of sklearn.calibration.CalibratedClassifierCV
        Calibrators trained per fold. Must use the fitcv method.
    """
    def __init__(self, max_coef = 5, max_size = None, variable_names = None,
                 outcome_name = None, verbose = True,  **kwargs):
        """
        Parameters
        ----------
        max_coef : float or 1d array, optional, default: None
            Maximum coefficient.
            None defaults to 5.
        max_size : int, optional, default: None
            Maximum number of regularized coefficients.
            None defaults to length of input variables.
        variable_names : list of str, optional, default: None
            Names of each features. Only needed if coefficients is not passed on initalization.
            None defaults to generic variable names.
        outcome_name : str, optional, default: None
            Name of the output class.
        coef_set: riskslim.coefficient_set.CoefficientSet
            Contraints (bounds) on coefficients of input variables.
            If None, this is constructed based on values passed to other initalization kwargs.
            If not None, other kwargs may be overwritten.
        c0_value : 1d array or float, optional, default: None
            L0-penalty for all parameters when an integer or for each parameter
            separately when an array.
            None defaults to 1e-6.
        verbose : bool, optional, default: True
            Prints out log information if True, supresses if False.
        settings : dict, optional, default: None
            Settings for warmstart (keys: 'init_*'), cplex (keys: 'cplex_*'), and lattice CPA.
            Use of this explicit kwarg allows cloning via sklearn (e.g. for grid search).
            Settings are combined with kwargs and then parsed. Parameters including in
            in kwargs will not be accessible or set after a clone.
        **kwargs
            - \*\*settings : unpacked dict
                Settings for warmstart (keys: \'init\_\'), cplex (keys: \'cplex\_\'), and lattice CPA.
                Defaults are defined in ``defaults.DEFAULT_LCPA_SETTINGS``.
        """
        self.verbose = verbose
        self.fitted = False
        self.optimizer = None

        # default classifier inputs
        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None

        # parameters for tuning
        # todo: check that these are the right type / shape
        self.max_size = max_size # positive integer-valued
        self.max_coef = max_coef # positive integer-valued

        # internals
        self._data = None
        self._variable_names = variable_names
        self._outcome_name = outcome_name if outcome_name is not None else "outcome"
        self._coef_set = None

        # todo: check that this
        self._settings = kwargs

        # output
        self.calibrated_estimator = None
        self.cv = None
        self.cv_results = None
        self.cv_calibrated_estimators = None
        self.reporter = None


    def __repr__(self):
        if self.fitted:
            return self.reporter.__repr__()
        else:
            return super().__repr__()


    def fit(self, X, y, sample_weights=None, **kwargs):
        """Fit RiskSLIM classifier.

        Parameters
        ----------
        X : 2d-array
            Observations (rows) and features (columns).
            With an addtional column of 1s for the intercept.
        y : 2d-array
            Class labels (+1, -1) with shape (n_rows, 1).
        sample_weights : 2d array, optional, default: None
            Sample weights with shape (n_features, 1). Must all be positive.
        """
        self._data = ClassificationDataset(X, y, self._variable_names, self._outcome_name, sample_weights)
        self.max_size = self._data.d if self.max_size is None else self.max_size
        self.classes_ = self._data.classes

        # todo: construct
        if self._coef_set is None:
            self._coef_set = CoefficientSet(self._data.variable_names, lb = -self.max_coef, ub = self.max_coef)
        self.max_coef = self._coef_set.max_coef

        # Initialize optimizer
        if self.optimizer is None:
            self.optimizer = RiskSLIMOptimizer(data = self._data, coef_set = self._coef_set, max_size = self.max_size, verbose=self.verbose, **kwargs)

        # fit
        self.optimizer.optimize(self._data.X, self._data.y, self._data.sample_weights, **kwargs)
        self.fitted = True

        # Attributes
        self.coef_ = self.optimizer.coefficients[1:]
        self.intercept_ = self.optimizer.coefficients[0]

        # pass initalize attributes from optimzier to self
        self._variable_types = self._data.variable_types

        # Inialize a reporter
        self.reporter = RiskScoreReporter.from_model(estimator=self)

        return self.__repr__()


    def predict(self, X):
        """Predict labels.

        Parameters
        ----------
        X : 2d-array
            Observations (rows) and features (columns).
            With an addtional column of 1s for the intercept.

        Returns
        -------
        y_pred : 1d-array
            Predicted labels of X.
        """
        assert self.fitted
        if self.calibrated_estimator is None:
            # Normal case
            y_pred = np.sign(proba = expit(self.decision_function(X)))
        elif isinstance(self.calibrated_estimator, CalibratedClassifierCV):
            # Calibrator
            y_pred = self.calibrated_estimator.predict(X)

        return y_pred

    def predict_proba(self, X):
        """Probability estimates.

        Parameters
        ----------
        X : 2d-array
            Observations (rows) and features (columns).
            With an addtional column of 1s for the intercept.

        Returns
        -------
        probs : 1d array
            Probability of classes.
        """
        assert self.fitted

        if self.calibrated_estimator is None:
            # Normal case
            proba = expit(self.decision_function(X))
        elif isinstance(self.calibrated_estimator, CalibratedClassifierCV):
            # Calibrator
            proba = self.calibrated_estimator.predict_proba(X)[:, 1]

        return proba

    def predict_log_proba(self, X):
        """Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.y``.
        """
        return np.log(self.predict_proba(X))

    def decision_function(self, X):
        """Predict confidence scores for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Confidence scores per ``(n_samples, n_classes)`` combination. In the
            binary case, confidence score for ``self.classes_[1]`` where >0 means
            this class would be predicted.
        """
        return X.dot(self.coef_) + self.intercept_

    def recalibrate(self, X, y, sample_weights=None, method= "sigmoid"):
        """Fit RiskSLIM classifier via Platt Scaling.

        Parameters
        ----------
        X : 2d-array
            Observations (rows) and features (columns).
            With an addtional column of 1s for the intercept.
        y : 2d-array
            Class labels (+1, -1) with shape (n_rows, 1).
        sample_weights : 2d array, optional, default: None
            Sample weights with shape (n_features, 1). Must all be positive.
        method : {"sigmoid", "isotonic"}
            Linear classifier used to recalibrate scores.
        """
        # todo: call check_data
        # todo: add support for kwargs (method = 'sigmoid') should be

        if not self.fitted:
            raise ValueError("fit RiskSLIM before calling recalibrate")

        # fit recalibrate final model
        clf = CalibratedClassifierCV(base_estimator = self, cv="prefit", method=method)
        clf.fit(X = X, y = y, sample_weights = sample_weights)
        self.calibrated_estimator = clf

        if self.cv_results is not None:
            # Compute an ensemble (1 per fold) of calibrators / recalibrate cv models
            cv_calibrated_estimators = []
            for train, _ in self.cv.split(X, y.reshape(-1)):
                clf = CalibratedClassifierCV(base_estimator = self, cv="prefit", method=method)
                if sample_weights is None:
                    clf.fit(X = X[train], y = y[train])
                else:
                    clf.fit(X = X[train], y = y[train], sample_weights = sample_weights[train])
                cv_calibrated_estimators.append(clf)

            self.cv_calibrated_estimators = cv_calibrated_estimators

    def fit_cv(
            self,
            X,
            y,
            sample_weights=None,
            k=5,
            scoring="roc_auc",
            n_jobs=1,
            **kwargs
            ):
        """Train RiskSLIM classifier.

        Parameters
        ----------
        X : 2d-array
            Observations (rows) and features (columns).
            With an addtional column of 1s for the intercept.
        y : 2d-array
            Class labels (+1, -1) with shape (n_rows, 1).
        sample_weights : 2d array, optional, default: None
            Sample weights with shape (n_samples, 1). Must all be positive.
        k : int, sklearn cross-validation generator or an iterable, default: 5
            Determines the cross-validation splitting strategy.
            Possible inputs for k are:

            - None, to use the default 5-fold cross validation,
            - int, to specify the number of folds in a ``(Stratified)KFold``,
            - ``CV splitter``,
            - An iterable yielding (train, test) splits as arrays of indices.

        scoring : str, callable, list, tuple, or dict, default: "roc_auc"
            Strategy to evaluate the performance of the cross-validated model on
            the test set.

            - a single string (see sklearn ``scoring_parameter``);
            - a callable (see sklearn ``scoring``) that returns a single value.

        n_jobs : int, optional, default: 1
            Number of jobs to run in parallel. -1 defaults to max cores or threads.
        """

        # Get scorer and cross-validator
        scoring = check_scoring(self, scoring)
        self.cv = check_cv(cv=k, y=y)

        # Run cross-validation
        fit_params = {"sample_weights": sample_weights}
        fit_params.update(kwargs)

        self.cv_results = cross_validate(
                self,
                X=X,
                y=y,
                cv=self.cv,
                return_estimator=True,
                scoring=scoring,
                fit_params= fit_params,
                n_jobs=n_jobs
                )
"""SMQ — Synthetic-score Meta-learning for Quantification."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.utils.validation import check_is_fitted

from .calibration import IsotonicPrevalenceCalibrator, uniform_prevalence_batches
from .metafeatures import score_moments
from .synthetic import curves_to_training_set, synthetic_curves
from .utils import normalize_prevalence, prevalence_from_labels


class SMQ(BaseEstimator):
    """Estimate class prevalences from synthetic score distributions.

    SMQ reformulates quantification as meta-learning: a regressor learns to map
    the meta-features of a score distribution (four moments per class) to the
    prevalence vector that produced it. The regressor is trained **only on
    synthetic scores**; real labelled data are used by the base classifier and,
    optionally, by a light per-class isotonic calibration (SMQ-ISO).

    Parameters
    ----------
    classifier : sklearn classifier, optional
        Probabilistic classifier that produces the scores. Defaults to a
        ``RandomForestClassifier(n_estimators=300)``, as in the paper.
    calibrate : bool, default=True
        Fit the isotonic calibration on real data (SMQ-ISO). ``False`` gives the
        purely synthetic SMQ.
    regressor : sklearn regressor, optional
        Multi-output regressor from meta-features to prevalences. Defaults to
        ``RandomForestRegressor(n_estimators=300, min_samples_leaf=2)``.
    n_prevalences, n_merging, n_replicates, sample_size : int
        Size of the synthetic training set: prevalence vectors sampled from
        ``Dirichlet(1)``, merging factors evenly spaced in ``[0, 1]``, samples
        per pair and instances per sample. Defaults reproduce the synthetic
        files of the published experiment (15 x 15 x 20 samples of 100
        instances).
    synthetic : dict, optional
        Pre-computed synthetic curves (``{(prevalence, m): [scores, ...]}``),
        e.g. loaded from the files of the published experiment. Overrides the
        four parameters above.
    cv : int, default=5
        Folds for the out-of-fold scores used by the calibration.
    calib_batch_size, calib_prevalences, calib_repeats : int
        Calibration batches drawn from the training data with the Uniform
        Prevalence Protocol (defaults: 10 prevalences x 25 repetitions of 500
        instances, as in the paper).
    random_state : int, optional
    n_jobs : int, optional
        Parallelism of the default random forests and of ``cross_val_predict``.

    Attributes
    ----------
    classes_ : ndarray
        Class labels seen in ``fit``; prevalences follow this order.
    classifier_ : fitted classifier
    regressor_ : fitted regressor
    calibrator_ : IsotonicPrevalenceCalibrator or None

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from smq import SMQ
    >>> X, y = load_digits(return_X_y=True)
    >>> q = SMQ(random_state=0).fit(X[:1200], y[:1200])
    >>> q.predict(X[1200:]).round(3)  # doctest: +SKIP
    """

    def __init__(
        self,
        classifier=None,
        calibrate: bool = True,
        regressor=None,
        n_prevalences: int = 15,
        n_merging: int = 15,
        n_replicates: int = 20,
        sample_size: int = 100,
        synthetic: dict | None = None,
        cv: int = 5,
        calib_batch_size: int = 500,
        calib_prevalences: int = 10,
        calib_repeats: int = 25,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ):
        self.classifier = classifier
        self.calibrate = calibrate
        self.regressor = regressor
        self.n_prevalences = n_prevalences
        self.n_merging = n_merging
        self.n_replicates = n_replicates
        self.sample_size = sample_size
        self.synthetic = synthetic
        self.cv = cv
        self.calib_batch_size = calib_batch_size
        self.calib_prevalences = calib_prevalences
        self.calib_repeats = calib_repeats
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------ fit
    def _default_classifier(self):
        return RandomForestClassifier(
            n_estimators=300, n_jobs=self.n_jobs, random_state=self.random_state
        )

    def _default_regressor(self):
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def fit(self, X, y):
        """Fit the classifier, the synthetic regressor and the calibration."""
        X = np.asarray(X)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("SMQ needs at least two classes.")
        self.n_classes_ = n_classes

        base = self.classifier if self.classifier is not None else self._default_classifier()
        self.classifier_ = clone(base).fit(X, y_idx)

        # 1) regressor trained only on synthetic score distributions
        curves = self.synthetic
        if curves is None:
            curves = synthetic_curves(
                n_classes,
                n_prevalences=self.n_prevalences,
                n_merging=self.n_merging,
                n_replicates=self.n_replicates,
                sample_size=self.sample_size,
                random_state=self.random_state,
            )
        Z, P = curves_to_training_set(curves, n_classes)
        reg = self.regressor if self.regressor is not None else self._default_regressor()
        self.regressor_ = clone(reg).fit(Z, P)

        # 2) optional isotonic calibration on real (out-of-fold) scores
        self.calibrator_ = None
        if self.calibrate:
            oof = cross_val_predict(
                clone(base), X, y_idx, cv=self.cv, method="predict_proba", n_jobs=self.n_jobs
            )
            estimated, true = [], []
            for idx in uniform_prevalence_batches(
                y_idx,
                n_classes,
                batch_size=self.calib_batch_size,
                n_prevalences=self.calib_prevalences,
                repeats=self.calib_repeats,
                random_state=self.random_state,
            ):
                estimated.append(self._regress(oof[idx]))
                true.append(prevalence_from_labels(y_idx[idx], n_classes))
            self.calibrator_ = IsotonicPrevalenceCalibrator().fit(estimated, true)
        return self

    # -------------------------------------------------------------- predict
    def _regress(self, scores) -> np.ndarray:
        z = score_moments(scores, self.n_classes_).reshape(1, -1)
        return normalize_prevalence(self.regressor_.predict(z)[0], self.n_classes_)

    def aggregate(self, scores) -> np.ndarray:
        """Prevalence estimate from a matrix of classifier scores."""
        check_is_fitted(self, "regressor_")
        p = self._regress(scores)
        if self.calibrator_ is not None:
            p = self.calibrator_.transform(p)
        return p

    def predict(self, X) -> np.ndarray:
        """Estimated prevalence of each class (in ``classes_`` order) in ``X``."""
        check_is_fitted(self, "classifier_")
        return self.aggregate(self.classifier_.predict_proba(np.asarray(X)))

    def predict_dict(self, X) -> dict:
        """Same as :meth:`predict`, keyed by class label."""
        return dict(zip(self.classes_.tolist(), self.predict(X).tolist()))

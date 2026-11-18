"""Post-hoc calibration that bridges the synthetic-to-real gap.

The regressor of SMQ never sees real data, so its estimates can be shifted when
real score distributions differ from the synthetic ones. SMQ-ISO fits one
isotonic regression per class on pairs ``(estimated, true)`` prevalences from
real calibration samples and applies it at prediction time (Section 3.4).
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .utils import class_counts, normalize_prevalence


class IsotonicPrevalenceCalibrator:
    """One monotone map per class from estimated to true prevalence.

    ``increasing='auto'`` lets each class pick its direction from the data, and
    predictions are clipped to ``[0, 1]`` before the vector is renormalised.
    """

    def fit(self, estimated, true):
        estimated = np.asarray(estimated, dtype=float)
        true = np.asarray(true, dtype=float)
        if estimated.shape != true.shape:
            raise ValueError(
                f"shape mismatch: estimated {estimated.shape} vs true {true.shape}"
            )
        self.n_classes_ = estimated.shape[1]
        self.models_ = []
        for c in range(self.n_classes_):
            iso = IsotonicRegression(
                y_min=0.0, y_max=1.0, increasing="auto", out_of_bounds="clip"
            )
            with warnings.catch_warnings():
                # 'auto' warns when the direction is uncertain; it still picks one
                warnings.simplefilter("ignore", category=UserWarning)
                iso.fit(estimated[:, c], true[:, c])
            self.models_.append(iso)
        return self

    def transform(self, prevalence) -> np.ndarray:
        prevalence = np.asarray(prevalence, dtype=float)
        calibrated = np.array(
            [self.models_[c].predict([prevalence[c]])[0] for c in range(self.n_classes_)]
        )
        return normalize_prevalence(calibrated, self.n_classes_)


def uniform_prevalence_batches(
    y,
    n_classes: int,
    batch_size: int = 500,
    n_prevalences: int = 10,
    repeats: int = 25,
    random_state=None,
):
    """Index batches following the Uniform Prevalence Protocol (UPP).

    Prevalence vectors are drawn uniformly from the simplex and each one is
    realised ``repeats`` times by sampling ``batch_size`` indices of ``y`` with
    replacement, class by class. Classes absent from ``y`` are skipped and the
    remaining proportions renormalised.

    Yields
    ------
    ndarray of int
        Indices into ``y`` for one batch.
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y, dtype=int)
    pools = [np.flatnonzero(y == c) for c in range(n_classes)]
    present = np.array([len(pool) > 0 for pool in pools])

    for _ in range(n_prevalences):
        prevalence = rng.dirichlet(np.ones(n_classes)) * present
        prevalence = prevalence / prevalence.sum()
        for _ in range(repeats):
            counts = class_counts(batch_size, prevalence)
            idx = [
                rng.choice(pools[c], size=counts[c], replace=True)
                for c in range(n_classes)
                if counts[c] > 0
            ]
            yield rng.permutation(np.concatenate(idx))

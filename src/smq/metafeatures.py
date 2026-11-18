"""Meta-features that describe a score distribution.

A sample is represented by the matrix of classifier scores ``S`` of shape
``(n_instances, n_classes)``. For every class column SMQ computes four
statistics — mean, variance, skewness and kurtosis — and concatenates them::

    z = [mu_1, var_1, skew_1, kurt_1, ..., mu_c, var_c, skew_c, kurt_c]

so ``z`` has ``4 * n_classes`` entries (Section 3.3 of the paper).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import kurtosis, skew

N_MOMENTS = 4
FEATURE_NAMES = ("mean", "var", "skew", "kurt")


def score_moments(scores, n_classes: int | None = None) -> np.ndarray:
    """Mean, variance, skewness and kurtosis of each score column.

    Parameters
    ----------
    scores : array-like of shape (n_instances, n_columns)
        Posterior probabilities (or any soft scores) of a sample.
    n_classes : int, optional
        Length of the target representation. Shorter feature vectors are
        zero-padded and longer ones truncated to ``4 * n_classes``, which lets
        a single regressor serve samples with fewer columns. Defaults to the
        number of columns in ``scores``.

    Returns
    -------
    ndarray of shape (4 * n_classes,)

    Notes
    -----
    Skewness and kurtosis use SciPy's defaults (biased estimators, Fisher's
    excess kurtosis), exactly as in the published experiment. A constant column
    makes both undefined; they are reported as 0 instead of NaN.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got shape {scores.shape}")
    n_classes = scores.shape[1] if n_classes is None else int(n_classes)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        feats = np.column_stack(
            [
                scores.mean(axis=0),
                scores.var(axis=0),
                skew(scores, axis=0),
                kurtosis(scores, axis=0),
            ]
        ).ravel()

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    expected = N_MOMENTS * n_classes
    if feats.size < expected:
        feats = np.concatenate([feats, np.zeros(expected - feats.size)])
    return feats[:expected]


def feature_names(n_classes: int) -> list[str]:
    """Readable names for the entries returned by :func:`score_moments`."""
    return [f"{name}_{c}" for c in range(n_classes) for name in FEATURE_NAMES]

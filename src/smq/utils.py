"""Small helpers shared by the rest of the package."""

from __future__ import annotations

import numpy as np


def normalize_prevalence(p, n_classes: int) -> np.ndarray:
    """Project any prevalence-like output onto the probability simplex.

    Accepts a dict ``{class_index: value}`` or an array. The vector is padded or
    truncated to ``n_classes``, non-finite entries become 0, values are clipped
    to ``[0, 1]`` and the result is renormalised to sum to one. An all-zero
    vector falls back to the uniform distribution.

    This is the ``safe_prevalence_vector`` routine used in the original
    experiment, kept bit-for-bit so that results stay comparable.
    """
    if isinstance(p, dict):
        vec = np.zeros(n_classes)
        for k, v in p.items():
            if int(k) < n_classes:
                vec[int(k)] = v
        p = vec
    else:
        p = np.asarray(p, dtype=float).ravel()

    if len(p) < n_classes:
        padded = np.zeros(n_classes)
        padded[: len(p)] = p
        p = padded

    p = np.nan_to_num(p[:n_classes], nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)

    total = p.sum()
    if total <= 0:
        return np.full(n_classes, 1.0 / n_classes)
    return p / total


def prevalence_from_labels(y, n_classes: int) -> np.ndarray:
    """Relative frequency of each class ``0 .. n_classes - 1`` in ``y``."""
    counts = np.bincount(np.asarray(y, dtype=int), minlength=n_classes)[:n_classes]
    return counts / max(counts.sum(), 1)


def class_counts(n: int, prevalence) -> np.ndarray:
    """Split ``n`` observations between classes in the given proportions.

    Each class receives the floor of its share; the observations left over go
    to the classes with the largest fractional parts (largest-remainder rule),
    so the counts always add up to ``n``.
    """
    prevalence = np.asarray(prevalence, dtype=float)
    exact = n * prevalence
    counts = np.floor(exact).astype(int)
    shortfall = n - counts.sum()
    if shortfall > 0:
        order = np.argsort(counts - exact, kind="stable")
        counts[order[:shortfall]] += 1
    return counts


def absolute_error(p_true, p_hat) -> float:
    """Mean absolute error between two prevalence vectors (the paper's MAE)."""
    return float(np.mean(np.abs(np.asarray(p_true) - np.asarray(p_hat))))

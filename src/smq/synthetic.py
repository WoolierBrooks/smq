"""Synthetic score distributions for training SMQ (multiclass MoSS).

MoSS (Model for Score Simulation; Maletzke et al., ICDM 2021) simulates the
scores a classifier would output for a sample with a known prevalence. A
*merging factor* ``m`` controls how much the classes overlap: ``m = 0`` gives
well-separated classes, ``m = 1`` indistinguishable ones.

The multiclass version used to train SMQ draws scores on the probability
simplex. The instances of class ``k`` get scores from a Dirichlet whose
concentration interpolates, according to ``m``, between a configuration
peaked at the vertex of class ``k`` and the uniform configuration, with an
overall scale that shrinks as ``m`` grows::

    scale   = 50 (1 - m) + 5
    center  = normalise(1/C everywhere, 1 at class k)
    alpha_k = eps + scale * ((1 - m) * center + m * 1/C)

The concentration rule is the one in the generator notebook of the thesis
repository (``experiments/exp_010/note.ipynb``), which writes the files
``moss_d500_lite_<C>.pkl`` used in the published run: 15 prevalence vectors
drawn from ``Dirichlet(1_C)`` x 15 merging factors evenly spaced in ``[0, 1]``
x 20 samples per pair, each with 100 instances. The committed notebook has
``n_samples=150``, but the original files have 100 instances per sample, as the
paper states, so 100 is the default here. The only other change is that
randomness goes through an explicit, seedable generator.
"""

from __future__ import annotations

import numpy as np

from .metafeatures import score_moments

EPS = 1e-3


def class_concentration(k: int, n_classes: int, merging_factor: float, eps: float = EPS):
    """Dirichlet concentration of the scores of class ``k``."""
    m = float(np.clip(merging_factor, 0.0, 1.0))
    scale = 50.0 * (1.0 - m) + 5.0
    center = np.full(n_classes, 1.0 / n_classes)
    center[k] = 1.0
    center = center / center.sum()
    uniform = np.full(n_classes, 1.0 / n_classes)
    return eps + scale * ((1.0 - m) * center + m * uniform)


def simulate_scores(n, prevalence, merging_factor, random_state=None, eps: float = EPS):
    """Draw ``n`` synthetic score vectors for a sample with a given prevalence.

    Parameters
    ----------
    n : int
        Number of instances in the sample.
    prevalence : array-like of shape (n_classes,)
        Class proportions of the sample.
    merging_factor : float
        Overlap between the class-conditional score distributions, in [0, 1].
    random_state : int, numpy Generator or None

    Returns
    -------
    scores : ndarray of shape (n, n_classes)
        Each row lies on the probability simplex.
    labels : ndarray of shape (n,)
        Class each row was drawn for.

    Notes
    -----
    Each class receives ``floor(n * p_k)`` instances and the last class takes
    whatever is left, as in the original generator.
    """
    rng = np.random.default_rng(random_state)
    prevalence = np.asarray(prevalence, dtype=float)
    n_classes = len(prevalence)
    counts = np.floor(n * prevalence).astype(int)
    counts[-1] = n - counts[:-1].sum()

    scores = np.zeros((n, n_classes))
    labels = np.zeros(n, dtype=int)
    start = 0
    for k, count in enumerate(counts):
        if count <= 0:
            continue
        alpha = class_concentration(k, n_classes, merging_factor, eps)
        scores[start : start + count] = rng.dirichlet(alpha, size=count)
        labels[start : start + count] = k
        start += count

    order = rng.permutation(n)
    return scores[order], labels[order]


def synthetic_curves(
    n_classes: int,
    n_prevalences: int = 15,
    n_merging: int = 15,
    n_replicates: int = 20,
    sample_size: int = 100,
    random_state=None,
) -> dict:
    """Generate the synthetic score distributions used to train SMQ.

    Returns a dict ``{(prevalence_tuple, merging_factor): [scores, ...]}`` with
    ``n_replicates`` score matrices of shape ``(sample_size, n_classes)`` per
    key — the layout of the ``moss_d500_lite_<C>.pkl`` files.
    """
    rng = np.random.default_rng(random_state)
    prevalences = rng.dirichlet(np.ones(n_classes), size=n_prevalences)
    merging = np.linspace(0.0, 1.0, n_merging)
    curves = {}
    for prevalence in prevalences:
        key = tuple(float(v) for v in np.round(prevalence, 4))
        for m in merging:
            curves[(key, round(float(m), 4))] = [
                simulate_scores(sample_size, prevalence, m, rng)[0]
                for _ in range(n_replicates)
            ]
    return curves


def curves_to_training_set(curves: dict, n_classes: int):
    """Turn synthetic curves into the regression set ``(Z, P)``.

    ``Z`` holds the meta-features of each synthetic sample and ``P`` the
    (rounded) prevalence vector that generated it, as in the experiment.
    """
    Z, P = [], []
    for (prevalence, _merging), matrices in curves.items():
        target = np.asarray(prevalence, dtype=float)
        for scores in matrices:
            Z.append(score_moments(scores, n_classes))
            P.append(target)
    return np.asarray(Z), np.asarray(P)

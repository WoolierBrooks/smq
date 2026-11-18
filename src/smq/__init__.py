"""SMQ: Synthetic Score Meta-Learning for Multiclass Quantification.

Riveros Cardus, J. C., Maletzke, A. G. & Zalewski, W. (2026).
IBERAMIA 2026 — Ibero-American Conference on Artificial Intelligence.
"""

from .calibration import IsotonicPrevalenceCalibrator, uniform_prevalence_batches
from .metafeatures import feature_names, score_moments
from .quantifier import SMQ
from .synthetic import curves_to_training_set, simulate_scores, synthetic_curves
from .utils import absolute_error, normalize_prevalence, prevalence_from_labels

__version__ = "1.0.0"

__all__ = [
    "SMQ",
    "IsotonicPrevalenceCalibrator",
    "uniform_prevalence_batches",
    "score_moments",
    "feature_names",
    "simulate_scores",
    "synthetic_curves",
    "curves_to_training_set",
    "normalize_prevalence",
    "prevalence_from_labels",
    "absolute_error",
    "__version__",
]

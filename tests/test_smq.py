import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from smq import (
    SMQ,
    IsotonicPrevalenceCalibrator,
    curves_to_training_set,
    normalize_prevalence,
    score_moments,
    simulate_scores,
    synthetic_curves,
    uniform_prevalence_batches,
)

FAST = dict(n_prevalences=3, n_merging=3, n_replicates=2, sample_size=60,
            calib_prevalences=3, calib_repeats=2, calib_batch_size=100, cv=3)


def small_smq(**kw):
    return SMQ(
        classifier=RandomForestClassifier(n_estimators=10, random_state=0),
        regressor=RandomForestRegressor(n_estimators=10, random_state=0),
        random_state=0, **FAST, **kw,
    )


def test_score_moments_layout_and_padding():
    scores = np.random.default_rng(0).dirichlet(np.ones(3), size=200)
    z = score_moments(scores)
    assert z.shape == (12,)
    assert z[0] == pytest.approx(scores[:, 0].mean())   # mean of class 0 comes first
    assert z[1] == pytest.approx(scores[:, 0].var())    # then its variance
    padded = score_moments(scores, n_classes=5)
    assert padded.shape == (20,) and np.all(padded[12:] == 0)


def test_score_moments_constant_column_is_finite():
    scores = np.column_stack([np.ones(50), np.zeros(50)])
    assert np.all(np.isfinite(score_moments(scores)))


def test_simulated_scores_lie_on_the_simplex():
    scores, labels = simulate_scores(150, [0.2, 0.3, 0.5], 0.4, random_state=0)
    assert scores.shape == (150, 3)
    assert np.allclose(scores.sum(axis=1), 1.0)
    assert np.bincount(labels).tolist() == [30, 45, 75]


def test_merging_factor_controls_overlap():
    sep, lab_sep = simulate_scores(3000, [1 / 3] * 3, 0.0, random_state=0)
    mix, lab_mix = simulate_scores(3000, [1 / 3] * 3, 1.0, random_state=0)
    acc_sep = (sep.argmax(1) == lab_sep).mean()
    acc_mix = (mix.argmax(1) == lab_mix).mean()
    assert acc_sep > 0.95 and acc_mix < 0.5


def test_synthetic_curves_layout_and_seed():
    a = synthetic_curves(4, n_prevalences=2, n_merging=3, n_replicates=2, sample_size=30, random_state=1)
    b = synthetic_curves(4, n_prevalences=2, n_merging=3, n_replicates=2, sample_size=30, random_state=1)
    assert len(a) == 6 and all(len(v) == 2 for v in a.values())
    key = next(iter(a))
    assert np.allclose(a[key][0], b[key][0])
    Z, P = curves_to_training_set(a, 4)
    assert Z.shape == (12, 16) and P.shape == (12, 4)


def test_normalize_prevalence_edge_cases():
    assert np.allclose(normalize_prevalence([0, 0, 0], 3), 1 / 3)
    assert np.allclose(normalize_prevalence({0: 2.0, 2: 2.0}, 3), [0.5, 0, 0.5])
    assert np.allclose(normalize_prevalence([np.nan, 1.0], 2), [0, 1])


def test_isotonic_calibrator_outputs_a_distribution():
    rng = np.random.default_rng(0)
    true = rng.dirichlet(np.ones(3), size=100)
    est = np.clip(true + rng.normal(0, 0.05, true.shape), 0, 1)
    cal = IsotonicPrevalenceCalibrator().fit(est, true)
    p = cal.transform(est[0])
    assert p.shape == (3,) and p.sum() == pytest.approx(1.0)


def test_upp_batches():
    y = np.repeat([0, 1, 2], 50)
    batches = list(uniform_prevalence_batches(y, 3, batch_size=40, n_prevalences=4, repeats=3, random_state=0))
    assert len(batches) == 12 and all(len(b) == 40 for b in batches)


@pytest.mark.parametrize("calibrate", [True, False])
def test_smq_fit_predict(calibrate):
    X, y = make_classification(n_samples=600, n_features=10, n_informative=6,
                               n_classes=4, n_clusters_per_class=1, random_state=0)
    q = small_smq(calibrate=calibrate).fit(X[:400], y[:400])
    p = q.predict(X[400:])
    assert p.shape == (4,) and p.sum() == pytest.approx(1.0) and np.all(p >= 0)
    assert (q.calibrator_ is not None) == calibrate


def test_smq_keeps_label_order():
    X, y = make_classification(n_samples=400, n_features=8, n_informative=5,
                               n_classes=3, n_clusters_per_class=1, random_state=1)
    labels = np.array(["cat", "dog", "owl"])[y]
    q = small_smq().fit(X, labels)
    assert list(q.predict_dict(X)) == ["cat", "dog", "owl"]

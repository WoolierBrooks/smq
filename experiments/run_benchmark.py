"""Benchmark of SMQ against 11 quantifiers on multiclass datasets.

Cleaned, configurable port of the script that produced the results in the
paper (``experiments/original/exp_040_original.py``). The protocol, the
baselines and every hyper-parameter are unchanged:

* features standardised with the training statistics;
* base classifier: ``RandomForestClassifier(n_estimators=300)`` everywhere;
* SMQ regressor: ``RandomForestRegressor(n_estimators=300, min_samples_leaf=2)``
  trained once per number of classes on the synthetic curves;
* SMQ-ISO calibration: out-of-fold scores (5-fold) on the training set,
  250 UPP batches of 500 instances (10 prevalences x 25 repetitions);
* evaluation: 1000 UPP batches of 500 instances (20 prevalences x 50
  repetitions) drawn from the test set; one row per batch and method with the
  mean absolute error (MAE) between true and estimated prevalences.

Usage
-----
    python experiments/run_benchmark.py \
        --synthetic-dir data/synthetic \
        --local-dir data/multiclass \
        --output results/raw/my_run.csv

Synthetic curves are read from ``<synthetic-dir>/moss_d500_lite_<C>.pkl``; when
a file is missing it is generated with :func:`smq.synthetic_curves` (pass
``--no-generate`` to skip SMQ for that number of classes instead, as the
original script did).
"""

from __future__ import annotations

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from smq import IsotonicPrevalenceCalibrator, score_moments, synthetic_curves
from smq.synthetic import curves_to_training_set
from smq.utils import normalize_prevalence

warnings.filterwarnings("ignore")

SEED = 42

# 24 UCI datasets shipped with QuaPy (train/test split defined by QuaPy)
QUAPY_DATASETS = [
    "dry-bean", "wine-quality", "academic-success", "digits", "letter",
    "abalone", "obesity", "nursery", "yeast", "hand_digits", "satellite",
    "shuttle", "cmc", "isolet", "waveform-v1", "molecular", "poker_hand",
    "connect-4", "mhr", "chess", "page_block", "phishing", "image_seg", "hcv",
]

# Local CSV files with 10-50 classes (stratified 70/30 split): (file, target)
LOCAL_DATASETS = [
    ("!dataset_372_internet_usage.csv", "class"),
    ("Avila.csv", "V11"),
    ("Chessgame.csv", "game"),
    ("dataset_1457_amazon-commerce-reviews.csv", "class"),
    ("dataset_313_spectrometer.csv", "class"),
    ("dataset_44478_amazon-commerce-reviews_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True.csv", "class"),
    ("dataset_44479_amazon-commerce-reviews_seed_1_nrows_2000_nclasses_10_ncols_100_stratify_True.csv", "class"),
    ("dataset_44480_amazon-commerce-reviews_seed_2_nrows_2000_nclasses_10_ncols_100_stratify_True.csv", "class"),
    ("dataset_44481_amazon-commerce-reviews_seed_3_nrows_2000_nclasses_10_ncols_100_stratify_True.csv", "class"),
    ("dataset_44482_amazon-commerce-reviews_seed_4_nrows_2000_nclasses_10_ncols_100_stratify_True.csv", "class"),
    ("digits.csv", "class"),
    ("fashion-mnist.csv", "class"),
    ("fashion-mnist_train.csv", "label"),
    ("fashion-mnist_test.csv", "label"),
    ("letter.csv", "class"),
    ("Mfeat.csv", "class"),
    ("Walking.csv", "Class"),
]


def rf(seed=SEED):
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)


# ---------------------------------------------------------------- datasets
def load_quapy(name):
    import quapy as qp

    dataset = qp.datasets.fetch_UCIMulticlassDataset(name, min_test_split=0.5, verbose=False)
    train, test = dataset.train_test
    return (
        train.instances,
        pd.factorize(train.labels)[0],
        test.instances,
        pd.factorize(test.labels)[0],
    )


def load_local(path, target):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise KeyError(f"column {target!r} not found")
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    y = pd.factorize(df[target].values)[0]

    # drop classes with a single example, then re-index 0..C-1
    classes, counts = np.unique(y, return_counts=True)
    mask = np.isin(y, classes[counts >= 2])
    X, y = X.values[mask], pd.factorize(y[mask])[0]

    n_classes = len(np.unique(y))
    if not 10 <= n_classes <= 50:
        raise ValueError(f"{n_classes} classes after filtering (expected 10-50)")
    return train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)


# ------------------------------------------------------------- baselines
def build_baselines():
    from mlquantify.adjust_counting import AC, CC, FM, PAC, PCC
    from mlquantify.likelihood import EMQ as EMQ_MLQ
    from mlquantify.neighbors import PWK, KDEyCS, KDEyHD, KDEyML
    from quapy.method.aggregative import EMQ as EMQ_QUAPY

    return {
        "EMQ_QUAPY": EMQ_QUAPY(rf(), calib=None, exact_train_prev=True),
        "EMQ_BCTS_QUAPY": EMQ_QUAPY(
            rf(), calib="bcts", val_split=0.2, exact_train_prev=True, on_calib_error="backup"
        ),
        "EMQ_MLQ": EMQ_MLQ(learner=rf(), calib_function=None, on_calib_error="backup"),
        "EMQ_BCTS_MLQ": EMQ_MLQ(learner=rf(), calib_function="bcts", on_calib_error="backup"),
        "CC": CC(learner=rf()),
        "PCC": PCC(learner=rf()),
        "AC": AC(learner=rf()),
        "PAC": PAC(learner=rf()),
        "FM": FM(learner=rf()),
        "KDEyML": KDEyML(learner=rf(), bandwidth=0.1),
        "KDEyHD": KDEyHD(learner=rf(), montecarlo_trials=500, random_state=SEED),
        "KDEyCS": KDEyCS(learner=rf(), bandwidth=0.1),
        "PWK": PWK(n_neighbors=10),
    }


# ------------------------------------------------------------------ SMQ
def smq_regressor(n_classes, synthetic_dir, generate, cache):
    """Regressor trained on synthetic curves, cached per number of classes."""
    if n_classes in cache:
        return cache[n_classes]
    path = os.path.join(synthetic_dir, f"moss_d500_lite_{n_classes}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            curves = pickle.load(f)
    elif generate:
        print(f"   generating synthetic curves for C={n_classes}")
        curves = synthetic_curves(n_classes, random_state=SEED + n_classes)
        os.makedirs(synthetic_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(curves, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"   {path} not found: skipping SMQ for C={n_classes}")
        cache[n_classes] = None
        return None

    Z, P = curves_to_training_set(curves, n_classes)
    model = RandomForestRegressor(
        n_estimators=300, n_jobs=-1, random_state=SEED, min_samples_leaf=2
    ).fit(Z, P)
    cache[n_classes] = model
    return model


def smq_predict(model, scores, n_classes):
    feat = score_moments(scores, n_classes)
    return normalize_prevalence(model.predict(feat.reshape(1, -1))[0][:n_classes], n_classes), feat


# ------------------------------------------------------------------ main
def run(args):
    from mlquantify.model_selection import UPP
    from mlquantify.utils import get_prev_from_labels

    entries = [(d, "quapy", None) for d in QUAPY_DATASETS] + [
        (f, "local", (os.path.join(args.local_dir, f), t)) for f, t in LOCAL_DATASETS
    ]
    if args.datasets:
        entries = [e for e in entries if e[0] in args.datasets]

    cache, rows = {}, []
    for name, source, meta in entries:
        print(f"\n== {name} [{source}]")
        try:
            if source == "quapy":
                Xtr, ytr, Xte, yte = load_quapy(name)
            else:
                Xtr, Xte, ytr, yte = load_local(*meta)
        except Exception as exc:  # noqa: BLE001 - report and move on, as in the original
            print(f"   skipped: {exc}")
            continue

        C = len(np.unique(ytr))
        yte = np.array([c if c < C else 0 for c in yte])
        scaler = StandardScaler()
        Xtr, Xte = scaler.fit_transform(Xtr), scaler.transform(Xte)
        print(f"   {C} classes, {len(ytr)} train / {len(yte)} test")

        # baselines
        fitted = {}
        for q_name, q in build_baselines().items():
            if args.methods and q_name not in args.methods:
                continue
            try:
                fitted[q_name] = q.fit(Xtr, ytr)
            except Exception as exc:  # noqa: BLE001
                print(f"   {q_name} failed: {exc}")

        # classifier whose scores SMQ reads at test time
        if "EMQ_BCTS_QUAPY" in fitted:
            clf = fitted["EMQ_BCTS_QUAPY"].classifier
        else:
            clf = rf().fit(Xtr, ytr)

        # SMQ (synthetic only) and its calibrations
        model = smq_regressor(C, args.synthetic_dir, not args.no_generate, cache)
        iso_cal = rf_cal = None
        if model is not None:
            oof = cross_val_predict(rf(), Xtr, ytr, cv=5, method="predict_proba", n_jobs=-1)
            calib = UPP(
                batch_size=args.batch_size,
                n_prevalences=args.calib_prevalences,
                repeats=args.calib_repeats,
                random_state=SEED,
            )
            est, true, feats = [], [], []
            for idx in calib.split(Xtr, ytr):
                p, f = smq_predict(model, oof[idx], C)
                est.append(p)
                feats.append(f)
                true.append(normalize_prevalence(get_prev_from_labels(ytr[idx], classes=np.arange(C)), C))
            est, true, feats = map(np.asarray, (est, true, feats))
            iso_cal = IsotonicPrevalenceCalibrator().fit(est, true)
            # extra variant kept for completeness (not reported in the paper)
            rf_cal = RandomForestRegressor(
                n_estimators=300, n_jobs=-1, random_state=SEED, min_samples_leaf=2
            ).fit(np.hstack([est, feats]), true)

        # evaluation protocol
        protocol = UPP(
            batch_size=args.batch_size,
            n_prevalences=args.test_prevalences,
            repeats=args.test_repeats,
            random_state=SEED,
        )

        def add(method, p_hat, p_true):
            rows.append(
                {"dataset": name, "method": method, "n_classes": C,
                 "mae": float(np.mean(np.abs(p_hat - p_true)))}
            )

        for idx in tqdm(protocol.split(Xte, yte), total=protocol.get_n_combinations(), leave=False):
            p_true = normalize_prevalence(get_prev_from_labels(yte[idx], classes=np.arange(C)), C)
            for q_name, q in fitted.items():
                try:
                    add(q_name, normalize_prevalence(q.predict(Xte[idx]), C), p_true)
                except Exception as exc:  # noqa: BLE001
                    print(f"   {q_name} failed on a batch: {exc}")
            if model is not None:
                p_smq, f = smq_predict(model, clf.predict_proba(Xte[idx]), C)
                add("SMQ", p_smq, p_true)
                add("SMQ_ISO", iso_cal.transform(p_smq), p_true)
                p_rf = rf_cal.predict(np.hstack([p_smq, f]).reshape(1, -1))[0][:C]
                add("SMQ_RF", normalize_prevalence(p_rf, C), p_true)

        pd.DataFrame(rows).to_csv(args.output, index=False)  # partial save per dataset
        print(f"   saved {len(rows)} rows to {args.output}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic-dir", default="data/synthetic", help="folder with moss_d500_lite_<C>.pkl")
    ap.add_argument("--local-dir", default="data/multiclass", help="folder with the local CSV datasets")
    ap.add_argument("--output", default="results/raw/benchmark.csv")
    ap.add_argument("--datasets", nargs="*", help="run only these datasets")
    ap.add_argument("--methods", nargs="*", help="run only these baselines")
    ap.add_argument("--no-generate", action="store_true", help="do not generate missing synthetic curves")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--test-prevalences", type=int, default=20)
    ap.add_argument("--test-repeats", type=int, default=50)
    ap.add_argument("--calib-prevalences", type=int, default=10)
    ap.add_argument("--calib-repeats", type=int, default=25)
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.random.seed(SEED)
    run(args)

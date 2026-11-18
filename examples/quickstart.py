"""Quickstart: estimate class prevalences under prior probability shift.

Run with:  python examples/quickstart.py
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from smq import SMQ, absolute_error, uniform_prevalence_batches
from smq.utils import prevalence_from_labels

# 10-class problem: handwritten digits
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=0
)

# SMQ-ISO (default) and the purely synthetic SMQ
smq_iso = SMQ(random_state=0, n_jobs=-1).fit(X_train, y_train)
smq_raw = SMQ(calibrate=False, random_state=0, n_jobs=-1).fit(X_train, y_train)

# Test samples whose class proportions differ from the training set
errors = {"CC": [], "SMQ": [], "SMQ-ISO": []}
for idx in uniform_prevalence_batches(
    y_test, n_classes=10, batch_size=300, n_prevalences=20, repeats=5, random_state=1
):
    true = prevalence_from_labels(y_test[idx], 10)
    # Classify and Count, for reference
    cc = np.bincount(smq_iso.classifier_.predict(X_test[idx]), minlength=10) / len(idx)
    errors["CC"].append(absolute_error(true, cc))
    errors["SMQ"].append(absolute_error(true, smq_raw.predict(X_test[idx])))
    errors["SMQ-ISO"].append(absolute_error(true, smq_iso.predict(X_test[idx])))

print("Mean absolute error over 100 shifted test samples")
for name, values in errors.items():
    print(f"  {name:8s} {np.mean(values):.4f}")

sample = X_test[:300]
print("\nEstimated prevalences of one sample:")
for label, p in smq_iso.predict_dict(sample).items():
    print(f"  digit {label}: {p:.3f}")

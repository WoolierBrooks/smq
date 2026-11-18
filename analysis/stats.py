"""Rank statistics used in the paper: Friedman test and Nemenyi post-hoc.

Follows Demšar (2006), "Statistical comparisons of classifiers over multiple
data sets", JMLR 7:1-30.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as f_dist, studentized_range


def rank_table(mae: pd.DataFrame) -> pd.DataFrame:
    """Rank methods inside each dataset (1 = lowest MAE, ties averaged)."""
    return mae.rank(axis=1, method="average", ascending=True)


def friedman(mae: pd.DataFrame) -> dict:
    """Friedman chi-square and the Iman-Davenport F correction."""
    n, k = mae.shape
    avg = rank_table(mae).mean()
    chi = 12 * n / (k * (k + 1)) * (np.sum(avg**2) - k * (k + 1) ** 2 / 4)
    ff = (n - 1) * chi / (n * (k - 1) - chi)
    return {
        "n_datasets": n,
        "n_methods": k,
        "chi2": float(chi),
        "chi2_p": float(chi2.sf(chi, k - 1)),
        "iman_davenport_F": float(ff),
        "iman_davenport_p": float(f_dist.sf(ff, k - 1, (k - 1) * (n - 1))),
    }


def nemenyi_cd(n_methods: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Critical difference of the Nemenyi test."""
    q = studentized_range.ppf(1 - alpha, n_methods, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets)))


def cliques(avg_ranks: pd.Series, cd: float) -> list[list[str]]:
    """Maximal groups of methods whose rank range is below the CD."""
    s = avg_ranks.sort_values()
    names, values = list(s.index), s.values
    groups = []
    for i in range(len(values)):
        j = i
        while j + 1 < len(values) and values[j + 1] - values[i] < cd:
            j += 1
        if j > i:
            group = names[i : j + 1]
            if not any(set(group) <= set(g) for g in groups):
                groups.append(group)
    return groups

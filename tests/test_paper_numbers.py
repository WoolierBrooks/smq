"""The numbers reported in the paper, recomputed from the published raw results."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))

pd = pytest.importorskip("pandas")
from make_figures import DEFAULT_INPUT, load, mae_table  # noqa: E402
from stats import nemenyi_cd, rank_table  # noqa: E402

# Fig. 4 of the paper: average ranks on the datasets with 11 to 50 classes
PAPER_RANKS_11_50 = {
    "SMQ_ISO": 4.20, "PCC": 4.53, "CC": 5.13, "SMQ": 5.47, "KDEyCS": 6.27,
    "EMQ_BCTS": 6.27, "PWK": 6.53, "EMQ": 7.87, "AC": 8.53, "KDEyML": 8.67,
    "KDEyHD": 8.80, "PAC": 9.20, "FM": 9.53,
}


@pytest.fixture(scope="module")
def tables():
    mae, n_classes, excluded = mae_table(load(DEFAULT_INPUT))
    return mae, n_classes, excluded


def test_benchmark_size(tables):
    mae, n_classes, excluded = tables
    assert mae.shape == (40, 13)
    assert excluded == ["dataset_313_spectrometer.csv"]
    assert (n_classes.loc[mae.index] >= 11).sum() == 15


def test_average_ranks_from_11_classes(tables):
    mae, n_classes, _ = tables
    avg = rank_table(mae[n_classes.loc[mae.index] >= 11]).mean()
    for method, rank in PAPER_RANKS_11_50.items():
        assert avg[method] == pytest.approx(rank, abs=0.005), method


def test_critical_difference():
    assert nemenyi_cd(13, 15) == pytest.approx(4.711, abs=5e-4)


def test_calibration_effect_on_median_mae(tables):
    mae, _, _ = tables
    assert mae["SMQ"].median() == pytest.approx(0.0639, abs=5e-5)
    assert mae["SMQ_ISO"].median() == pytest.approx(0.0611, abs=5e-5)

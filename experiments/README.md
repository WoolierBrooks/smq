# Experiments

| File | What it does |
|---|---|
| `run_benchmark.py` | Benchmark of the paper: SMQ, SMQ-ISO and 11 baselines on 41 multiclass datasets (40 reported). |
| `generate_synthetic.py` | Synthetic training curves, one `moss_d500_lite_<C>.pkl` per number of classes. |
| `datasets.csv` | Datasets, source, number of classes and target column. |
| `original/exp_040_original.py` | The script that produced the published results, verbatim, for provenance. |

## Protocol

Unchanged from the original script; only paths became command-line options.

- **Splits.** The 24 UCI datasets use the train/test split defined by QuaPy (`fetch_UCIMulticlassDataset(..., min_test_split=0.5)`). The local CSV files use a stratified 70/30 split (`random_state=42`); classes with a single example are dropped and only files with 10 to 50 classes are kept. Features are standardised with training statistics.
- **Classifier.** `RandomForestClassifier(n_estimators=300, random_state=42)` for every aggregative method. SMQ reads the test scores of the classifier fitted inside EMQ-BCTS (QuaPy).
- **Synthetic curves.** For each number of classes C: 15 prevalence vectors from Dirichlet(1_C) x 15 merging factors in `linspace(0, 1, 15)` x 20 samples of 150 instances (`src/smq/synthetic.py`). In the published run they were generated once, without a fixed seed, by the notebook `experiments/exp_010/note.ipynb` of the thesis repository; `generate_synthetic.py` produces statistically equivalent files with a seed.
- **SMQ regressor.** `RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42)`, one per number of classes, trained on the 4,500 synthetic samples (4 moments x C features).
- **SMQ-ISO calibration.** 5-fold out-of-fold scores on the training set; 250 UPP batches of 500 instances (10 prevalences x 25 repetitions, `random_state=42`); one isotonic regression per class.
- **Evaluation.** 1000 UPP batches of 500 instances per dataset (20 prevalences x 50 repetitions, `random_state=42`, mlquantify's `UPP`); mean absolute error between true and estimated prevalence vectors.

## Method names in the raw results

`results/raw/exp040_per_batch_mae.csv.gz` keeps the names written by the original script. The analysis maps them as follows; `run_benchmark.py` already writes the new names.

| Raw name | Paper | Note |
|---|---|---|
| `MoSS_BASE_<C>` | SMQ | synthetic only |
| `MoSS_ISO_CALIBRATED_<C>` | SMQ-ISO | isotonic calibration |
| `MoSS_RF_CALIBRATED_<C>` | — | random-forest calibration, not reported |
| `EMQ_MLQ`, `EMQ_BCTS_MLQ` | EMQ, EMQ-BCTS | mlquantify implementations (reported) |
| `EMQ_QUAPY`, `EMQ_BCTS_QUAPY` | — | QuaPy implementations, not reported |

`dataset_313_spectrometer.csv` (35 classes) has no SMQ results, because no synthetic file was generated for 35 classes, and several baselines failed on it; it is excluded so that every method is compared on the same 40 datasets.

## Datasets

24 UCI datasets are downloaded automatically by QuaPy. The 16 local CSV files go in `data/multiclass/` (or pass `--local-dir`); seven come from OpenML, identified by the number in the file name (`dataset_<id>_...`). See `datasets.csv` for the full list.

## Changes from the original script

- paths are options instead of hard-coded server paths;
- the stray character in `LOCAL_DATASETS = [...]w` of the verbatim copy (a syntax error introduced after the run) is gone;
- missing synthetic files are generated on the fly instead of skipping SMQ (`--no-generate` restores the old behaviour);
- outputs use the paper's method names and columns `dataset, method, n_classes, mae`.

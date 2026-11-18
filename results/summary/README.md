# Summary statistics

Mean absolute error (MAE) per dataset is the average over its 1000 test batches.
Ranks are computed per dataset (1 = lowest MAE) and averaged.

## All datasets (3–50 classes)

40 datasets, 13 methods. Friedman χ² = 46.67 (p = 5.32e-06); Iman–Davenport F = 4.20 (p = 2.69e-06); Nemenyi CD = 2.885 (α = 0.05).

| Method | Avg. rank | Median rank | Median MAE | Mean MAE |
|---|---:|---:|---:|---:|
| PCC | 5.03 | 3.5 | 0.0539 | 0.0867 |
| PWK | 5.50 | 5.0 | 0.0444 | 0.0822 |
| CC | 5.96 | 5.0 | 0.0609 | 0.0928 |
| EMQ-BCTS | 6.47 | 6.0 | 0.0545 | 0.0919 |
| KDEy-ML | 6.60 | 6.0 | 0.0489 | 0.0980 |
| KDEy-CS | 6.60 | 6.0 | 0.0494 | 0.0906 |
| **SMQ-ISO** | 6.90 | 7.5 | 0.0611 | 0.0977 |
| KDEy-HD | 7.05 | 8.0 | 0.0494 | 0.0907 |
| **SMQ** | 7.45 | 8.0 | 0.0639 | 0.0979 |
| AC | 7.76 | 8.0 | 0.0602 | 0.0970 |
| EMQ | 7.97 | 7.0 | 0.0515 | 0.0931 |
| PAC | 8.05 | 9.0 | 0.0602 | 0.0968 |
| FM | 9.65 | 10.5 | 0.0576 | 0.1139 |

Not significantly different from the best (PCC): PCC, PWK, CC, EMQ-BCTS, KDEy-ML, KDEy-CS, SMQ-ISO, KDEy-HD, SMQ, AC.

## Datasets with ≥ 11 classes

15 datasets, 13 methods. Friedman χ² = 40.98 (p = 4.95e-05); Iman–Davenport F = 4.13 (p = 1.18e-05); Nemenyi CD = 4.711 (α = 0.05).

| Method | Avg. rank | Median rank | Median MAE | Mean MAE |
|---|---:|---:|---:|---:|
| **SMQ-ISO** | 4.20 | 2.0 | 0.0252 | 0.0275 |
| PCC | 4.53 | 4.0 | 0.0154 | 0.0258 |
| CC | 5.13 | 4.0 | 0.0170 | 0.0272 |
| **SMQ** | 5.47 | 4.0 | 0.0283 | 0.0297 |
| EMQ-BCTS | 6.27 | 6.0 | 0.0183 | 0.0270 |
| KDEy-CS | 6.27 | 6.0 | 0.0198 | 0.0266 |
| PWK | 6.53 | 7.0 | 0.0214 | 0.0281 |
| EMQ | 7.87 | 8.0 | 0.0248 | 0.0287 |
| AC | 8.53 | 9.0 | 0.0255 | 0.0319 |
| KDEy-ML | 8.67 | 9.0 | 0.0362 | 0.0351 |
| KDEy-HD | 8.80 | 9.0 | 0.0245 | 0.0302 |
| PAC | 9.20 | 11.0 | 0.0251 | 0.0330 |
| FM | 9.53 | 10.0 | 0.0292 | 0.0388 |

Not significantly different from the best (SMQ-ISO): SMQ-ISO, PCC, CC, SMQ, EMQ-BCTS, KDEy-CS, PWK, EMQ, AC, KDEy-ML, KDEy-HD.

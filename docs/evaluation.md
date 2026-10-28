# Evaluation

Each synthesizer is scored on three things:

| Axis | Measures |
|---|---|
| Fidelity | per-column KS and total-variation distance, error in pairwise dependence, and a discriminator that tries to tell real rows from synthetic ones (AUC 0.5 means it can't) |
| Utility | a fraud model trained on synthetic data and tested on real data, compared with one trained on real data |
| Privacy | distance to the closest training record against a holdout baseline, exact copies, and a membership-inference attack (AUC 0.5 means no leak), run on all rows and on fraud rows alone |

## Simulated transactions

`python -m synthfraud benchmark`: 30,000 transactions, 2% fraud across four fraud types. The synthesizers see 60% of the rows; 20% is a privacy holdout and 20% is the real test set. `results/report.html` has the full plots.

| Synthesizer | Formal privacy | Discriminator AUC (lower is better) | Fraud PR-AUC trained on synthetic (real data: 0.998) | Membership AUC, all rows / fraud rows (lower is better) |
|---|---|---|---|---|
| Gaussian copula | none | 0.717 | 0.026 | 0.500 / 0.548 |
| Conditional copula | none | 0.840 | 0.994 | 0.497 / 0.565 |
| Conditional copula | ε = 8 | 0.838 | 0.936 | 0.500 / 0.561 |
| Conditional copula | ε = 2 | 0.852 | 0.654 | 0.500 / 0.567 |
| Conditional copula | ε = 1 | 0.859 | 0.459 | 0.500 / 0.546 |
| TVAE | none | 0.776 | 0.983 | 0.508 / 0.626 |
| TVAE with DP-SGD | ε = 8 | 0.896 | 0.023 | 0.502 / 0.553 |

- A single global copula reproduces every column well, but a fraud model trained on its output is useless (PR-AUC 0.026). Fraud shows up in combinations of values, and one correlation matrix averages them away.
- Fitting one copula per fraud type fixes this: its data trains a fraud model almost as good as real data (0.994 against 0.998).
- TVAE looks the most realistic to the discriminator, but the membership attack on fraud rows rises to 0.626. It is memorising the rare victims, which a realism score alone would hide.
- Differential privacy costs utility predictably: at ε = 8 the conditional copula keeps 94% of real-data performance, at ε = 2 it keeps 65% and at ε = 1 it keeps 46%.
- DP-SGD on the neural model loses the fraud classes entirely at this data size.

With only a few hundred fraud rows, a fraud-row membership AUC of about 0.55 is at the noise floor; compare the differences between models rather than the absolute values.

## Real card transactions

`python -m synthfraud benchmark --ulb creditcard.pq --out results/real_ulb` runs every synthesizer on the [ULB / Worldline card-fraud data](https://api.openml.org/d/1597): 284,807 real transactions with 492 frauds. Results go to `results/real_ulb/results.json`. The HTML report is regenerated locally rather than committed, because it holds 10 MB of plot data.

This data has no fraud types, so the conditional copula conditions only on fraud against legitimate. The privacy bins are fixed in advance rather than read from the data: 48 hours of time, amounts up to €30,000 and the anonymised components within ±120 on an asinh scale. 60,000 synthetic rows are drawn per model.

| Synthesizer | Formal privacy | Discriminator AUC | Fraud PR-AUC trained on synthetic (real data: 0.766) | Membership AUC, all rows / fraud rows |
|---|---|---|---|---|
| Gaussian copula | none | 0.999 | 0.002 | 0.504 / 0.522 |
| Conditional copula | none | 0.997 | 0.679 | 0.504 / 0.535 |
| Conditional copula | ε = 8 | 0.996 | 0.750 | 0.502 / 0.524 |
| Conditional copula | ε = 2 | 0.997 | 0.675 | 0.506 / 0.519 |
| TVAE | none | 0.993 | 0.697 | 0.504 / 0.531 |
| TVAE with DP-SGD | ε = 8 | 0.999 | 0.002 | 0.505 / 0.521 |

The ranking holds on real data: a single copula loses the fraud, conditioning on the label keeps most of it, and the private conditional copula still trains a useful fraud model. At ε = 8 it matched the non-private model, a difference within the noise of about 295 training frauds. DP-SGD on the neural model fails again.

Two things differ from the simulated data. Every model is easy to tell apart from real rows (discriminator AUC ≥ 0.99), because 30 heavy-tailed continuous columns are much harder to reproduce exactly; the synthetic data is useful for training fraud models, not for inspecting individual records. And no model leaked membership measurably (all AUCs within 0.04 of 0.5).

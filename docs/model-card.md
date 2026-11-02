# Model card: synthetic fraud data generators

Three synthesizers: a Gaussian copula baseline, a conditional copula (one per fraud type, optionally
differentially private) and a TVAE (optionally DP-SGD). Numbers come from `docs/evaluation.md` and
`results/`.

## Intended use

Producing synthetic transaction tables that keep fraud patterns, for training and testing fraud
models, sharing with vendors and research partners, and demos. Not for inspecting individual
records, and not as a privacy guarantee unless a differentially private model is used.

## Evaluation

**Simulated transactions** (30,000 rows, 2% fraud in four types):

| Synthesizer | ε | Fraud PR-AUC trained on synthetic (real: 0.998) | Membership AUC on fraud rows |
|---|---|---|---|
| Gaussian copula | none | 0.026 | 0.548 |
| Conditional copula | none | 0.994 | 0.565 |
| Conditional copula | 8 | 0.936 | 0.561 |
| Conditional copula | 2 | 0.654 | 0.567 |
| Conditional copula | 1 | 0.459 | 0.546 |
| TVAE | none | 0.983 | **0.626** (memorises rare rows) |
| TVAE with DP-SGD | 8 | 0.023 | 0.553 |

**Real card transactions** (ULB / Worldline, 284,807 rows, 492 frauds; fraud vs legitimate only):

| Synthesizer | ε | Fraud PR-AUC trained on synthetic (real: 0.766) | Discriminator AUC |
|---|---|---|---|
| Gaussian copula | none | 0.002 | 0.999 |
| Conditional copula | none | 0.679 | 0.997 |
| Conditional copula | 8 | 0.750 | 0.996 |
| Conditional copula | 2 | 0.675 | 0.997 |
| TVAE | none | 0.697 | 0.993 |
| TVAE with DP-SGD | 8 | 0.002 | 0.999 |

No model leaked membership measurably on the real data (all AUCs within 0.04 of 0.5).

## Known failure modes

- Every model is easy to tell from real rows on the real data (discriminator AUC ≥ 0.99).
- DP-SGD on the neural model loses the fraud classes at these data sizes.
- With a few hundred fraud rows, membership AUCs around 0.55 are at the noise floor; compare models,
  not absolute values.
- Formal guarantees cover only the DP models, and only if bounds and categories come from outside the
  private table.

## Before relying on it

Fit on the bank's own data in its controlled environment, run `evaluate` (fidelity, utility,
membership attacks) on a holdout, and choose ε with the privacy office.

# Synthetic Fraud Data

Generate transaction data that fraud teams, vendors and researchers can use without seeing a single real customer. The synthetic data keeps the fraud patterns that make it worth having, and it comes with measurements of how faithful, useful and private it actually is.

Fraud data is the hardest data in a bank to share: it is personal, confidential, and the interesting part (fraud) is under 2% of rows. Naive synthetic generators either lose those rare patterns, so models trained on them are useless, or memorise them, so the rare victims are exactly who leaks. This project measures both failure modes and provides a generator that avoids them.

## Synthesizers

| Model | How it works | Privacy |
|---|---|---|
| **Gaussian copula** | Exact marginals plus one global correlation matrix | none (baseline) |
| **Conditional copula** | One copula **per fraud type**, so card testing, account takeover, UPI scams and skimming each keep their own joint pattern | optional **differential privacy** |
| **TVAE** | Conditional variational autoencoder with mode-specific normalisation (CTGAN/TVAE paper) and learned per-column output noise, conditioned on fraud type and with rare types oversampled | optional DP-SGD |

**How the DP conditional copula works.** Every person belongs to exactly one fraud-type class, so classes are disjoint and the privacy budget applies to each in parallel. Within a class it releases three things with the Gaussian mechanism:
- marginals, as histograms over **public bins** (domain bounds from product knowledge, not from the data)
- the copula correlation, from a noisy sum of clipped normal scores
- class sizes

Accounting is in zero-concentrated DP and converted to (ε, δ = 1e-5). DP-SGD for the TVAE uses per-example clipping via `torch.func`, Poisson sampling and a Rényi-DP accountant.

## How it is scored

| Axis | Measures |
|---|---|
| **Fidelity** | Per-column KS / total-variation distance, error in pairwise dependence, and a **discriminator** that tries to tell real rows from synthetic (AUC 0.5 = indistinguishable) |
| **Utility** | **Train on synthetic, test on real (TSTR)** for the task the data exists for, fraud detection, compared with training on real data |
| **Privacy** | Distance to the closest training record against a holdout baseline, exact copies, and a **membership-inference attack** (AUC 0.5 = no leakage), also run separately on **fraud rows**, the rare records most at risk |

## Results

`python -m synthfraud benchmark`: 30,000 transactions, 2% fraud across four fraud types. The synthesizers see 60% of the rows; 20% is a privacy holdout and 20% is the real test set for utility.

| Synthesizer | Formal privacy | Discriminator AUC ↓ | Fraud PR-AUC, trained on synthetic (real = 0.998) | Membership AUC, all rows / fraud rows ↓ |
|---|---|---|---|---|
| Gaussian copula | none | 0.717 | **0.026** | 0.500 / 0.548 |
| Conditional copula | none | 0.840 | **0.994** | 0.497 / 0.565 |
| Conditional copula | **ε = 8** | 0.838 | **0.936** | 0.500 / 0.561 |
| Conditional copula | **ε = 2** | 0.852 | 0.654 | 0.500 / 0.567 |
| Conditional copula | **ε = 1** | 0.859 | 0.459 | 0.500 / 0.546 |
| TVAE | none | **0.776** | 0.983 | 0.508 / **0.626** |
| TVAE (DP-SGD) | ε = 8 | 0.896 | 0.023 | 0.502 / 0.553 |

What the numbers say:

1. **A single global model throws away the fraud.** The plain copula reproduces every column almost perfectly, yet a fraud model trained on its output is useless (PR-AUC 0.026). Fraud lives in *combinations* of values, and one correlation matrix averages them away.
2. **Conditioning on fraud type fixes it.** The conditional copula's synthetic data trains a fraud model almost as good as real data (0.994 vs 0.998).
3. **The neural model looks the most realistic, and it memorises the rare victims.** TVAE has the best discriminator score, but the membership attack on fraud rows rises to 0.626. That is exactly the leak this project exists to catch, and a "looks realistic" score alone would hide it.
4. **Differential privacy costs utility in a predictable way.** At ε = 8 the conditional copula keeps 94% of real-data fraud-detection performance with a formal guarantee. At ε = 2 it keeps 65%, at ε = 1 it keeps 46%.
5. **DP-SGD on a neural model is the wrong tool for 2%-rare classes at this data size.** With oversampling disabled (it would break the accounting), the noise drowns the fraud classes. The DP conditional copula is the practical choice.

At about 0.55 the fraud-row membership AUC is at the noise floor for a few hundred fraud rows. Look at the *gap* between models, not the absolute value. Open `results/report.html` for the full plots.

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q                                          # 8 tests
python -m synthfraud benchmark                               # all synthesizers -> results/report.html
python -m synthfraud fit --model ccopula --epsilon 2 --data private.csv --out model.joblib
python -m synthfraud sample model.joblib -n 100000 --out synthetic.csv
python -m synthfraud sample model.joblib -n 5000 --fraud-type upi_scam --out upi_scams.csv
python -m synthfraud evaluate --real private.csv --synthetic synthetic.csv
streamlit run app.py                                         # generate, inspect, download
```

The studio lets you upload your own table (or use the demo data), pick a synthesizer and privacy budget, and see the scorecard, distribution plots, fraud-type mix and privacy detail. You can then download the CSV.

**Your own data.** Match the schema in `synthfraud/source.py` (or edit `SCHEMA`), and set the public bounds in `BOUNDS` and category domains in `CATEGORIES` from product knowledge. Do not derive them from the private table itself, or the DP guarantee no longer covers them.

## Choosing ε

- **ε ≈ 8:** close to non-private utility. A reasonable choice for sharing inside a regulated group.
- **ε ≈ 1–2:** for vendors and research partners. Fraud patterns survive at reduced strength.
- **Without DP:** however good the scores look, there is no formal guarantee. The fraud-row membership attack shows why that matters.

## Layout

```
synthfraud/
  source.py     schema, demo data generator, public bounds
  transform.py  mode-specific normalisation (Bayesian GMM) + one-hot, reversible
  models.py     Gaussian copula, conditional copula (+DP), TVAE (+DP-SGD), privacy accounting
  evaluate.py   fidelity, utility (TSTR), privacy (DCR, membership inference)
  pipeline.py   split, assess, HTML report
app.py          Streamlit studio
results/        last benchmark (report.html, results.json)
```

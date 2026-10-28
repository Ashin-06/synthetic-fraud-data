# Synthetic Fraud Data

Generates transaction data that fraud teams, vendors and researchers can use without seeing real customers. The synthetic data keeps the fraud patterns that make it useful, and every run reports how faithful, useful and private it is.

Fraud data is hard to share: it is personal and confidential, and the interesting part, the fraud, is under 2% of rows. Simple generators either lose those rare patterns, so models trained on the output are useless, or memorise them, so the rare victims are the ones who leak. This project measures both problems and includes a generator that avoids them.

## Synthesizers

| Model | How it works | Privacy |
|---|---|---|
| Gaussian copula | exact column distributions plus one correlation matrix | none (baseline) |
| Conditional copula | one copula per fraud type, so card testing, account takeover, UPI scams and skimming each keep their own pattern | optional differential privacy |
| TVAE | a conditional variational autoencoder, conditioned on fraud type, with rare types oversampled | optional DP-SGD |

The private conditional copula releases each fraud type's column histograms (over bins fixed in advance), its correlation matrix and its size with Gaussian noise, and reports the total privacy cost as (ε, δ = 1e-5).

Each run is scored on fidelity (can a classifier tell real from synthetic?), utility (does a fraud model trained on synthetic data work on real data?) and privacy (can an attacker tell whether a record was in the training data, including for fraud rows specifically?).

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q
python -m synthfraud benchmark                               # all synthesizers, report in results/report.html
python -m synthfraud fit --model ccopula --epsilon 2 --data private.csv --out model.joblib
python -m synthfraud sample model.joblib -n 100000 --out synthetic.csv
python -m synthfraud sample model.joblib -n 5000 --fraud-type upi_scam --out upi_scams.csv
python -m synthfraud evaluate --real private.csv --synthetic synthetic.csv
streamlit run app.py                                         # generate, inspect and download
```

In the Streamlit studio you can upload a table or use the demo data, pick a synthesizer and privacy budget, check the scorecard and plots, and download the CSV.

Model files are pickles, which can run code when loaded. Only load model files you created yourself or received over a trusted channel.

## Using your own data

Match the schema in `synthfraud/source.py` (or edit `SCHEMA`), and set the bounds in `BOUNDS` and category values in `CATEGORIES` from product knowledge. Don't derive them from the private table itself, or the privacy guarantee no longer covers them.

## Results

| Test | Result |
|---|---|
| Simulated data, fraud model trained on synthetic (real data gives 0.998) | Gaussian copula 0.026, conditional copula 0.994, with ε = 8 0.936 |
| Simulated data, membership attack on fraud rows | TVAE 0.626 (memorising rare rows); copulas about 0.55 |
| Real card transactions (ULB), fraud model trained on synthetic (real data gives 0.766) | conditional copula with ε = 8: 0.750 |
| Real card transactions, membership attack | no measurable leak for any model |

Full tables are in [docs/evaluation.md](docs/evaluation.md).

## Choosing ε

- About 8: close to non-private utility; reasonable for sharing inside a regulated group.
- About 1–2: for vendors and research partners; fraud patterns survive at reduced strength.
- No differential privacy: no formal guarantee, however good the scores look.

## Project layout

```
synthfraud/
  source.py     schema, demo data, public bounds, loader for the ULB data
  transform.py  reversible column encoding
  models.py     copulas, TVAE and privacy accounting
  evaluate.py   fidelity, utility and privacy scores
  pipeline.py   data split, scoring and HTML report
app.py          Streamlit studio
```

## Limitations

- Formal privacy guarantees only apply to the differentially private models.
- On the real dataset every model is easy to tell apart from real rows, so the synthetic data suits training fraud models rather than inspecting individual records.
- DP-SGD on the neural model doesn't work for classes this rare at this data size.
- There is no API service yet; it runs as a CLI and a dashboard.

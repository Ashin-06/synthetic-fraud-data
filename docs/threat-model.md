# Threat model

What the synthetic data service protects, from whom, and what is still open. Written for the 1.0.0
code; nobody independent has reviewed it.

## What it protects

| Asset | Where it lives |
|---|---|
| The private training table | only where `fit` runs; never sent to the API |
| Fitted models (histograms, correlations, network weights) | signed `.joblib` files in `SYNTHFRAUD_MODEL_DIR` |
| The model signing key | `SYNTHFRAUD_MODEL_KEY` |
| Who generated what | `generations` table |

A fitted model is derived from private data. A differentially private model's release is covered by
its (ε, δ); a non-private model's is not, and it can leak rare rows.

## Threats and what stops them

| Threat | Mitigation | Residual risk |
|---|---|---|
| A malicious model file runs code when loaded (pickle) | Only files whose HMAC matches the model key are loaded, from the verified bytes | Anyone with the model key can sign a file; keep it with whoever runs `fit` |
| Synthetic rows reveal a real customer | DP synthesizers with reported ε; membership attacks in every evaluation | Non-private models have no guarantee; TVAE memorises rare fraud rows (membership AUC 0.626 on simulated data) |
| Repeated sampling erodes privacy | — | Sampling from a DP model is post-processing and costs no further budget; the budget was spent at `fit` |
| Bounds derived from the private data break the guarantee | Documented: set bounds and categories from product knowledge | Not checked by code |
| Unauthorised sampling | API keys with roles; every request logged with the caller | — |
| Flooding | 100,000 rows per request, per-client rate limit, 64 KB body cap | Rate limit is per process |

## Not covered

- Where and how `fit` runs on private data: that is the bank's controlled environment.
- TLS (terminate it at the ingress).

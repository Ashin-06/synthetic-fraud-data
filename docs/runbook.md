# Runbook

How to fit, sign, deploy and operate the sampling service. Commands assume the repository root.

## Fit and sign a model (inside the bank's controlled environment)

```bash
export SYNTHFRAUD_MODEL_KEY=$(python -c "import secrets,base64;print(base64.b64encode(secrets.token_bytes(32)).decode())")
python -m synthfraud fit --model ccopula --epsilon 2 --data private.csv --out models/ccopula-eps2.joblib
# -> models/ccopula-eps2.joblib and models/ccopula-eps2.joblib.sig
```

Set the bounds and categories in `synthfraud/source.py` from product knowledge, not from the
private table, or the privacy guarantee does not cover them.

## Deploy

```bash
docker build -t synthetic-fraud-data:1.0.0 .
python -m synthfraud.ops keygen              # API keys per consumer team and monitor
docker run -d --name synthfraud -p 8009:8009 -v synthfraud-data:/data -v $PWD/models:/models:ro \
  -e SYNTHFRAUD_KEYS_FILE=/run/secrets/synthfraud-keys.json -e SYNTHFRAUD_MODEL_KEY=... \
  synthetic-fraud-data:1.0.0
curl -fsS http://127.0.0.1:8009/readyz
```

The image runs in production mode as a non-root user. It refuses to start, and logs why, without
keys, the model key, or at least one signed model in `/models`. Unsigned or altered model files are
skipped and logged as `model skipped`.

## Configure

| Variable | Required in production | Meaning |
|---|---|---|
| `SYNTHFRAUD_ENV` | — | `development` (default) or `production` |
| `SYNTHFRAUD_API_KEYS` / `SYNTHFRAUD_KEYS_FILE` | yes | JSON `{"sha256:<hex>": {"user": "...", "role": "..."}}`. Roles: `consumer` (sample, read), `viewer`, `monitor`, `admin` |
| `SYNTHFRAUD_MODEL_DIR` | yes | directory of signed `*.joblib` models |
| `SYNTHFRAUD_MODEL_KEY` | yes | base64, at least 32 bytes; `SYNTHFRAUD_MODEL_KEY_PREVIOUS` for rotation |
| `SYNTHFRAUD_DATABASE_URL` | yes | the generation log: `sqlite:////data/synthfraud.db` or `postgresql://...` |
| `SYNTHFRAUD_HOST`, `SYNTHFRAUD_PORT` | — | default `127.0.0.1:8009` |
| `SYNTHFRAUD_RATE_LIMIT`, `SYNTHFRAUD_MAX_BODY_BYTES` | — | default 120/min per client, 64,000 bytes |

## Watch

- `GET /healthz`; `GET /readyz` (database and at least one model); `GET /metrics` (role `monitor`):
  requests, latency, `errors_total{kind="model_rejected"}`, `synthfraud_decisions_total{decision="rows_generated"}`.

## Rotate keys

**Model key.** Set `SYNTHFRAUD_MODEL_KEY_PREVIOUS` to the old key and `SYNTHFRAUD_MODEL_KEY` to a new
one; re-sign models with the new key (`synthfraud.signing.sign`), restart, then drop `_PREVIOUS`.
**API keys.** Add the new hash, restart, hand over the key, remove the old hash, restart.

## Back up and restore

The models are files you already keep; back up the generation log like any SQLite or Postgres
database (`sqlite3` backup API, or `pg_dump`).

## Incidents

1. **A model leaked private data** (a membership attack or a reviewer finds real records). Remove the
   file from the model directory, restart, look up who sampled it in `/v1/generations`, and follow the
   data-breach procedure.
2. **`model skipped` in the logs.** A file without a valid signature reached the model directory:
   find out how before re-signing anything.

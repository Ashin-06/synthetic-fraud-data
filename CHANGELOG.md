# Changelog

All notable changes to this project. Versions follow [semantic versioning](https://semver.org/).

## 1.0.0 — 2026-11-02

### Added
- A sampling API (`python -m synthfraud serve`): `GET /v1/models`, `POST /v1/sample` (JSON or CSV,
  up to 100,000 rows, optionally one fraud type), `GET /v1/generations`. Models are fitted offline
  with `fit`; private data never passes through the API.
- Signed model files: with `SYNTHFRAUD_MODEL_KEY` set, `fit` writes `<model>.sig` (HMAC-SHA256),
  and the API only unpickles files whose signature matches, from the bytes it verified.
- A generation log (who drew how many rows from which model, at what ε) in SQLite or Postgres,
  numbered migrations and `python -m synthfraud migrate`.
- Settings from `SYNTHFRAUD_*` environment variables, checked at startup. `SYNTHFRAUD_ENV=production`
  refuses to start without hashed API keys, a database, `SYNTHFRAUD_MODEL_KEY` and at least one
  signed model in `SYNTHFRAUD_MODEL_DIR`.
- Per-client API keys with roles (`consumer`, `viewer`, `monitor`, `admin`); rate limit, body cap,
  security headers, CORS only for listed origins; JSON logs with request IDs; `/healthz`, `/readyz`
  and Prometheus `/metrics`.
- `pyproject.toml`, pinned lock files with CPU-only torch from PyTorch's index, a container image
  (slim Python 3.12, non-root, health check), CI with ruff, bandit, pip-audit and coverage.
- `SECURITY.md`, `docs/threat-model.md`, `docs/runbook.md`, `docs/model-card.md`, MIT licence.

### Unchanged
- The synthesizers, the privacy accounting and every number in the README and `docs/evaluation.md`.

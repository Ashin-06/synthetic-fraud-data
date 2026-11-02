"""The sampling API: settings, signed model loading, auth, limits, headers, metrics, health,
the generation log and migrations."""
import base64
import hashlib
import importlib
import json
import os
import subprocess
import sys

import joblib
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from synthfraud import db, ops
from synthfraud.models import ConditionalCopula
from synthfraud.signing import load_verified, sign
from synthfraud.source import generate

MODEL_KEY = b"m" * 32
MK = base64.b64encode(MODEL_KEY).decode()
KEY = "synthfraud_" + "x" * 40
HASHED = {f"sha256:{hashlib.sha256(KEY.encode()).hexdigest()}": {"user": "ds-team", "role": "consumer"}}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    m = ConditionalCopula(dp_epsilon=2).fit(generate(3000))
    joblib.dump(m, d / "ccopula-eps2.joblib")
    sign(d / "ccopula-eps2.joblib", MODEL_KEY)
    joblib.dump(m, d / "unsigned.joblib")  # no .sig: must be skipped
    joblib.dump(m, d / "tampered.joblib")
    sign(d / "tampered.joblib", MODEL_KEY)
    with open(d / "tampered.joblib", "ab") as f:
        f.write(b"\x00")  # changed after signing
    return d


def boot(monkeypatch, **env):
    for k in [k for k in os.environ if k.startswith("SYNTHFRAUD_")]:
        monkeypatch.delenv(k)
    for k, v in env.items():
        monkeypatch.setenv(f"SYNTHFRAUD_{k}", v)
    import synthfraud.api as api

    return importlib.reload(api)


def test_only_signed_models_are_loaded(monkeypatch, model_dir):
    api = boot(monkeypatch, MODEL_DIR=str(model_dir), MODEL_KEY=MK)
    assert set(api.MODELS) == {"ccopula-eps2"}
    with pytest.raises(ValueError, match="no signature"):
        load_verified(model_dir / "unsigned.joblib", [MODEL_KEY])
    with pytest.raises(ValueError, match="not signed"):
        load_verified(model_dir / "tampered.joblib", [MODEL_KEY])
    with pytest.raises(ValueError, match="not signed"):
        load_verified(model_dir / "ccopula-eps2.joblib", [b"k" * 32])  # another key
    assert api.MODELS["ccopula-eps2"]["epsilon"] != "none (no formal privacy guarantee)"


def test_production_refuses_without_models_or_keys(monkeypatch, model_dir, tmp_path):
    prod = {"ENV": "production", "API_KEYS": json.dumps(HASHED), "DATABASE_URL": "sqlite://"}
    with pytest.raises(ops.ConfigError, match="MODEL_KEY"):
        boot(monkeypatch, **prod, MODEL_DIR=str(model_dir))
    with pytest.raises(ops.ConfigError, match="MODEL_DIR"):
        boot(monkeypatch, **prod, MODEL_KEY=MK)
    with pytest.raises(ops.ConfigError, match="no signed models"):
        boot(monkeypatch, **prod, MODEL_KEY=MK, MODEL_DIR=str(tmp_path))
    api = boot(monkeypatch, **prod, MODEL_KEY=MK, MODEL_DIR=str(model_dir))
    assert api.settings.production and set(api.MODELS) == {"ccopula-eps2"}
    with pytest.raises(ops.ConfigError, match="without API keys"):
        ops.load_settings({"SYNTHFRAUD_HOST": "0.0.0.0"})


def test_sampling_is_logged_and_the_log_survives_a_restart(monkeypatch, model_dir, tmp_path):
    url = os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path / 'gen.db'}"
    c = TestClient(boot(monkeypatch, MODEL_DIR=str(model_dir), MODEL_KEY=MK, DATABASE_URL=url).app)
    r = c.post("/v1/sample", json={"model": "ccopula-eps2", "n": 50, "seed": 7}).json()
    assert len(r["rows"]) == 50 and r["epsilon"] == c.get("/v1/models").json()["models"][0]["epsilon"]
    csv = c.post("/v1/sample", json={"model": "ccopula-eps2", "n": 20, "format": "csv", "fraud_type": "upi_scam"})
    assert csv.headers["content-type"].startswith("text/csv") and len(csv.text.strip().splitlines()) == 21
    assert c.post("/v1/sample", json={"model": "nope", "n": 5}).status_code == 404
    assert c.post("/v1/sample", json={"model": "ccopula-eps2", "n": 10**7}).status_code == 422

    c = TestClient(boot(monkeypatch, MODEL_DIR=str(model_dir), MODEL_KEY=MK, DATABASE_URL=url).app)  # restart
    log = c.get("/v1/generations").json()["generations"]
    mine = [g for g in log if g["model"] == "ccopula-eps2"]
    assert {g["rows_requested"] for g in mine} >= {50, 20} and mine[0]["requested_by"] == "demo"
    assert any(g["fraud_type"] == "upi_scam" for g in mine)


def test_roles_and_metrics(monkeypatch, model_dir):
    keys = {"k-ds": {"user": "ds", "role": "consumer"}, "k-view": {"user": "vic", "role": "viewer"},
            "k-mon": {"user": "prom", "role": "monitor"}}
    c = TestClient(boot(monkeypatch, MODEL_DIR=str(model_dir), MODEL_KEY=MK, API_KEYS=json.dumps(keys)).app)
    body = {"model": "ccopula-eps2", "n": 5}
    assert c.post("/v1/sample", json=body).status_code == 401
    assert c.post("/v1/sample", json=body, headers={"x-api-key": "k-view"}).status_code == 403
    assert c.post("/v1/sample", json=body, headers={"x-api-key": "k-ds"}).status_code == 200
    assert c.get("/v1/generations", headers={"x-api-key": "k-view"}).status_code == 200
    m = c.get("/metrics", headers={"x-api-key": "k-mon"})
    assert m.status_code == 200 and 'synthfraud_decisions_total{decision="rows_generated"}' in m.text


def test_limits_headers_errors_and_health(monkeypatch, model_dir):
    api = boot(monkeypatch, MODEL_DIR=str(model_dir), MODEL_KEY=MK, RATE_LIMIT="3")
    c = TestClient(api.app)
    codes = [c.get("/v1/models").status_code for _ in range(5)]
    assert codes[:3] == [200] * 3 and codes[3:] == [429, 429]
    api = boot(monkeypatch, MODEL_DIR=str(model_dir), MODEL_KEY=MK, MAX_BODY_BYTES="200")

    @api.app.get("/boom")
    def boom():
        raise RuntimeError("secret detail: customer 9876543210")

    c = TestClient(api.app)
    assert c.post("/v1/sample", json={"model": "x" * 300}).status_code == 413
    crash = c.get("/boom")
    assert crash.status_code == 500 and "9876543210" not in crash.text and crash.json()["request_id"]
    bad = c.post("/v1/sample", json={"model": "SECRET", "n": "lots"})
    assert bad.status_code == 422 and "lots" not in bad.text
    r = c.get("/v1/models", headers={"x-request-id": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123" and r.headers["content-security-policy"] == ops.API_CSP
    assert c.get("/healthz").json() == {"status": "ok"}
    assert c.get("/readyz").json()["checks"] == {"database": True, "models": True}


def test_fit_signs_when_a_key_is_set(tmp_path):
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("SYNTHFRAUD_")}, "SYNTHFRAUD_MODEL_KEY": MK}
    out = tmp_path / "m.joblib"
    r = subprocess.run([sys.executable, "-m", "synthfraud", "fit", "--model", "copula", "--out", str(out)], env=env,
                       capture_output=True, text=True, timeout=600, check=False)
    assert r.returncode == 0 and (tmp_path / "m.joblib.sig").exists()
    load_verified(out, [MODEL_KEY])


def test_migrations_apply_once(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    eng = db.engine(url)
    assert db.migrate(eng) == ["001_init.sql"] and db.migrate(eng) == []
    with eng.connect() as c:
        tables = {r[0] for r in c.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'"))}
    assert {"generations", "schema_migrations"} <= tables
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("SYNTHFRAUD_")}, "SYNTHFRAUD_DATABASE_URL": url}
    out = subprocess.run([sys.executable, "-m", "synthfraud", "migrate"], env=env, capture_output=True, text=True,
                         timeout=300, check=False)
    assert out.returncode == 0 and "up to date" in out.stdout

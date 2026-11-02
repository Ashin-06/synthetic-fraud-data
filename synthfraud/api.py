"""Sampling service: serves synthetic rows from fitted synthesizers.

    GET  /v1/models          the loaded models and their privacy budget [read]
    POST /v1/sample          draw rows from one model, as JSON or CSV [sample]
    GET  /v1/generations     who drew what [read]
    GET  /healthz, /readyz, /metrics

Models are fitted offline (`python -m synthfraud fit`), so private data never passes through the
API. The service loads every <name>.joblib in SYNTHFRAUD_MODEL_DIR whose .sig matches
SYNTHFRAUD_MODEL_KEY; anything else is skipped and logged. In development with no model
directory, a differentially private conditional copula (epsilon 2) is fitted on generated demo
data at start-up. Settings: ops.py and docs/runbook.md.

    python -m synthfraud serve
"""
from __future__ import annotations

import io
import json
import math
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import text

from . import db, ops
from .signing import load_verified

settings = ops.load_settings()
ops.setup_logging(settings.log_level)
MODEL_DIR = os.environ.get("SYNTHFRAUD_MODEL_DIR", "").strip()
if settings.production and not MODEL_DIR:
    raise ops.ConfigError("production needs SYNTHFRAUD_MODEL_DIR: signed models fitted with `python -m synthfraud fit`")
MAX_ROWS = 100_000

engine = db.connect(settings.database_url)
_lock = threading.Lock()


def _epsilon(model) -> str:
    eps = getattr(model, "spent_epsilon", None)
    if eps is None:
        eps = getattr(model, "dp_epsilon", None)
    return "none (no formal privacy guarantee)" if eps is None or eps == math.inf else f"{float(eps):.2f}"


def _load_models() -> Dict[str, Dict[str, Any]]:
    if not MODEL_DIR:
        from .models import ConditionalCopula
        from .source import generate

        m = ConditionalCopula(dp_epsilon=2).fit(generate(20_000))
        ops.log.warning("no SYNTHFRAUD_MODEL_DIR: serving a demo model fitted on generated data")
        return {"demo-ccopula-eps2": {"model": m, "sha256": "demo", "epsilon": _epsilon(m)}}
    keys = [settings.secret["MODEL_KEY"], *settings.previous["MODEL_KEY"]]
    out = {}
    for f in sorted(Path(MODEL_DIR).glob("*.joblib")):
        try:
            model, digest = load_verified(f, keys)
        except ValueError as e:
            ops.log.error("model skipped", extra={"fields": {"file": f.name, "reason": str(e)}})
            ops.ERRORS.labels(kind="model_rejected").inc()
            continue
        out[f.stem] = {"model": model, "sha256": digest, "epsilon": _epsilon(model)}
    if settings.production and not out:
        raise ops.ConfigError(f"no signed models in {MODEL_DIR}")
    return out


MODELS = _load_models()
app = FastAPI(title="Synthetic fraud data service", version="1.0.0")
ops.install(app, settings, {"database": lambda: engine.connect().close(), "models": lambda: bool(MODELS)})


class SampleRequest(BaseModel):
    model: str = Field(..., max_length=100)
    n: int = Field(1000, ge=1, le=MAX_ROWS)
    fraud_type: Optional[str] = Field(None, max_length=40)
    seed: Optional[int] = Field(None, ge=0, le=2**32 - 1)
    format: Literal["json", "csv"] = "json"


@app.get("/v1/models", dependencies=[Depends(ops.require("read"))])
def list_models() -> Dict[str, Any]:
    return {"models": [{"name": k, "type": type(v["model"]).__name__, "epsilon": v["epsilon"], "sha256": v["sha256"]}
                       for k, v in MODELS.items()]}


@app.post("/v1/sample")
def sample(req: SampleRequest, ident: dict = Depends(ops.require("sample"))):
    entry = MODELS.get(req.model)
    if entry is None:
        raise HTTPException(404, "model not found")
    try:
        df = entry["model"].sample(req.n, condition={"fraud_type": req.fraud_type} if req.fraud_type else None, seed=req.seed)
    except (KeyError, ValueError):
        raise HTTPException(400, "this model can't be conditioned on that fraud type") from None
    with _lock, engine.begin() as c:
        c.execute(text("INSERT INTO generations VALUES (:i, :t, :u, :m, :h, :n, :f, :e)"),
                  {"i": uuid.uuid4().hex, "t": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                   "u": ident["user"] or "demo", "m": req.model, "h": entry["sha256"], "n": req.n,
                   "f": req.fraud_type, "e": entry["epsilon"]})
    ops.DECISIONS.labels(decision="rows_generated").inc(req.n)
    if req.format == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return Response(buf.getvalue(), media_type="text/csv")
    # via pandas' JSON writer: numpy scalars and NaN come out as plain JSON
    return {"model": req.model, "epsilon": entry["epsilon"], "rows": json.loads(df.to_json(orient="records"))}


@app.get("/v1/generations", dependencies=[Depends(ops.require("read"))])
def generations(limit: int = Query(100, ge=1, le=1000)) -> Dict[str, Any]:
    with engine.connect() as c:
        rows = c.execute(text("SELECT * FROM generations ORDER BY created_at DESC LIMIT :n"), {"n": limit}).mappings().all()
    return {"generations": [dict(r) for r in rows]}

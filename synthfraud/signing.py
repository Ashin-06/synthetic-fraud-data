"""Signed model files. A fitted model is a joblib (pickle) file, and unpickling runs code, so
the service only loads files whose HMAC-SHA256 matches SYNTHFRAUD_MODEL_KEY. `fit` writes the
signature next to the model as <model>.sig."""
from __future__ import annotations

import hashlib
import hmac
import io
from pathlib import Path

import joblib


def sign(path: str | Path, key: bytes) -> Path:
    data = Path(path).read_bytes()
    sig = Path(f"{path}.sig")
    sig.write_text(hmac.new(key, data, hashlib.sha256).hexdigest() + "\n", encoding="ascii")
    return sig


def load_verified(path: str | Path, keys: list[bytes]) -> tuple[object, str]:
    """The model and its SHA-256, or ValueError if no key's HMAC matches its .sig file."""
    data = Path(path).read_bytes()
    sig_file = Path(f"{path}.sig")
    if not sig_file.exists():
        raise ValueError(f"{Path(path).name} has no signature file")
    sig = sig_file.read_text(encoding="ascii").strip()
    if not any(hmac.compare_digest(hmac.new(k, data, hashlib.sha256).hexdigest(), sig) for k in keys):
        raise ValueError(f"{Path(path).name} is not signed with the configured model key")
    # load the bytes that were verified, not the file again (it could change in between)
    return joblib.load(io.BytesIO(data)), hashlib.sha256(data).hexdigest()  # nosec B301: HMAC-verified above

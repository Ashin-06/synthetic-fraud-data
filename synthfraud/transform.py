"""Reversible encoding of mixed-type tables for neural generators.

Continuous columns use mode-specific normalisation (Xu et al., CTGAN/TVAE 2019): a
variational Gaussian mixture finds the column's modes, each value is encoded as the
one-hot mode plus its scaled offset inside that mode. Heavy-tailed money columns are
log-transformed first. Categorical columns are one-hot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture

LOG_COLUMNS = {"amount", "avg_amount_30d", "distance_km", "device_age_days"}


@dataclass
class ColumnInfo:
    name: str
    kind: str  # continuous | categorical
    width: int
    gmm: BayesianGaussianMixture | None = None
    modes: np.ndarray | None = None  # active mode indices
    categories: list | None = None
    lo: float = 0.0
    hi: float = 0.0
    integer: bool = False


class DataTransformer:
    def __init__(self, max_modes: int = 8, seed: int = 0):
        self.max_modes, self.seed = max_modes, seed
        self.columns: list[ColumnInfo] = []

    def fit(self, df: pd.DataFrame, continuous, categorical) -> "DataTransformer":
        self.columns = []
        for c in continuous:
            x = self._fwd(c, df[c].to_numpy(float))
            gmm = BayesianGaussianMixture(n_components=self.max_modes, weight_concentration_prior=1e-3, max_iter=200,
                                          random_state=self.seed).fit(x.reshape(-1, 1))
            active = np.flatnonzero(gmm.weights_ > 5e-3)
            self.columns.append(ColumnInfo(c, "continuous", 1 + len(active), gmm, active, lo=float(df[c].min()),
                                           hi=float(df[c].max()), integer=bool(np.allclose(df[c] % 1, 0))))
        for c in categorical:
            cats = sorted(df[c].astype(str).unique())
            self.columns.append(ColumnInfo(c, "categorical", len(cats), categories=cats))
        return self

    @staticmethod
    def _fwd(name, x):
        return np.log1p(np.clip(x, 0, None)) if name in LOG_COLUMNS else x

    @staticmethod
    def _inv(name, x):
        return np.expm1(x) if name in LOG_COLUMNS else x

    @property
    def width(self) -> int:
        return sum(c.width for c in self.columns)

    def spans(self) -> list[tuple[ColumnInfo, int, int]]:
        out, i = [], 0
        for c in self.columns:
            out.append((c, i, i + c.width))
            i += c.width
        return out

    def transform(self, df: pd.DataFrame, rng=None) -> np.ndarray:
        rng = rng or np.random.default_rng(self.seed)
        parts = []
        for c in self.columns:
            if c.kind == "continuous":
                x = self._fwd(c.name, df[c.name].to_numpy(float)).reshape(-1, 1)
                means = c.gmm.means_.ravel()[c.modes]
                stds = np.sqrt(c.gmm.covariances_.ravel()[c.modes])
                probs = c.gmm.predict_proba(x)[:, c.modes] + 1e-9
                probs /= probs.sum(1, keepdims=True)
                pick = np.minimum((probs.cumsum(1) < rng.random(len(x))[:, None]).sum(1), len(c.modes) - 1)
                alpha = np.clip((x.ravel() - means[pick]) / (4 * stds[pick]), -0.99, 0.99)
                onehot = np.eye(len(c.modes))[pick]
                parts.append(np.column_stack([alpha, onehot]))
            else:
                idx = pd.Categorical(df[c.name].astype(str), categories=c.categories).codes
                parts.append(np.eye(len(c.categories))[np.clip(idx, 0, None)])
        return np.hstack(parts).astype(np.float32)

    def inverse(self, z: np.ndarray) -> pd.DataFrame:
        out = {}
        for c, s, e in self.spans():
            block = z[:, s:e]
            if c.kind == "continuous":
                alpha, mode = np.clip(block[:, 0], -1, 1), block[:, 1:].argmax(1)
                means = c.gmm.means_.ravel()[c.modes][mode]
                stds = np.sqrt(c.gmm.covariances_.ravel()[c.modes])[mode]
                x = np.clip(self._inv(c.name, alpha * 4 * stds + means), c.lo, c.hi)
                out[c.name] = np.round(x) if c.integer else np.round(x, 2)
            else:
                out[c.name] = np.array(c.categories)[block.argmax(1)]
        return pd.DataFrame(out)

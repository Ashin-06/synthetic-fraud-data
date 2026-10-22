"""Scorecard for a synthetic table: is it faithful, is it useful, and is it private?

Fidelity  per-column distribution distance (KS for numbers, total variation for
          categories), pairwise dependence error, and a discriminator test (can a model
          tell real rows from synthetic? AUC 0.5 = indistinguishable).
Utility   train on synthetic, test on real (TSTR) against train-real-test-real, for the
          task the data exists for: fraud detection.
Privacy   distance to closest record against a holdout baseline, near-copies, and a
          distance-based membership-inference attack (AUC 0.5 = no leakage), including
          on the rarest fraud records, which are the easiest to memorise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors

from .source import SCHEMA, Schema


def _encode(frames: list[pd.DataFrame], schema: Schema) -> list[np.ndarray]:
    """Shared numeric encoding: robust-scaled numbers (log for money) + one-hot categories."""
    allc = pd.concat(frames, keys=range(len(frames)))
    num = allc[list(schema.continuous)].astype(float).copy()
    for c in ("amount", "avg_amount_30d", "distance_km", "device_age_days"):
        if c in num:
            num[c] = np.log1p(num[c].clip(lower=0))
    num = (num - num.median()) / (num.quantile(0.75) - num.quantile(0.25)).replace(0, 1)
    parts = [num]
    if schema.categorical:  # a table can be all numbers once the label columns are set aside
        parts.append(pd.get_dummies(allc[list(schema.categorical)].astype(str)).astype(float))
    m = pd.concat(parts, axis=1).to_numpy(np.float32)
    sizes = np.cumsum([0] + [len(f) for f in frames])
    return [m[sizes[i]:sizes[i + 1]] for i in range(len(frames))]


def fidelity(real: pd.DataFrame, syn: pd.DataFrame, schema: Schema = SCHEMA, seed: int = 0) -> dict:
    cols = {}
    for c in schema.continuous:
        cols[c] = float(stats.ks_2samp(real[c], syn[c]).statistic)
    for c in schema.categorical:
        p = real[c].astype(str).value_counts(normalize=True)
        q = syn[c].astype(str).value_counts(normalize=True)
        cols[c] = float(0.5 * p.subtract(q, fill_value=0).abs().sum())
    r, s = _encode([real, syn], schema)
    with np.errstate(divide="ignore", invalid="ignore"):  # constant one-hot columns have zero variance
        cr = np.nan_to_num(np.corrcoef(r, rowvar=False))
        cs = np.nan_to_num(np.corrcoef(s, rowvar=False))
    n = min(len(real), len(syn), 8000)
    rng = np.random.default_rng(seed)
    x = np.vstack([r[rng.choice(len(r), n, replace=False)], s[rng.choice(len(s), n, replace=False)]])
    y = np.r_[np.zeros(n), np.ones(n)]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=150, random_state=0), x, y, cv=3, method="predict_proba")[:, 1]
    return {"column_distance_mean": float(np.mean(list(cols.values()))), "column_distance": cols,
            "correlation_error": float(np.abs(cr - cs).mean()), "discriminator_auc": float(roc_auc_score(y, p))}


def utility(train_real: pd.DataFrame, syn: pd.DataFrame, test_real: pd.DataFrame, schema: Schema = SCHEMA) -> dict:
    feats = Schema(schema.continuous, tuple(c for c in schema.categorical if c not in (schema.target, schema.condition)))
    tr, sy, te = _encode([train_real, syn, test_real], feats)
    y_tr, y_sy, y_te = (d[schema.target].astype(str).to_numpy() == "1" for d in (train_real, syn, test_real))
    out = {}
    for name, x, yy in (("train_real", tr, y_tr), ("train_synthetic", sy, y_sy)):
        if yy.sum() < 5:
            out[name] = float("nan")
            continue
        clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, class_weight="balanced", random_state=0).fit(x, yy)
        out[name] = float(average_precision_score(y_te, clf.predict_proba(te)[:, 1]))
    out["tstr_ratio"] = out["train_synthetic"] / out["train_real"] if out["train_real"] else float("nan")
    return out


def privacy(train: pd.DataFrame, holdout: pd.DataFrame, syn: pd.DataFrame, schema: Schema = SCHEMA, seed: int = 0) -> dict:
    tr, ho, sy = _encode([train, holdout, syn], schema)
    nn_train = NearestNeighbors(n_neighbors=1).fit(tr)
    dcr_syn = nn_train.kneighbors(sy)[0].ravel()
    dcr_hold = nn_train.kneighbors(ho)[0].ravel()
    # membership inference: members (train) should not sit closer to synthetic data than non-members (holdout)
    nn_syn = NearestNeighbors(n_neighbors=1).fit(sy)
    rng = np.random.default_rng(seed)
    k = min(len(tr), len(ho), 4000)
    mi = rng.choice(len(tr), k, replace=False)
    hi = rng.choice(len(ho), k, replace=False)
    d_mem, d_non = nn_syn.kneighbors(tr[mi])[0].ravel(), nn_syn.kneighbors(ho[hi])[0].ravel()
    mia = roc_auc_score(np.r_[np.ones(k), np.zeros(k)], -np.r_[d_mem, d_non])
    # the same attack restricted to fraud records (rare, so most at risk of memorisation)
    fr_tr = np.flatnonzero(train[schema.target].astype(str).to_numpy() == "1")
    fr_ho = np.flatnonzero(holdout[schema.target].astype(str).to_numpy() == "1")
    if len(fr_tr) >= 10 and len(fr_ho) >= 10:
        a = nn_syn.kneighbors(tr[fr_tr])[0].ravel()
        b = nn_syn.kneighbors(ho[fr_ho])[0].ravel()
        mia_rare = roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], -np.r_[a, b])
    else:
        mia_rare = float("nan")
    p5 = float(np.percentile(dcr_hold, 5))
    return {"dcr_synthetic_median": float(np.median(dcr_syn)), "dcr_holdout_median": float(np.median(dcr_hold)),
            "share_closer_than_holdout_p5": float((dcr_syn < p5).mean()), "near_copies": int((dcr_syn < 1e-6).sum()),
            "membership_auc": float(mia), "membership_auc_fraud_rows": float(mia_rare)}


def scorecard(fid: dict, uti: dict, pri: dict) -> dict:
    """0-100 sub-scores with explicit, documented mappings."""
    f = 100 * max(0.0, 1 - fid["column_distance_mean"] * 2) * 0.5 + 100 * max(0.0, 1 - 2 * abs(fid["discriminator_auc"] - 0.5)) * 0.5
    u = 100 * min(1.0, max(0.0, uti["tstr_ratio"])) if np.isfinite(uti["tstr_ratio"]) else 0.0
    worst = max(pri["membership_auc"], pri["membership_auc_fraud_rows"] if np.isfinite(pri["membership_auc_fraud_rows"]) else 0)
    p = 100 * max(0.0, 1 - 2 * max(0.0, worst - 0.5) * 2.5) * (0.5 if pri["near_copies"] else 1.0)
    p *= max(0.0, 1 - max(0.0, pri["share_closer_than_holdout_p5"] - 0.05) * 4)
    return {"fidelity": round(f, 1), "utility": round(u, 1), "privacy": round(p, 1)}

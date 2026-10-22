"""Fit -> sample -> evaluate -> report, for any synthesizer."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .evaluate import fidelity, privacy, scorecard, utility
from .source import SCHEMA


def split(df: pd.DataFrame, seed: int = 0):
    """train (synthesizer sees it) / holdout (privacy baseline) / test (utility)."""
    idx = np.random.default_rng(seed).permutation(len(df))
    a, b = int(0.6 * len(df)), int(0.8 * len(df))
    return df.iloc[idx[:a]].reset_index(drop=True), df.iloc[idx[a:b]].reset_index(drop=True), df.iloc[idx[b:]].reset_index(drop=True)


def assess(model, train, holdout, test, n: int | None = None, seed: int = 0, log=print) -> dict:
    t0 = time.time()
    syn = model.sample(n or len(train), seed=seed)
    fid, uti, pri = fidelity(train, syn), utility(train, syn, test), privacy(train, holdout, syn)
    eps = getattr(model, "spent_epsilon", math.inf)
    res = {"model": model.name + (f" (DP eps={eps:.1f})" if math.isfinite(eps) else ""), "epsilon": eps,
           "fidelity": fid, "utility": uti, "privacy": pri, "score": scorecard(fid, uti, pri), "seconds": round(time.time() - t0, 1)}
    log(f"{res['model']:28s} fidelity {res['score']['fidelity']:5.1f}  utility {res['score']['utility']:5.1f}  "
        f"privacy {res['score']['privacy']:5.1f}  | discr. AUC {fid['discriminator_auc']:.3f}  TSTR {uti['train_synthetic']:.3f}/"
        f"{uti['train_real']:.3f}  MIA {pri['membership_auc']:.3f}/{pri['membership_auc_fraud_rows']:.3f}")
    return res, syn


def html_report(results: list[dict], real: pd.DataFrame, samples: dict[str, pd.DataFrame], path: str | Path) -> None:
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Scorecard", "Amount distribution (log)", "Fraud-type mix", "Membership inference AUC"))
    names = [r["model"] for r in results]
    for k, color in (("fidelity", "#2a9d8f"), ("utility", "#e9c46a"), ("privacy", "#e76f51")):
        fig.add_bar(x=names, y=[r["score"][k] for r in results], name=k, marker_color=color, row=1, col=1)
    fig.add_histogram(x=np.log10(real["amount"] + 1), name="real", opacity=0.55, histnorm="probability", row=1, col=2)
    for name, s in samples.items():
        fig.add_histogram(x=np.log10(s["amount"] + 1), name=name, opacity=0.55, histnorm="probability", row=1, col=2)
    mix = real[SCHEMA.condition].value_counts(normalize=True)
    fig.add_bar(x=mix.index, y=mix.values, name="real mix", row=2, col=1)
    for name, s in samples.items():
        m = s[SCHEMA.condition].value_counts(normalize=True).reindex(mix.index).fillna(0)
        fig.add_bar(x=m.index, y=m.values, name=f"{name} mix", row=2, col=1)
    fig.update_yaxes(type="log", row=2, col=1)
    fig.add_bar(x=names, y=[r["privacy"]["membership_auc"] for r in results], name="all rows", row=2, col=2)
    fig.add_bar(x=names, y=[r["privacy"]["membership_auc_fraud_rows"] for r in results], name="fraud rows", row=2, col=2)
    fig.add_hline(y=0.5, line_dash="dot", row=2, col=2)
    fig.update_layout(height=820, barmode="group", title="Synthetic fraud data report")
    rows = "".join(
        f"<tr><td>{r['model']}</td><td>{r['score']['fidelity']}</td><td>{r['score']['utility']}</td><td>{r['score']['privacy']}</td>"
        f"<td>{r['fidelity']['discriminator_auc']:.3f}</td><td>{r['utility']['train_synthetic']:.3f} / {r['utility']['train_real']:.3f}</td>"
        f"<td>{r['privacy']['membership_auc']:.3f}</td><td>{r['privacy']['near_copies']}</td></tr>" for r in results)
    table = ("<table><tr><th>model</th><th>fidelity</th><th>utility</th><th>privacy</th><th>discriminator AUC</th>"
             "<th>fraud PR-AUC synth / real</th><th>membership AUC</th><th>exact copies</th></tr>" + rows + "</table>")
    style = ("<style>body{font-family:system-ui,sans-serif;margin:24px;color:#14181d}table{border-collapse:collapse;margin:16px 0}"
             "td,th{border-bottom:1px solid #ddd;padding:6px 10px;text-align:left}</style>")
    Path(path).write_text(f"<!doctype html><meta charset=utf-8><title>Synthetic data report</title>{style}<h1>Synthetic fraud data report</h1>"
                          f"{table}{fig.to_html(full_html=False, include_plotlyjs='cdn')}", encoding="utf-8")


def compare(models: list, df: pd.DataFrame, out: str = "results", seed: int = 0, log=print) -> list[dict]:
    train, holdout, test = split(df, seed)
    Path(out).mkdir(parents=True, exist_ok=True)
    results, samples = [], {}
    for m in models:
        log(f"fitting {m.name} ...")
        m.fit(train, log=log) if m.name == "tvae" else m.fit(train)
        r, syn = assess(m, train, holdout, test, seed=seed, log=log)
        results.append(r)
        samples[r["model"]] = syn
    html_report(results, train, samples, Path(out) / "report.html")
    (Path(out) / "results.json").write_text(json.dumps(results, indent=2, default=float))
    return results

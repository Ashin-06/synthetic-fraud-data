"""Command line.

    python -m synthfraud benchmark                     # every synthesizer, scorecard + report.html -> results/
    python -m synthfraud fit --model ccopula --epsilon 2 --out model.joblib [--data private.csv]
    python -m synthfraud sample model.joblib -n 50000 --out synthetic.csv [--fraud-type upi_scam]
    python -m synthfraud evaluate --real private.csv --synthetic synthetic.csv
"""
from __future__ import annotations

import argparse
import json
import sys

import joblib
import pandas as pd

from .models import ConditionalCopula, GaussianCopula, TVAE
from .pipeline import compare, split
from .source import generate, load_csv


def build(name: str, eps: float | None, epochs: int):
    if name == "copula":
        return GaussianCopula()
    if name == "ccopula":
        return ConditionalCopula(dp_epsilon=eps)
    return TVAE(epochs=epochs, dp_epsilon=eps, lr=2e-3 if eps else 1e-3)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="synthfraud")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("--rows", type=int, default=30_000)
    b.add_argument("--epochs", type=int, default=150)
    b.add_argument("--out", default="results")
    f = sub.add_parser("fit")
    f.add_argument("--model", choices=["copula", "ccopula", "tvae"], default="ccopula")
    f.add_argument("--epsilon", type=float, help="differential privacy budget (omit for no DP)")
    f.add_argument("--epochs", type=int, default=150)
    f.add_argument("--data", help="CSV with the schema columns (default: generated demo data)")
    f.add_argument("--out", default="model.joblib")
    s = sub.add_parser("sample")
    s.add_argument("model")
    s.add_argument("-n", type=int, default=10_000)
    s.add_argument("--fraud-type")
    s.add_argument("--out", default="synthetic.csv")
    e = sub.add_parser("evaluate")
    e.add_argument("--real", required=True)
    e.add_argument("--synthetic", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "benchmark":
        df = generate(a.rows)
        models = [GaussianCopula(), ConditionalCopula(), ConditionalCopula(dp_epsilon=8), ConditionalCopula(dp_epsilon=2),
                  ConditionalCopula(dp_epsilon=1), TVAE(epochs=a.epochs), TVAE(epochs=max(a.epochs // 4, 10), dp_epsilon=8, lr=2e-3)]
        compare(models, df, a.out)
        print(f"\nreport: {a.out}/report.html")
    elif a.cmd == "fit":
        df = load_csv(a.data) if a.data else generate()
        m = build(a.model, a.epsilon, a.epochs)
        (m.fit(df, log=print) if a.model == "tvae" else m.fit(df))
        joblib.dump(m, a.out)
        eps = getattr(m, "spent_epsilon", float("inf"))
        print(f"saved {a.out}" + (f"  (differential privacy: epsilon={eps:.2f}, delta={m.delta})" if eps != float("inf") else "  (no formal privacy guarantee)"))
    elif a.cmd == "sample":
        m = joblib.load(a.model)
        df = m.sample(a.n, condition={"fraud_type": a.fraud_type} if a.fraud_type else None)
        df.to_csv(a.out, index=False)
        print(f"wrote {len(df)} rows to {a.out}; fraud types: {df['fraud_type'].value_counts().to_dict()}")
    else:
        from .evaluate import fidelity, privacy, scorecard, utility

        real = load_csv(a.real)
        syn = pd.read_csv(a.synthetic, dtype={"is_fraud": str})
        train, holdout, test = split(real)
        fid, uti, pri = fidelity(train, syn), utility(train, syn, test), privacy(train, holdout, syn)
        print(json.dumps({"score": scorecard(fid, uti, pri), "fidelity": {k: v for k, v in fid.items() if k != "column_distance"},
                          "utility": uti, "privacy": pri}, indent=2))


if __name__ == "__main__":
    main()

"""Synthetic data studio.

    streamlit run app.py
"""
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from synthfraud.evaluate import fidelity, privacy, scorecard, utility
from synthfraud.models import ConditionalCopula, GaussianCopula, TVAE
from synthfraud.pipeline import split
from synthfraud.source import FRAUD_TYPES, generate, load_csv

st.set_page_config(page_title="Synthetic Fraud Data", layout="wide")
st.title("Synthetic Fraud Data Studio")
st.caption("Generate shareable transaction data with real fraud patterns, and measure how faithful, useful and private it is.")

with st.sidebar:
    up = st.file_uploader("Private table (CSV with the schema columns)", type=["csv"])
    rows = st.slider("Demo rows (if no file)", 5000, 40000, 15000, 5000)
    kind = st.selectbox("Synthesizer", ["Conditional copula", "Gaussian copula", "Conditional VAE (TVAE)"])
    dp = st.toggle("Differential privacy", value=True, disabled=kind == "Gaussian copula")
    eps = st.select_slider("Privacy budget ε (smaller = more private)", [0.5, 1.0, 2.0, 4.0, 8.0, 16.0], value=2.0, disabled=not dp)
    epochs = st.slider("TVAE epochs", 10, 200, 60, 10, disabled=kind != "Conditional VAE (TVAE)")
    go_btn = st.button("Generate and evaluate", type="primary", width="stretch")


@st.cache_data
def data(rows, file):
    return load_csv(file) if file is not None else generate(rows)


real = data(rows, up)
tab_gen, tab_bench = st.tabs(["Generate", "Benchmark"])

with tab_gen:
    if go_btn:
        train, holdout, test = split(real)
        with st.spinner("Fitting…"):
            if kind == "Gaussian copula":
                m = GaussianCopula().fit(train)
            elif kind == "Conditional copula":
                m = ConditionalCopula(dp_epsilon=eps if dp else None).fit(train)
            else:
                m = TVAE(epochs=epochs, dp_epsilon=eps if dp else None, lr=2e-3 if dp else 1e-3).fit(train)
            syn = m.sample(len(train), seed=1)
            fid, uti, pri = fidelity(train, syn), utility(train, syn, test), privacy(train, holdout, syn)
            sc = scorecard(fid, uti, pri)
        st.session_state["last"] = (syn, fid, uti, pri, sc, getattr(m, "spent_epsilon", math.inf), train)
    if "last" in st.session_state:
        syn, fid, uti, pri, sc, spent, train = st.session_state["last"]
        c = st.columns(4)
        c[0].metric("Fidelity", sc["fidelity"])
        c[1].metric("Utility", sc["utility"], f"fraud PR-AUC {uti['train_synthetic']:.3f} vs {uti['train_real']:.3f} real", delta_color="off")
        c[2].metric("Privacy", sc["privacy"], f"membership AUC {pri['membership_auc']:.3f}", delta_color="off")
        c[3].metric("Formal guarantee", "none" if not math.isfinite(spent) else f"ε = {spent:.2f}", "δ = 1e-5" if math.isfinite(spent) else None, delta_color="off")
        l, r = st.columns(2)
        fig = go.Figure([go.Histogram(x=train["amount"].clip(upper=train["amount"].quantile(0.99)), name="real", opacity=0.6, histnorm="probability"),
                         go.Histogram(x=syn["amount"].clip(upper=train["amount"].quantile(0.99)), name="synthetic", opacity=0.6, histnorm="probability")])
        fig.update_layout(barmode="overlay", title="Amount", height=320)
        l.plotly_chart(fig, width="stretch")
        mix_r = train["fraud_type"].value_counts(normalize=True).reindex(FRAUD_TYPES).fillna(0)
        mix_s = syn["fraud_type"].value_counts(normalize=True).reindex(FRAUD_TYPES).fillna(0)
        fig = go.Figure([go.Bar(x=FRAUD_TYPES, y=mix_r, name="real"), go.Bar(x=FRAUD_TYPES, y=mix_s, name="synthetic")])
        fig.update_layout(yaxis_type="log", title="Fraud-type mix", height=320)
        r.plotly_chart(fig, width="stretch")
        dist = pd.Series(fid["column_distance"]).sort_values()
        fig = go.Figure(go.Bar(x=dist.values, y=dist.index, orientation="h"))
        fig.update_layout(title="Per-column distance (KS / total variation; lower is better)", height=380)
        st.plotly_chart(fig, width="stretch")
        st.markdown("**Privacy detail**")
        st.json({k: round(v, 4) if isinstance(v, float) else v for k, v in pri.items()})
        st.dataframe(syn.head(200), hide_index=True, width="stretch")
        st.download_button("Download synthetic CSV", syn.to_csv(index=False), "synthetic_transactions.csv", "text/csv")
    else:
        st.info("Choose a synthesizer and press Generate.")

with tab_bench:
    rep = Path("results/report.html")
    if rep.exists():
        st.components.v1.html(rep.read_text(encoding="utf-8"), height=1100, scrolling=True)
    else:
        st.info("Run `python -m synthfraud benchmark` to produce the full comparison.")

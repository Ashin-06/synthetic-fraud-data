import math

import numpy as np
import pandas as pd
import pytest

from synthfraud.evaluate import fidelity, privacy, utility
from synthfraud.models import ConditionalCopula, GaussianCopula, TVAE, _rho_for_eps, _zcdp_to_eps, rdp_epsilon
from synthfraud.pipeline import split
from synthfraud.source import SCHEMA, generate
from synthfraud.transform import DataTransformer


@pytest.fixture(scope="module")
def data():
    return split(generate(6000, fraud_rate=0.04, seed=2))


def test_transformer_round_trip(data):
    train, _, _ = data
    tf = DataTransformer().fit(train, SCHEMA.continuous, SCHEMA.categorical)
    back = tf.inverse(tf.transform(train))
    for c in SCHEMA.categorical:
        assert (back[c] == train[c].astype(str)).mean() == 1.0
    rel = np.abs(back["amount"] - train["amount"]) / (train["amount"] + 1)
    assert np.median(rel) < 0.05


def test_conditional_copula_keeps_fraud_patterns(data):
    train, holdout, test = data
    syn = ConditionalCopula().fit(train).sample(len(train), seed=1)
    ct = syn[syn.fraud_type == "card_testing"]
    assert len(ct) > 10 and ct["amount"].median() < 60 and ct["txn_count_24h"].median() > 8
    assert (syn.loc[syn.fraud_type == "none", "is_fraud"] == "0").all()
    u = utility(train, syn, test)
    assert u["train_synthetic"] > 0.6 * u["train_real"]


def test_plain_copula_loses_fraud_signal(data):
    train, _, test = data
    u = utility(train, GaussianCopula().fit(train).sample(len(train), seed=1), test)
    assert u["train_synthetic"] < 0.5 * u["train_real"]


def test_dp_accounting():
    assert _zcdp_to_eps(_rho_for_eps(2.0, 1e-5), 1e-5) == pytest.approx(2.0, rel=1e-3)
    assert rdp_epsilon(0.01, 1.0, 1000) < rdp_epsilon(0.01, 1.0, 4000)
    m = ConditionalCopula(dp_epsilon=1.0).fit(generate(3000, seed=5))
    assert m.spent_epsilon == pytest.approx(1.0, rel=1e-3)


def test_dp_reduces_membership_leak_on_rare_rows(data):
    train, holdout, _ = data
    plain = privacy(train, holdout, ConditionalCopula().fit(train).sample(len(train), seed=3))
    private = privacy(train, holdout, ConditionalCopula(dp_epsilon=0.5).fit(train).sample(len(train), seed=3))
    assert private["membership_auc_fraud_rows"] <= plain["membership_auc_fraud_rows"] + 0.02
    assert private["near_copies"] == 0


def test_conditional_sampling(data):
    train, _, _ = data
    s = ConditionalCopula().fit(train).sample(300, condition={"fraud_type": "upi_scam"})
    assert set(s.fraud_type) == {"upi_scam"} and (s.channel == "upi").mean() > 0.9


def test_tvae_smoke(data):
    train, _, _ = data
    m = TVAE(epochs=3, batch=256, device="cpu").fit(train)
    s = m.sample(500, condition={"fraud_type": "skimming"})
    assert len(s) == 500 and set(s.columns) == set(SCHEMA.columns) and set(s.fraud_type) == {"skimming"}
    assert math.isinf(m.spent_epsilon)


def test_fidelity_detects_garbage(data):
    train, _, _ = data
    junk = train.sample(frac=1, random_state=0).copy()
    junk["amount"] = junk["amount"] * 10
    assert fidelity(train, junk)["column_distance"]["amount"] > 0.5

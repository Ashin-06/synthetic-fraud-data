import numpy as np
import pandas as pd

from synthfraud.evaluate import utility
from synthfraud.models import ConditionalCopula
from synthfraud.source import ULB_BOUNDS, ULB_CATEGORIES, ULB_SCHEMA, load_ulb


def _ulb(tmp_path, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"Time": rng.uniform(0, 172_800, n), **{f"V{i}": rng.normal(size=n) for i in range(1, 29)},
                       "Amount": rng.gamma(2, 40, n), "Class": (rng.random(n) < 0.05).astype(int)})
    df.loc[df.Class == 1, "V14"] -= 4  # a fraud signal for the utility check
    f = tmp_path / "cc.csv"
    df.to_csv(f, index=False)
    return load_ulb(f)


def test_ulb_table_maps_onto_the_schema(tmp_path):
    df = _ulb(tmp_path)
    assert list(df.columns) == ULB_SCHEMA.columns
    assert set(df["fraud_type"]) == {"none", "fraud"} and set(df["is_fraud"]) == {"0", "1"}


def test_dp_copula_and_utility_work_on_an_all_numeric_table(tmp_path):
    df = _ulb(tmp_path)
    m = ConditionalCopula(ULB_SCHEMA, dp_epsilon=8, bounds=ULB_BOUNDS, categories=ULB_CATEGORIES).fit(df)
    syn = m.sample(2000, seed=1)
    assert np.isfinite(m.spent_epsilon) and set(syn["fraud_type"]) <= {"none", "fraud"}
    u = utility(df.iloc[:2000], syn, df.iloc[2000:], ULB_SCHEMA)  # used to crash: no non-label categorical columns
    assert 0 <= u["train_synthetic"] <= 1

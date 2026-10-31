import numpy as np
import pandas as pd
import pytest

from src.features import CyclicTimeTransformer, InteractionTransformer, OOFWoEEncoder, woe_table


def test_cyclic_december_and_january_are_neighbours():
    X = pd.DataFrame({"month": [12, 1, 6]})
    out = CyclicTimeTransformer(periods={"month": 12}).fit_transform(X)
    pts = out[["month_sin", "month_cos"]].to_numpy()
    dec_jan = np.linalg.norm(pts[0] - pts[1])
    dec_jun = np.linalg.norm(pts[0] - pts[2])
    assert dec_jan < 0.6 < dec_jun
    assert np.allclose((pts ** 2).sum(axis=1), 1.0)


def test_cyclic_passes_other_columns_and_names():
    X = pd.DataFrame({"m": [1, 2], "m_missing": [0.0, 1.0]})
    tr = CyclicTimeTransformer(periods={"m": 12}).fit(X)
    out = tr.transform(X)
    assert list(out.columns) == ["m_missing", "m_sin", "m_cos"] == list(tr.get_feature_names_out())


def test_cyclic_rejects_unknown_column():
    with pytest.raises(ValueError, match="немає на вході"):
        CyclicTimeTransformer(periods={"hour": 24}).fit(pd.DataFrame({"m": [1]}))


def test_interactions_ratio_with_epsilon_and_product():
    X = pd.DataFrame({"a": [10.0, 5.0], "b": [2.0, 0.0]})
    tr = InteractionTransformer(ratios=[["a", "b"]], products=[["a", "b"]], epsilon=1e-5).fit(X)
    out = tr.transform(X)
    assert out.loc[0, "a_div_b"] == pytest.approx(10 / (2 + 1e-5))
    assert out.loc[1, "a_div_b"] == pytest.approx(5 / 1e-5)
    assert list(out["a_x_b"]) == [20.0, 0.0]
    assert tr.n_nonfinite_ == 0


def test_interactions_replace_nonfinite():
    X = pd.DataFrame({"a": [1.0], "b": [-1e-5]})
    tr = InteractionTransformer(ratios=[["a", "b"]]).fit(X)
    assert tr.transform(X).loc[0, "a_div_b"] == 0.0
    assert tr.n_nonfinite_ == 1


def test_woe_table_sign_convention():
    values = np.array(["good_cat"] * 9 + ["bad_cat"] * 9)
    y = np.array([0] * 8 + [1] + [1] * 8 + [0])
    table = woe_table(values, y, smoothing=0.5)
    assert table.loc["good_cat", "woe"] > 0 > table.loc["bad_cat", "woe"]


def test_oof_does_not_leak_own_label_for_singletons():
    rng = np.random.default_rng(0)
    common = rng.choice(["a", "b"], size=200)
    y = rng.integers(0, 2, size=200)
    singletons = [f"s{i}" for i in range(40)]
    X = pd.DataFrame({"c": np.r_[common, singletons]})
    y = np.r_[y, [0, 1] * 20]
    enc = OOFWoEEncoder(n_splits=5, smoothing=0.5, random_state=0)
    oof = enc.fit_transform(X, y)[200:, 0]
    plain = enc.transform(X)[200:, 0]
    assert np.allclose(oof, 0.0)
    assert np.corrcoef(plain, y[200:])[0, 1] < -0.99


def test_oof_requires_binary_target():
    with pytest.raises(ValueError, match="бінарну"):
        OOFWoEEncoder().fit(pd.DataFrame({"c": ["a", "b"]}), [0, 2])
    with pytest.raises(ValueError, match="потребує"):
        OOFWoEEncoder().fit(pd.DataFrame({"c": ["a"]}))


def test_purpose_woe_matches_reference_direction(dataset):
    X, y = dataset
    mask = X["purpose"].notna()
    table = woe_table(X.loc[mask, "purpose"].to_numpy(), y[mask].to_numpy(), smoothing=0.5)
    assert table.loc["radio/television", "woe"] > 0 > table.loc["education", "woe"]

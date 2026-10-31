import numpy as np
import pandas as pd
import pytest

from src.extraction import SignalExtractor, TextExtractor, clean_text, sequences_by_key

CORPUS = ["late fee on my car loan", "interest rate on personal loan was wrong",
          "collector called about loan payment", "fee charged twice on payment"]


def test_clean_text_handles_none_and_redactions():
    assert clean_text(None) == "" and clean_text(float("nan")) == ""
    assert clean_text("I paid XXXX to   the bank") == "I paid to the bank"


def test_text_extractor_shapes_and_empty_inputs():
    X = pd.DataFrame({"t": CORPUS + ["", None]})
    ex = TextExtractor(max_features=500, n_components=2, random_state=0)
    Z = ex.fit_transform(X)
    assert Z.shape == (6, 2)
    assert ex.n_empty_fit_ == 2
    assert np.allclose(Z[4:], 0.0)
    assert list(ex.get_feature_names_out()) == ["t_svd0", "t_svd1"]
    Z_new = ex.transform(pd.DataFrame({"t": [None, "loan fee"]}))
    assert Z_new.shape == (2, 2) and ex.n_empty_transform_ == 1


def test_text_extractor_all_empty_fails_loud():
    with pytest.raises(ValueError, match="порожні"):
        TextExtractor().fit(pd.DataFrame({"t": ["", None]}))


def test_signal_extractor_empty_short_nonfinite():
    ex = SignalExtractor(n_fft=16, hop_length=4, n_mels=16, n_mfcc=13).fit()
    signals = [np.array([]), None, np.array([3.0]), np.array([0, -1, 2, np.nan, -3, 1.0] * 4)]
    out = ex.transform(signals)
    assert out.shape == (4, 15)
    assert out.iloc[:3].isna().all().all()
    assert out.iloc[3].notna().all()
    assert ex.report_ == {"total": 4, "empty": 2, "too_short": 1, "padded": 0, "nonfinite_removed": 4}


def test_signal_extractor_zcr_reflects_sign_changes():
    ex = SignalExtractor().fit()
    alternating = np.tile([1.0, -1.0], 16)
    constant = np.ones(32)
    zcr = ex.transform([alternating, constant])["sig_zcr"].to_numpy()
    assert zcr[0] > 0.5 > zcr[1]


def test_sequences_by_key_orders_and_fills_missing_keys():
    events = pd.DataFrame({"k": [1, 1, 2], "t": [2, 1, 1], "v": [20.0, 10.0, 5.0]})
    seqs = sequences_by_key(events, [1, 2, 3], "k", "v", ["t"])
    assert list(seqs[1]) == [10.0, 20.0] and list(seqs[2]) == [5.0] and seqs[3].size == 0

from __future__ import annotations

import re
import warnings
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

_REDACTION = re.compile(r"\bx{2,}\b", flags=re.IGNORECASE)


def clean_text(value) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", _REDACTION.sub(" ", value)).strip()


class TextExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, max_features: int = 500, n_components: int = 5, stop_words: str = "english",
                 token_pattern: str = r"(?u)\b[^\W\d_]{2,}\b", random_state: int = 0):
        self.max_features = max_features
        self.n_components = n_components
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.random_state = random_state

    @staticmethod
    def _texts(X) -> np.ndarray:
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if X.shape[1] != 1:
            raise ValueError(f"TextExtractor очікує одну текстову колонку, отримано {X.shape[1]}")
        return np.array([clean_text(v) for v in X.iloc[:, 0]], dtype=object)

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        texts = self._texts(X)
        self.input_name_ = str((X.columns[0] if isinstance(X, pd.DataFrame) else "text"))
        self.n_features_in_ = 1
        self.n_empty_fit_ = int(sum(1 for t in texts if not t))
        if self.n_empty_fit_ == len(texts):
            raise ValueError("TextExtractor: усі тексти навчальної вибірки порожні — словник не побудувати")
        self.vectorizer_ = TfidfVectorizer(max_features=self.max_features, stop_words=self.stop_words,
                                           token_pattern=self.token_pattern)
        tfidf = self.vectorizer_.fit_transform(texts)
        n_components = min(self.n_components, tfidf.shape[1] - 1)
        self.svd_ = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        return self.svd_.fit_transform(tfidf)

    def transform(self, X):
        check_is_fitted(self, "svd_")
        texts = self._texts(X)
        self.n_empty_transform_ = int(sum(1 for t in texts if not t))
        return self.svd_.transform(self.vectorizer_.transform(texts))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "svd_")
        return np.asarray([f"{self.input_name_}_svd{i}" for i in range(self.svd_.n_components)], dtype=object)


class SignalExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, sr: float = 12.0, n_fft: int = 16, hop_length: int = 4, n_mels: int = 16,
                 n_mfcc: int = 13, min_length: int = 2, prefix: str = "sig"):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.min_length = min_length
        self.prefix = prefix

    def fit(self, X=None, y=None):
        if self.n_mfcc > self.n_mels:
            raise ValueError("n_mfcc не може перевищувати n_mels")
        self.n_features_in_ = 1
        return self

    def feature_names(self):
        return [f"{self.prefix}_zcr", f"{self.prefix}_centroid"] + [
            f"{self.prefix}_mfcc{i}" for i in range(self.n_mfcc)]

    def _one(self, signal) -> np.ndarray:
        import librosa

        y = np.asarray(signal if signal is not None else [], dtype=float).ravel()
        finite = np.isfinite(y)
        self.report_["nonfinite_removed"] += int((~finite).sum())
        y = y[finite]
        if y.size == 0:
            self.report_["empty"] += 1
            return np.full(len(self.feature_names()), np.nan)
        if y.size < self.min_length:
            self.report_["too_short"] += 1
            return np.full(len(self.feature_names()), np.nan)
        if y.size < self.n_fft:
            self.report_["padded"] += 1
            y = np.pad(y, (0, self.n_fft - y.size))
        kw = dict(n_fft=self.n_fft, hop_length=self.hop_length)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.n_fft, hop_length=self.hop_length)
            centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr, **kw)
            mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc, n_mels=self.n_mels, **kw)
        return np.concatenate([[zcr.mean()], [np.nan_to_num(centroid).mean()], mfcc.mean(axis=1)])

    def transform(self, X: Sequence) -> pd.DataFrame:
        self.report_ = {"total": 0, "empty": 0, "too_short": 0, "padded": 0, "nonfinite_removed": 0}
        signals = list(X.iloc[:, 0]) if isinstance(X, pd.DataFrame) else list(X)
        rows = []
        for s in signals:
            self.report_["total"] += 1
            rows.append(self._one(s))
        index = X.index if isinstance(X, (pd.DataFrame, pd.Series)) else None
        return pd.DataFrame(np.vstack(rows) if rows else np.empty((0, len(self.feature_names()))),
                            columns=self.feature_names(), index=index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names(), dtype=object)


def sequences_by_key(events: pd.DataFrame, keys: Sequence, key_col: str, value_col: str,
                     order_cols: Sequence[str]) -> Dict:
    ordered = events.sort_values([key_col, *order_cols])
    grouped = {k: g[value_col].to_numpy(dtype=float) for k, g in ordered.groupby(key_col, sort=False)}
    return {k: grouped.get(k, np.array([], dtype=float)) for k in keys}

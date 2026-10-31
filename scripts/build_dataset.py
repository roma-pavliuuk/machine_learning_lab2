import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.autofe import run_dfs
from src.extraction import SignalExtractor, sequences_by_key

PROCESSED = ROOT / "data" / "processed"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tabular(cfg) -> pd.DataFrame:
    path = (ROOT / cfg["tabular"]["path"]).resolve()
    actual = sha256(path)
    if actual != cfg["tabular"]["sha256"]:
        sys.exit(f"{path}: SHA-256 {actual} не збігається з версією ЛР №1")
    return pd.read_csv(path)


def sample_complaints(cfg, n: int, seed: int) -> pd.DataFrame:
    tc = cfg["text_corpus"]
    df = pd.read_csv(ROOT / tc["archive"], dtype=str, low_memory=False)
    pool = df[df["product"] == tc["product"]]
    if tc.get("require_narrative"):
        pool = pool[pool["consumer_complaint_narrative"].notna()]
    pool = pool.assign(complaint_id=pool["complaint_id"].astype(int)).sort_values("complaint_id")
    sample = pool.sample(n=n, random_state=seed).reset_index(drop=True)
    cols = ["complaint_id", "date_received", "sub_product", "issue", "company", "state",
            "consumer_complaint_narrative"]
    sample = sample[cols]
    out = ROOT / tc["sample_out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    return sample, len(pool)


def sample_payments(cfg, n: int) -> tuple:
    pc = cfg["payments"]
    ids = (pd.read_csv(ROOT / pc["clients_archive"], usecols=["SK_ID_CURR"])["SK_ID_CURR"]
           .sort_values().head(n).tolist())
    events = pd.read_csv(ROOT / pc["installments_archive"])
    events = events[events["SK_ID_CURR"].isin(ids)].copy()
    events["DAYS_LATE"] = events["DAYS_ENTRY_PAYMENT"] - events["DAYS_INSTALMENT"]
    events.to_csv(ROOT / pc["sample_out"], index=False)
    return ids, events


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "sources.yaml").read_text(encoding="utf-8"))
    n, seed = cfg["n_rows"], cfg["seed"]
    PROCESSED.mkdir(parents=True, exist_ok=True)

    tab = load_tabular(cfg)
    if len(tab) != n:
        sys.exit(f"Очікувалось {n} заявок, отримано {len(tab)}")

    complaints, pool_size = sample_complaints(cfg, n, seed)
    ids, events = sample_payments(cfg, n)

    data = tab.copy()
    data.insert(0, "applicant_id", np.arange(n))
    data["source_complaint_id"] = complaints["complaint_id"].to_numpy()
    data["source_sk_id_curr"] = ids
    data["complaint_text"] = complaints["consumer_complaint_narrative"].to_numpy()
    data["complaint_company"] = complaints["company"].to_numpy()
    received = pd.to_datetime(complaints["date_received"], format="%m/%d/%Y")
    data["complaint_month"] = received.dt.month.to_numpy()
    data["complaint_weekday"] = received.dt.weekday.to_numpy()

    sc = cfg["signal"]
    seqs = sequences_by_key(events, ids, "SK_ID_CURR", sc["value"], sc["order"])
    extractor = SignalExtractor(**sc["extractor"]).fit()
    signal_features = extractor.transform([seqs[k] for k in ids])
    data = pd.concat([data, signal_features.set_index(data.index)], axis=1)

    af = cfg["autofe"]
    sig_cols = extractor.feature_names()
    clients = data[["applicant_id", *af["client_columns"], *sig_cols]]
    pay_cols = cfg["payments"]["columns"]
    payments = events.assign(applicant_id=events["SK_ID_CURR"].map({k: i for i, k in enumerate(ids)}))
    payments = payments[["applicant_id", *pay_cols]].reset_index(drop=True)
    payments.insert(0, "payment_id", np.arange(len(payments)))
    no_tf = af["no_transform_payment_columns"]
    settings = {
        "max_depth": af["max_depth"],
        "agg_primitives": af["agg_primitives"],
        "trans_primitives": af["trans_primitives"],
        "primitive_options": {
            "divide_numeric": {"ignore_columns": {"clients": sig_cols, "payments": no_tf}},
            "natural_logarithm": {"ignore_columns": {"clients": sig_cols, "payments": no_tf}},
        },
    }
    matrix, lineage, n_infinite = run_dfs(clients, payments, client_index="applicant_id", event_index="payment_id",
                              event_key="applicant_id", settings=settings)
    identity = set(clients.columns)
    new_cols = [c for c in matrix.columns if c not in identity]
    lineage_new = [e for e in lineage if e["name"] in new_cols]
    data = pd.concat([data, matrix.loc[data["applicant_id"], new_cols].set_index(data.index)], axis=1)

    out = PROCESSED / "dataset.csv"
    data.to_csv(out, index=False)
    (PROCESSED / "dfs_lineage.json").write_text(json.dumps(lineage_new, indent=1, ensure_ascii=False),
                                                encoding="utf-8")
    manifest = {
        "rows": len(data),
        "columns": data.shape[1],
        "tabular_sha256": cfg["tabular"]["sha256"],
        "text_corpus": {"pool_size": pool_size, "sample_size": n, "seed": seed,
                        "sample_sha256": sha256(ROOT / cfg["text_corpus"]["sample_out"]),
                        "empty_after_cleaning": int(sum(
                            1 for t in data["complaint_text"] if not isinstance(t, str) or not t.strip()))},
        "payments": {"clients": len(ids), "events": len(events),
                     "clients_without_events": int(sum(1 for k in ids if len(seqs[k]) == 0))},
        "signal_report": extractor.report_,
        "dfs": {"input_features": len(af["client_columns"]) + len(sig_cols) + len(pay_cols),
                "output_features": matrix.shape[1], "new_features": len(new_cols),
                "infinite_replaced_by_nan": n_infinite},
        "dataset_sha256": sha256(out),
    }
    (PROCESSED / "build_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

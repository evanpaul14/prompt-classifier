"""Normalize, deduplicate, and validate the unified dataset."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import pandas as pd
from datasketch import MinHash, MinHashLSH


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.lower().split())


def _char_ngrams(text: str, n: int) -> list[str]:
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def _make_minhash(text: str, num_perm: int, n: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for gram in _char_ngrams(text, n):
        m.update(gram.encode("utf-8"))
    return m


def build(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    min_len: int = cfg.get("min_prompt_len", 5)
    max_len: int = cfg.get("max_prompt_len", 20000)
    mh_cfg = cfg.get("minhash", {})
    mh_threshold: float = mh_cfg.get("threshold", 0.8)
    mh_num_perm: int = mh_cfg.get("num_perm", 128)
    mh_n: int = mh_cfg.get("n_gram", 5)
    stats: dict = {}

    df = raw.copy()
    stats["raw_total"] = len(df)

    # 1. Normalize and filter by length
    df["prompt_norm"] = df["prompt"].astype(str).map(_normalize)
    mask = df["prompt_norm"].str.len().between(min_len, max_len)
    stats["dropped_length"] = int((~mask).sum())
    df = df[mask].reset_index(drop=True)

    # 2. Exact-match dedup on normalized prompt
    before = len(df)
    # prefer harmful > jailbreak > safe when deduplicating
    subtype_priority = {"harmful": 0, "jailbreak": 1, "safe": 2}
    df["_priority"] = df["subtype"].map(lambda s: subtype_priority.get(s, 3))
    df = df.sort_values("_priority").drop_duplicates(subset="prompt_norm", keep="first")
    df = df.drop(columns=["_priority"]).reset_index(drop=True)
    stats["dropped_exact_dedup"] = before - len(df)

    # 3. MinHash near-dup pass
    print(f"Running MinHash near-dup (threshold={mh_threshold}, {mh_num_perm} perms)...")
    lsh = MinHashLSH(threshold=mh_threshold, num_perm=mh_num_perm)
    keep_mask = [True] * len(df)
    for i, row in enumerate(df.itertuples(index=False)):
        key = str(i)
        m = _make_minhash(row.prompt_norm, mh_num_perm, mh_n)
        try:
            neighbors = lsh.query(m)
        except Exception:
            neighbors = []
        if neighbors:
            # A near-dup of a previously seen item — drop unless it has higher priority
            keep_mask[i] = False
        else:
            lsh.insert(key, m)
    before = len(df)
    df = df[keep_mask].reset_index(drop=True)
    stats["dropped_minhash_dedup"] = before - len(df)
    print(f"  MinHash dropped {stats['dropped_minhash_dedup']:,} near-dups")

    # 4. Cross-source leak check: safe prompts that match block prompts after normalization
    block_norms = set(df.loc[df["y"] == 1, "prompt_norm"])
    safe_leak = (df["y"] == 0) & df["prompt_norm"].isin(block_norms)
    stats["dropped_leak_check"] = int(safe_leak.sum())
    df = df[~safe_leak].reset_index(drop=True)

    # 5. Per-source stats
    stats["per_source"] = df.groupby(["source", "y"]).size().to_dict()
    stats["per_label"] = df["y"].value_counts().to_dict()
    length = df["prompt"].str.len()
    stats["length_quantiles"] = {
        "p25": float(length.quantile(0.25)),
        "p50": float(length.quantile(0.50)),
        "p75": float(length.quantile(0.75)),
        "p95": float(length.quantile(0.95)),
    }
    stats["total_after_dedup"] = len(df)

    # Clean up helper column before saving
    df = df.drop(columns=["prompt_norm"])

    out_path = Path(cfg["data"]["unified_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    stats_path = Path(cfg["data"]["stats_path"])
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"Saved {len(df):,} rows to {out_path}")
    print(json.dumps({k: v for k, v in stats.items() if k != "per_source"}, indent=2))
    return df

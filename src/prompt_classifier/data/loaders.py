"""One load_<source>() per dataset. Each returns DataFrame[prompt, y, source, subtype]."""
from __future__ import annotations

import pandas as pd
from datasets import load_dataset


def _base_df(prompts: list[str], y: int, source: str, subtype: str) -> pd.DataFrame:
    return pd.DataFrame({
        "prompt": prompts,
        "y": y,
        "source": source,
        "subtype": subtype,
    })


def load_verazuo() -> pd.DataFrame:
    ds = load_dataset("verazuo/jailbreak_llms", split="train")
    texts = [str(r) for r in ds["prompt"] if r]
    return _base_df(texts, 1, "verazuo", "jailbreak")


def load_jackhhao() -> pd.DataFrame:
    ds = load_dataset("jackhhao/jailbreak-classification", split="train")
    rows = []
    for item in ds:
        label_str = str(item.get("type", "")).lower()
        y = 0 if label_str == "benign" else 1
        subtype = "safe" if y == 0 else "jailbreak"
        rows.append({"prompt": str(item["prompt"]), "y": y, "source": "jackhhao", "subtype": subtype})
    return pd.DataFrame(rows)


def load_jailbreakv() -> pd.DataFrame:
    ds = load_dataset(
        "JailbreakV-28K/JailBreakV-28k",
        "JailBreakV_28K",
        split="JailBreakV_28K",
        trust_remote_code=True,
    )
    rows = []
    for item in ds:
        # Keep text-only entries; use the redteam query (final user turn)
        if str(item.get("format", "")).lower() != "text":
            continue
        text = item.get("redteam_query") or item.get("jailbreak_query") or item.get("prompt")
        if text:
            rows.append({"prompt": str(text), "y": 1, "source": "jailbreakv", "subtype": "jailbreak"})
    return pd.DataFrame(rows)


def load_salad() -> pd.DataFrame:
    ds = load_dataset("OpenSafetyLab/Salad-Data", "base_set", split="train", trust_remote_code=True)
    # Try common column names
    col = next((c for c in ["prompt", "question", "instruction"] if c in ds.column_names), None)
    if col is None:
        raise ValueError(f"Cannot find prompt column in Salad-Data. Columns: {ds.column_names}")
    taxonomy_col = next((c for c in ["category", "domain", "task"] if c in ds.column_names), None)
    rows = []
    for item in ds:
        subtype = str(item.get(taxonomy_col, "harmful")) if taxonomy_col else "harmful"
        rows.append({"prompt": str(item[col]), "y": 1, "source": "salad", "subtype": subtype})
    return pd.DataFrame(rows)


def load_advbench() -> pd.DataFrame:
    ds = load_dataset("walledai/AdvBench", split="train")
    col = next((c for c in ["prompt", "goal", "behavior"] if c in ds.column_names), None)
    if col is None:
        raise ValueError(f"Cannot find prompt column in AdvBench. Columns: {ds.column_names}")
    texts = [str(item[col]) for item in ds if item[col]]
    return _base_df(texts, 1, "advbench", "harmful")


def load_harmbench() -> pd.DataFrame:
    ds = load_dataset("walledai/HarmBench", split="train")
    col = next((c for c in ["prompt", "behavior", "instruction"] if c in ds.column_names), None)
    if col is None:
        raise ValueError(f"Cannot find prompt column in HarmBench. Columns: {ds.column_names}")
    texts = [str(item[col]) for item in ds if item[col]]
    return _base_df(texts, 1, "harmbench", "harmful")


def load_benign() -> pd.DataFrame:
    ds = load_dataset("LLM-LAT/benign-dataset", split="train")
    col = next((c for c in ["prompt", "instruction", "input", "text"] if c in ds.column_names), None)
    if col is None:
        raise ValueError(f"Cannot find prompt column in benign-dataset. Columns: {ds.column_names}")
    texts = [str(item[col]) for item in ds if item[col]]
    return _base_df(texts, 0, "benign", "safe")


def load_all(cfg: dict | None = None) -> pd.DataFrame:
    loaders = [
        load_verazuo,
        load_jackhhao,
        load_jailbreakv,
        load_salad,
        load_advbench,
        load_harmbench,
        load_benign,
    ]
    frames = []
    for fn in loaders:
        print(f"Loading {fn.__name__}...")
        try:
            df = fn()
            print(f"  {len(df):,} rows")
            frames.append(df)
        except Exception as e:
            print(f"  WARNING: {fn.__name__} failed: {e}")
    return pd.concat(frames, ignore_index=True)

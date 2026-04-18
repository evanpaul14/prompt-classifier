"""Stratified train/val/test splits and 5-fold CV iterator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def _composite_key(df: pd.DataFrame) -> pd.Series:
    return df["y"].astype(str) + "_" + df["source"]


def _content_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def make_splits(unified: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed: int = cfg.get("seed", 42)
    test_frac: float = cfg.get("test_frac", 0.15)
    val_frac: float = cfg.get("val_frac", 0.15)
    test_prevalence: float = cfg.get("test_block_prevalence", 0.5)

    strat_key = _composite_key(unified)

    # First carve out test set
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(sss_test.split(unified, strat_key))
    trainval = unified.iloc[trainval_idx].reset_index(drop=True)
    test = unified.iloc[test_idx].reset_index(drop=True)

    # Carve val from trainval
    val_frac_adj = val_frac / (1 - test_frac)
    strat_key_tv = _composite_key(trainval)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_adj, random_state=seed + 1)
    train_idx, val_idx = next(sss_val.split(trainval, strat_key_tv))
    train = trainval.iloc[train_idx].reset_index(drop=True)
    val = trainval.iloc[val_idx].reset_index(drop=True)

    # Downsample block side of train to ~1:1
    train = _downsample_to_ratio(train, seed=seed)

    # Fix test prevalence at test_block_prevalence
    test = _fix_prevalence(test, target_block_frac=test_prevalence, seed=seed)

    # Save
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(cfg["data"]["train_path"], index=False)
    val.to_parquet(cfg["data"]["val_path"], index=False)
    test.to_parquet(cfg["data"]["test_path"], index=False)

    manifest = {
        "unified_hash": _content_hash(unified),
        "seed": seed,
        "test_frac": test_frac,
        "val_frac": val_frac,
        "test_block_prevalence": test_prevalence,
        "counts": {
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
            "train_block": int((train["y"] == 1).sum()),
            "train_safe": int((train["y"] == 0).sum()),
            "test_block": int((test["y"] == 1).sum()),
            "test_safe": int((test["y"] == 0).sum()),
        },
    }
    with open(cfg["data"]["manifest_path"], "w") as f:
        json.dump(manifest, f, indent=2)

    print("Splits created:")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        block = (df["y"] == 1).sum()
        safe = (df["y"] == 0).sum()
        print(f"  {name}: {len(df):,} total  |  block={block:,}  safe={safe:,}")

    return train, val, test


def _downsample_to_ratio(df: pd.DataFrame, seed: int = 42, ratio: float = 1.0) -> pd.DataFrame:
    block = df[df["y"] == 1]
    safe = df[df["y"] == 0]
    n_target = int(min(len(block), len(safe)) * ratio)
    rng = np.random.default_rng(seed)
    if len(block) > n_target:
        block = block.iloc[rng.choice(len(block), n_target, replace=False)]
    if len(safe) > n_target:
        safe = safe.iloc[rng.choice(len(safe), n_target, replace=False)]
    return pd.concat([block, safe]).sample(frac=1, random_state=seed).reset_index(drop=True)


def _fix_prevalence(df: pd.DataFrame, target_block_frac: float = 0.5, seed: int = 42) -> pd.DataFrame:
    block = df[df["y"] == 1]
    safe = df[df["y"] == 0]
    n_total = len(df)
    n_block_target = int(round(n_total * target_block_frac))
    n_safe_target = n_total - n_block_target
    rng = np.random.default_rng(seed + 99)
    if len(block) >= n_block_target:
        block = block.iloc[rng.choice(len(block), n_block_target, replace=False)]
    if len(safe) >= n_safe_target:
        safe = safe.iloc[rng.choice(len(safe), n_safe_target, replace=False)]
    return pd.concat([block, safe]).sample(frac=1, random_state=seed).reset_index(drop=True)


def cv_iterator(
    train: pd.DataFrame, val: pd.DataFrame, cfg: dict, n_splits: int = 5
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield (fold_train, fold_val) pairs over train∪val using StratifiedKFold."""
    combined = pd.concat([train, val], ignore_index=True)
    strat_key = _composite_key(combined)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.get("seed", 42))
    for fold_train_idx, fold_val_idx in skf.split(combined, strat_key):
        fold_train = combined.iloc[fold_train_idx].reset_index(drop=True)
        fold_val = combined.iloc[fold_val_idx].reset_index(drop=True)
        # Downsample block in fold_train
        fold_train = _downsample_to_ratio(fold_train, seed=cfg.get("seed", 42))
        yield fold_train, fold_val

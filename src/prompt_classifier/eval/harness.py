"""Evaluate any model against the held-out test set and write a report JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from prompt_classifier.eval.metrics import binary_report


def evaluate(
    model_name: str,
    predict_fn: Callable[[list[str]], tuple[np.ndarray, np.ndarray]],
    cfg: dict,
) -> dict:
    """Load test split, run predict_fn, write reports/<model_name>.json.

    predict_fn(texts) -> (y_pred: ndarray[int], y_prob: ndarray[float])
    """
    test = pd.read_parquet(cfg["data"]["test_path"])
    texts = test["prompt"].tolist()
    y_true = test["y"].values

    print(f"Running inference on {len(texts):,} test examples...")
    y_pred, y_prob = predict_fn(texts)

    source = test["source"].values if "source" in test.columns else None
    report = binary_report(
        y_true, y_pred, y_prob,
        source=source,
        fpr_target=cfg.get("fpr_target", 0.05),
        fnr_target=cfg.get("fnr_target", 0.10),
    )
    report["model"] = model_name

    reports_dir = Path(cfg["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{model_name}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== {model_name} results ===")
    print(f"  F1:        {report['f1']:.4f}")
    print(f"  FPR:       {report['fpr']:.4f}  ({'OK' if report['targets_met']['fpr_ok'] else 'FAIL >'} {cfg.get('fpr_target',0.05)})")
    print(f"  FNR:       {report['fnr']:.4f}  ({'OK' if report['targets_met']['fnr_ok'] else 'FAIL >'} {cfg.get('fnr_target',0.10)})")
    print(f"  AUROC:     {report.get('auroc')}")
    print(f"  Report saved to {report_path}")
    return report

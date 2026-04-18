"""Shared metric function used by all four models."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def binary_report(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    y_prob: list | np.ndarray,
    source: list | np.ndarray | None = None,
    fpr_target: float = 0.05,
    fnr_target: float = 0.10,
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Threshold scan: find the threshold that pins FPR ≤ fpr_target
    threshold_at_fpr, f1_at_fpr, fnr_at_fpr = _threshold_at_fpr(y_true, y_prob, fpr_target)

    report = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "auprc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "threshold_at_fpr_5pct": {
            "threshold": float(threshold_at_fpr),
            "f1": float(f1_at_fpr),
            "fnr": float(fnr_at_fpr),
        },
        "targets_met": {
            "fpr_ok": bool(fpr <= fpr_target),
            "fnr_ok": bool(fnr <= fnr_target),
        },
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

    if source is not None:
        source = np.asarray(source)
        per_source = {}
        for src in np.unique(source):
            mask = source == src
            if mask.sum() == 0:
                continue
            yt, yp, ypr = y_true[mask], y_pred[mask], y_prob[mask]
            tn_s, fp_s, fn_s, tp_s = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            per_source[src] = {
                "n": int(mask.sum()),
                "f1": float(f1_score(yt, yp, zero_division=0)),
                "fpr": float(fp_s / (fp_s + tn_s)) if (fp_s + tn_s) > 0 else 0.0,
                "fnr": float(fn_s / (fn_s + tp_s)) if (fn_s + tp_s) > 0 else 0.0,
            }
        report["per_source"] = per_source

    return report


def _threshold_at_fpr(
    y_true: np.ndarray, y_prob: np.ndarray, fpr_target: float
) -> tuple[float, float, float]:
    thresholds = np.linspace(0, 1, 201)
    best_threshold = 0.5
    best_f1 = 0.0
    best_fnr = 1.0
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t, labels=[0, 1]).ravel()
        fpr_t = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        if fpr_t <= fpr_target:
            f1_t = float(f1_score(y_true, y_pred_t, zero_division=0))
            fnr_t = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = float(t)
                best_fnr = fnr_t
    return best_threshold, best_f1, best_fnr

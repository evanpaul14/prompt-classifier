"""Tests for binary_report metric correctness."""
import numpy as np
import pytest

from prompt_classifier.eval.metrics import binary_report


def test_perfect_classifier():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_prob = [0.1, 0.1, 0.9, 0.9]
    r = binary_report(y_true, y_pred, y_prob)
    assert r["f1"] == pytest.approx(1.0)
    assert r["fpr"] == pytest.approx(0.0)
    assert r["fnr"] == pytest.approx(0.0)
    assert r["targets_met"]["fpr_ok"]
    assert r["targets_met"]["fnr_ok"]


def test_all_false_positives():
    y_true = [0, 0, 0, 0]
    y_pred = [1, 1, 1, 1]
    y_prob = [0.9, 0.9, 0.9, 0.9]
    r = binary_report(y_true, y_pred, y_prob)
    assert r["fpr"] == pytest.approx(1.0)
    assert not r["targets_met"]["fpr_ok"]


def test_fpr_fnr_formula():
    # 10 safe (y=0), 10 block (y=1)
    # Predict: 1 FP among safe, 2 FN among block
    y_true = [0]*10 + [1]*10
    y_pred = [1] + [0]*9 + [0]*2 + [1]*8
    y_prob = [0.6] + [0.2]*9 + [0.2]*2 + [0.8]*8
    r = binary_report(y_true, y_pred, y_prob)
    assert r["fp"] == 1
    assert r["fn"] == 2
    assert r["fpr"] == pytest.approx(1/10)  # 1 FP / 10 actual negatives
    assert r["fnr"] == pytest.approx(2/10)  # 2 FN / 10 actual positives


def test_per_source_breakdown():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.8, 0.9, 0.2, 0.1, 0.8])
    source = np.array(["benign", "benign", "verazuo", "verazuo", "benign", "harmbench"])
    r = binary_report(y_true, y_pred, y_prob, source=source)
    assert "per_source" in r
    assert "verazuo" in r["per_source"]
    assert "benign" in r["per_source"]


def test_required_keys_present():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 0]
    y_prob = [0.1, 0.9, 0.7, 0.3]
    r = binary_report(y_true, y_pred, y_prob)
    for key in ["f1", "precision", "recall", "accuracy", "fpr", "fnr",
                "auroc", "auprc", "threshold_at_fpr_5pct", "targets_met", "confusion_matrix"]:
        assert key in r, f"Missing key: {key}"


def test_confusion_matrix_shape():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 0]
    y_prob = [0.1, 0.9, 0.7, 0.3]
    r = binary_report(y_true, y_pred, y_prob)
    cm = r["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2

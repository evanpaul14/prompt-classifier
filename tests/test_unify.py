"""Tests for normalization, exact dedup, MinHash dedup, and leak check."""
import pandas as pd
import pytest

BASE_CFG = {
    "min_prompt_len": 5,
    "max_prompt_len": 20000,
    "minhash": {"threshold": 0.8, "num_perm": 64, "n_gram": 3},
    "data": {
        "unified_path": "/tmp/test_unified.parquet",
        "stats_path": "/tmp/test_unified_stats.json",
    },
}


def _make_df(rows):
    return pd.DataFrame(rows)


def test_exact_dedup_removes_duplicates():
    from prompt_classifier.data.unify import build
    raw = _make_df([
        {"prompt": "Same prompt", "y": 1, "source": "a", "subtype": "jailbreak"},
        {"prompt": "same prompt", "y": 1, "source": "b", "subtype": "jailbreak"},  # dupe after normalize
        {"prompt": "Different prompt", "y": 0, "source": "c", "subtype": "safe"},
    ])
    result = build(raw, BASE_CFG)
    assert len(result) == 2  # dupe removed


def test_length_filter():
    from prompt_classifier.data.unify import build
    raw = _make_df([
        {"prompt": "ok", "y": 1, "source": "a", "subtype": "jailbreak"},  # too short (len=2)
        {"prompt": "long enough prompt", "y": 1, "source": "a", "subtype": "jailbreak"},
    ])
    result = build(raw, BASE_CFG)
    assert len(result) == 1
    assert result.iloc[0]["prompt"] == "long enough prompt"


def test_leak_check_removes_safe_matching_block():
    from prompt_classifier.data.unify import build
    raw = _make_df([
        {"prompt": "harmful content here", "y": 1, "source": "a", "subtype": "harmful"},
        {"prompt": "Harmful content here", "y": 0, "source": "benign", "subtype": "safe"},  # normalizes to same
    ])
    result = build(raw, BASE_CFG)
    # Safe row should be removed; block row kept
    assert (result["y"] == 1).all()


def test_priority_dedup_keeps_harmful_over_jailbreak():
    from prompt_classifier.data.unify import build
    raw = _make_df([
        {"prompt": "exact same text", "y": 1, "source": "harmbench", "subtype": "harmful"},
        {"prompt": "exact same text", "y": 1, "source": "verazuo", "subtype": "jailbreak"},
    ])
    result = build(raw, BASE_CFG)
    assert len(result) == 1
    assert result.iloc[0]["subtype"] == "harmful"


def test_output_schema():
    from prompt_classifier.data.unify import build
    raw = _make_df([
        {"prompt": "a valid prompt text", "y": 1, "source": "a", "subtype": "jailbreak"},
    ])
    result = build(raw, BASE_CFG)
    assert "prompt" in result.columns
    assert "y" in result.columns
    assert "source" in result.columns
    # prompt_norm should not be in the output
    assert "prompt_norm" not in result.columns

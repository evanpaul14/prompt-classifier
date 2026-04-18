"""Smoke tests for data loaders using mocked HF datasets."""
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


REQUIRED_COLS = {"prompt", "y", "source", "subtype"}


def _mock_dataset(rows: list[dict]):
    ds = MagicMock()
    ds.__iter__ = lambda self: iter(rows)
    ds.column_names = list(rows[0].keys()) if rows else []
    return ds


def test_load_verazuo():
    from prompt_classifier.data.loaders import load_verazuo
    mock_ds = _mock_dataset([{"prompt": "jailbreak prompt"}])
    with patch("prompt_classifier.data.loaders.load_dataset", return_value=mock_ds):
        df = load_verazuo()
    assert set(df.columns) >= REQUIRED_COLS
    assert (df["y"] == 1).all()
    assert (df["subtype"] == "jailbreak").all()
    assert len(df) == 1


def test_load_jackhhao_both_labels():
    from prompt_classifier.data.loaders import load_jackhhao
    mock_ds = _mock_dataset([
        {"prompt": "bad prompt", "type": "jailbreak"},
        {"prompt": "good prompt", "type": "benign"},
    ])
    with patch("prompt_classifier.data.loaders.load_dataset", return_value=mock_ds):
        df = load_jackhhao()
    assert set(df.columns) >= REQUIRED_COLS
    assert set(df["y"].unique()) == {0, 1}
    assert len(df) == 2


def test_load_jailbreakv_text_only():
    from prompt_classifier.data.loaders import load_jailbreakv
    mock_ds = _mock_dataset([
        {"format": "text", "redteam_query": "text jailbreak"},
        {"format": "image", "redteam_query": "image jailbreak"},
    ])
    with patch("prompt_classifier.data.loaders.load_dataset", return_value=mock_ds):
        df = load_jailbreakv()
    # only text row kept
    assert len(df) == 1
    assert df.iloc[0]["prompt"] == "text jailbreak"


def test_load_benign():
    from prompt_classifier.data.loaders import load_benign
    mock_ds = _mock_dataset([{"instruction": "help me with this"}])
    mock_ds.column_names = ["instruction"]
    with patch("prompt_classifier.data.loaders.load_dataset", return_value=mock_ds):
        df = load_benign()
    assert set(df.columns) >= REQUIRED_COLS
    assert (df["y"] == 0).all()
    assert (df["subtype"] == "safe").all()


def test_load_all_partial_failure():
    """load_all should continue even if one source fails."""
    from prompt_classifier.data.loaders import load_all

    def mock_load_dataset(*args, **kwargs):
        raise RuntimeError("dataset unavailable")

    with patch("prompt_classifier.data.loaders.load_dataset", side_effect=mock_load_dataset):
        df = load_all()
    # all failed, result is empty or very small — no crash
    assert isinstance(df, pd.DataFrame)

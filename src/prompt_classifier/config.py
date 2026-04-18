from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Merge _base config if specified
    if "_base" in cfg:
        base_path = Path(cfg.pop("_base"))
        if not base_path.is_absolute():
            base_path = path.parent.parent / base_path
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base

    return cfg

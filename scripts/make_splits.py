"""Create stratified train/val/test splits from unified.parquet."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from prompt_classifier.config import load_config
from prompt_classifier.data.splits import make_splits
from prompt_classifier.seeds import set_all_seeds

cfg = load_config("configs/base.yaml")
set_all_seeds(cfg["seed"])
unified = pd.read_parquet(cfg["data"]["unified_path"])
make_splits(unified, cfg)

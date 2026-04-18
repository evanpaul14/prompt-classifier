"""Download and unify all datasets into data/processed/unified.parquet."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_classifier.config import load_config
from prompt_classifier.data import loaders, unify
from prompt_classifier.seeds import set_all_seeds

cfg = load_config("configs/base.yaml")
set_all_seeds(cfg["seed"])
raw = loaders.load_all(cfg)
unify.build(raw, cfg)

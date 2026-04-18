"""Train a model. Usage: python scripts/train.py --model m1"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["m1", "m2", "m3", "m4"])
args = parser.parse_args()

if args.model == "m1":
    from prompt_classifier.models.m1_tfidf_lr import run
    run("configs/model1_tfidf_lr.yaml")
elif args.model == "m2":
    from prompt_classifier.models.m2_arctic_ffn import run
    run("configs/model2_arctic_ffn.yaml")
elif args.model == "m3":
    from prompt_classifier.models.m3_roberta_frozen import run
    run("configs/model3_roberta_frozen.yaml")
elif args.model == "m4":
    from prompt_classifier.models.m4_roberta_ft import run
    run("configs/model4_roberta_ft.yaml")

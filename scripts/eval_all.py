"""Aggregate reports/*.json into reports/comparison.md."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_classifier.config import load_config

cfg = load_config("configs/base.yaml")
reports_dir = Path(cfg["reports_dir"])

models = ["m1", "m2", "m3", "m4"]
rows = []
for m in models:
    p = reports_dir / f"{m}.json"
    if not p.exists():
        print(f"WARNING: {p} not found, skipping")
        continue
    with open(p) as f:
        r = json.load(f)
    fpr_ok = "✓" if r["targets_met"]["fpr_ok"] else "✗"
    fnr_ok = "✓" if r["targets_met"]["fnr_ok"] else "✗"
    rows.append({
        "Model": m,
        "F1": f"{r['f1']:.4f}",
        "Precision": f"{r['precision']:.4f}",
        "Recall": f"{r['recall']:.4f}",
        "FPR": f"{r['fpr']:.4f} {fpr_ok}",
        "FNR": f"{r['fnr']:.4f} {fnr_ok}",
        "AUROC": f"{r.get('auroc', 'N/A')}",
    })

if not rows:
    print("No reports found. Run training first.")
    sys.exit(1)

headers = list(rows[0].keys())
header_line = "| " + " | ".join(headers) + " |"
sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
data_lines = ["| " + " | ".join(r[h] for h in headers) + " |" for r in rows]

md = "\n".join([header_line, sep_line] + data_lines)
out = reports_dir / "comparison.md"
out.write_text(f"# Model Comparison\n\nTargets: FPR < 5% (✓/✗), FNR < 10% (✓/✗)\n\n{md}\n")
print(f"Saved {out}")
print(md)

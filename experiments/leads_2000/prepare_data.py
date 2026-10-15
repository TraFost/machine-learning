"""Prepare a separate 2,000-lead experiment without changing the first PoC."""

import csv
import json
import random
from pathlib import Path

from prepare_data import generate_leads


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"


def main() -> None:
    if (ROOT / "experiment.json").exists():
        raise SystemExit("This experiment already exists. Keep its data fixed for comparison.")
    # A new seed avoids simply repeating the first 200 generated examples.
    leads = generate_leads(count=2000, seed=43)
    for index, row in enumerate(leads, 1):
        row["lead_id"] = f"E2000-{index:04}"
    random.Random(42).shuffle(leads)
    groups = {"train": leads[:1200], "validation": leads[1200:1600], "test": leads[1600:]}
    ids = [{row["lead_id"] for row in group} for group in groups.values()]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Data groups overlap.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in {"synthetic_leads": leads, **groups}.items():
        with (DATA_DIR / f"{name}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (ROOT / "experiment.json").write_text(
        json.dumps({
            "leads": 2000, "generation_seed": 43, "split_seed": 42,
            "train": 1200, "validation": 400, "test": 400,
            "buying_process": "Same invented process as prepare_data.generate_leads",
            "models": ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting"],
            "settings": "Original model settings. No hyperparameter search.",
            "ranking_checks": [10, 100],
            "main_ranking_check": "top 100 of 400 (same 25% selection share as top 10 of 40)",
            "test_policy": "Reserved for the final selected model. Do not compare all algorithms on test.",
        }, indent=2) + "\n", encoding="utf-8",
    )
    print("Separate experiment: 2,000 synthetic leads.")
    for name in ["train", "validation"]:
        rows = groups[name]
        buyers = sum(row["purchased_within_30_days"] for row in rows)
        print(f"{name}: {len(rows)} leads, {buyers} buyers ({buyers / len(rows):.1%}).")
    print("test: 400 leads reserved. No test outcome summary is shown.")
    print("The original 200-lead data and results remain in place.")


if __name__ == "__main__":
    main()

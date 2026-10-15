"""Check the 50-tree candidate on the already inspected test group."""

import json
from pathlib import Path

from experiments.leads_2000.random_forest.evaluate_test import main as evaluate


def main() -> None:
    evaluate(n_estimators=50, follow_up=True)
    root = Path(__file__).parent / "results"
    original_path = root / "final_test" / "metrics.json"
    if not original_path.exists():
        return
    original = json.loads(original_path.read_text(encoding="utf-8"))
    current = json.loads((root / "follow_up_test_50_trees" / "metrics.json").read_text(encoding="utf-8"))
    if original["test_sha256"] != current["test_sha256"]:
        raise SystemExit("These runs use different test data. Do not compare them as the same group.")
    lines = [
        "# Original and follow-up forest checks", "",
        "The 50-tree setting was chosen from validation results. All other model settings stayed the same.",
        "The test group was already inspected before tuning. This comparison is exploratory, not a new final test.", "",
        "| Trees | Top-100 buyers | Expected buyers with random ties | Brier | Log loss |",
        "|---:|---:|---:|---:|---:|",
    ]
    for trees, metrics in [(100, original), (50, current)]:
        r = metrics["top_100"]
        lines.append(f"| {trees} | {r['displayed_buyers']} | {r['expected_buyers_with_random_ties']:.2f} | "
                     f"{metrics['brier']:.4f} | {metrics['log_loss']:.4f} |")
    lines.extend(["", "The original final_test files remain unchanged.",
                  "Do not keep adjusting settings to improve this reused test result."])
    report = "\n".join(lines) + "\n"
    (root / "follow_up_test_50_trees" / "comparison.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

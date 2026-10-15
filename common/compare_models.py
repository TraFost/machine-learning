"""Compare saved validation predictions. Do not train or read test data."""

import csv
from pathlib import Path

from sklearn.metrics import brier_score_loss, log_loss

from common.data import LABEL, read_leads
from common.ranking import ranking_summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    validation = read_leads("validation")
    expected = {row["lead_id"]: row for row in validation}
    lines = [
        "# Model comparison on the same validation leads",
        "",
        "All models learn from the same training leads and use the same features.",
        "Logistic regression uses its original C=1.0 setting.",
        "Decision tree uses max_depth=3 and min_samples_leaf=10. These are starting settings.",
        "Random forest uses 100 trees, max_depth=3, and min_samples_leaf=10. These are starting settings.",
        "Gradient boosting uses 100 stages, learning_rate=0.05, max_depth=2, and min_samples_leaf=10.",
        "The original models are compared here. Separate tuning reports show the other settings.",
        "",
        "| Model | Displayed top-10 buyers | Expected buyers with random ties | Brier score | Log loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for algorithm, name in [
        ("logistic_regression", "Logistic regression"),
        ("decision_tree", "Decision tree"),
        ("random_forest", "Random forest"),
        ("gradient_boosting", "Gradient-boosted trees"),
    ]:
        path = root / algorithm / "results" / "validation_predictions.csv"
        if not path.exists():
            raise SystemExit(f"First run: uv run python -m {algorithm}.train_model")
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != len(expected) or {row["lead_id"] for row in rows} != set(expected):
            raise SystemExit(f"{name} predictions do not match validation leads.")
        for row in rows:
            if any(row[key] != value for key, value in expected[row["lead_id"]].items()):
                raise SystemExit(f"{name} predictions use different validation data.")
            row["predicted_purchase_probability"] = float(row["predicted_purchase_probability"])
        _, actual, average = ranking_summary(rows)
        labels = [int(row[LABEL]) for row in rows]
        probabilities = [row["predicted_purchase_probability"] for row in rows]
        lines.append(
            f"| {name} | {actual} | {average:.2f} | "
            f"{brier_score_loss(labels, probabilities):.4f} | "
            f"{log_loss(labels, probabilities, labels=[0, 1]):.4f} |"
        )
    lines.extend([
        "",
        "More buyers is better. Lower Brier score and log loss are better.",
        "The displayed top 10 uses lead ID to order equal scores. That order is not learned by the model.",
        "Expected buyers averages possible random choices inside a tie. Use this for a fairer ranking comparison.",
        "The tree can show more buyers in one displayed list without having a better expected result.",
        "Different algorithms do not guarantee better results. This is one small synthetic validation group.",
        "No final model has been selected. The test set was not opened.",
    ])
    report = "\n".join(lines) + "\n"
    output = root / "results" / "model_comparison.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

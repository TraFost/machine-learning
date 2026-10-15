"""Step 6: change only regularization and compare validation results."""

import csv
import json
from hashlib import sha256

from sklearn.metrics import brier_score_loss, log_loss

from logistic_regression.train_model import DATA_DIR, LABEL, RESULTS_DIR, build_model, features, read_leads


from common.ranking import ranking_summary


def main() -> None:
    training, validation = read_leads("train"), read_leads("validation")
    if len(validation) < 10:
        raise SystemExit("This exercise needs at least 10 validation leads.")
    if {row["lead_id"] for row in training} & {row["lead_id"] for row in validation}:
        raise SystemExit("Training and validation groups overlap.")
    y_train = [int(row[LABEL]) for row in training]
    y_validation = [int(row[LABEL]) for row in validation]
    experiment = RESULTS_DIR / "regularization"
    experiment.mkdir(parents=True, exist_ok=True)
    summaries = []

    for c, strength in [(0.1, "Stronger limit"), (1.0, "Original setting"), (10.0, "Weaker limit")]:
        model = build_model(c)
        model.fit(features(training), y_train)
        buyer_column = list(model.named_steps["predictor"].classes_).index(1)
        # Use the same six-decimal precision as the existing saved prediction files.
        probabilities = [
            round(float(p), 6)
            for p in model.predict_proba(features(validation))[:, buyer_column]
        ]
        predicted_rows = [
            {**row, "predicted_purchase_probability": probability}
            for row, probability in zip(validation, probabilities)
        ]
        ranked, actual_buyers, expected_buyers = ranking_summary(predicted_rows)
        folder = experiment / f"C_{str(c).replace('.', 'p')}"
        folder.mkdir(exist_ok=True)
        with (folder / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(predicted_rows[0]))
            writer.writeheader()
            writer.writerows(predicted_rows)
        with (folder / "validation_ranked.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["rank", "selected_top_10", *ranked[0]])
            writer.writeheader()
            for position, row in enumerate(ranked, 1):
                writer.writerow({"rank": position, "selected_top_10": int(position <= 10), **row})
        with (folder / "learned_weights.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["feature", "learned_weight"])
            predictor = model.named_steps["predictor"]
            writer.writerow(["intercept", float(predictor.intercept_[0])])
            writer.writerows(zip(model.named_steps["encode"].get_feature_names_out(), predictor.coef_[0]))
        summary = {
            "C": c,
            "meaning": strength,
            "displayed_top_10_buyers": actual_buyers,
            "expected_top_10_buyers_with_random_ties": expected_buyers,
            "validation_brier": float(brier_score_loss(y_validation, probabilities)),
            "validation_log_loss": float(log_loss(y_validation, probabilities, labels=[0, 1])),
        }
        summaries.append(summary)
        metadata = {
            **summary,
            "model": "LogisticRegression",
            "max_iter": 1000,
            "random_state": 42,
            "train_sha256": sha256((DATA_DIR / "train.csv").read_bytes()).hexdigest(),
            "validation_sha256": sha256((DATA_DIR / "validation.csv").read_bytes()).hexdigest(),
        }
        (folder / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Compare logistic regression settings",
        "",
        "Changed: C, the regularization setting. Original C was 1.0.",
        "Smaller C puts a stronger limit on learned feature weights. Larger C puts a weaker limit on them.",
        "This does not add features, add examples, or switch algorithms.",
        f"Each version learns from the same {len(training)} training leads and predicts for the same {len(validation)} validation leads.",
        "The encoding, solver, iteration limit, and random seed stay the same.",
        "",
        "Main check: expected buyers in the top 10 when tied leads are selected randomly.",
        "Also check probability errors. Lower Brier score and log loss are better.",
        "",
        "| C | Weight limit | Buyers in displayed top 10 | Expected buyers with random ties | Brier score | Log loss |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['C']} | {row['meaning']} | {row['displayed_top_10_buyers']} | "
            f"{row['expected_top_10_buyers_with_random_ties']:.2f} | "
            f"{row['validation_brier']:.4f} | {row['validation_log_loss']:.4f} |"
        )
    lines.extend([
        "",
        "The displayed list breaks ties by lead ID. The expected count averages across possible random choices inside the cutoff tie.",
        "A fractional expected count is an average, not a fraction of a person.",
        "A setting can change probabilities without changing which leads appear near the top.",
        "",
        "Each setting has its own predictions, ranked list, learned weights, and settings file.",
        "The original result files remain in place. No setting has been selected automatically.",
        "Only 40 validation leads are available. Small score differences may not persist on new data.",
        "The test set was not opened. The dataset and split files were not changed.",
        "",
        "Reference: [scikit-learn LogisticRegression: C](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).",
    ])
    report = "\n".join(lines) + "\n"
    output = experiment / "comparison.md"
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

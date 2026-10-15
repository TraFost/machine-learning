"""Change one tree setting at a time. Evaluate only validation leads."""

import csv
import json
from hashlib import sha256

from sklearn.metrics import brier_score_loss, log_loss
from sklearn.tree import export_text

from common.data import DATA_DIR, LABEL, features, read_leads
from common.ranking import ranking_summary
from decision_tree.train_model import RESULTS_DIR, build_model


def main() -> None:
    training, validation = read_leads("train"), read_leads("validation")
    if len(validation) < 10:
        raise SystemExit("This exercise needs at least 10 validation leads.")
    if {row["lead_id"] for row in training} & {row["lead_id"] for row in validation}:
        raise SystemExit("Training and validation groups overlap.")
    train_labels = [int(row[LABEL]) for row in training]
    validation_labels = [int(row[LABEL]) for row in validation]
    experiment = RESULTS_DIR / "hyperparameters"
    experiment.mkdir(parents=True, exist_ok=True)
    # Each change is compared with the original (depth=3, leaf=10).
    candidates = [
        (3, 10, "Original"),
        (2, 10, "Less depth"),
        (5, 10, "More depth"),
        (3, 5, "Smaller end groups"),
        (3, 20, "Larger end groups"),
    ]
    summaries = []
    for depth, leaf, change in candidates:
        model = build_model(max_depth=depth, min_samples_leaf=leaf)
        model.fit(features(training), train_labels)
        predictor = model.named_steps["predictor"]
        buyer_column = list(predictor.classes_).index(1)
        train_probabilities = model.predict_proba(features(training))[:, buyer_column]
        probabilities = [
            round(float(value), 6)
            for value in model.predict_proba(features(validation))[:, buyer_column]
        ]
        rows = [
            {**row, "predicted_purchase_probability": probability}
            for row, probability in zip(validation, probabilities)
        ]
        ranked, found, expected = ranking_summary(rows)
        folder = experiment / f"depth_{depth}_leaf_{leaf}"
        folder.mkdir(exist_ok=True)
        with (folder / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (folder / "validation_ranked.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["rank", "selected_top_10", *ranked[0]])
            writer.writeheader()
            for position, row in enumerate(ranked, 1):
                writer.writerow({"rank": position, "selected_top_10": int(position <= 10), **row})
        (folder / "learned_tree.txt").write_text(
            "Training counts at each end branch are [nonbuyers, buyers].\n"
            "For yes/no fields: <= 0.50 means no; > 0.50 means yes.\n\n"
            + export_text(
                predictor,
                feature_names=list(model.named_steps["encode"].get_feature_names_out()),
                show_weights=True,
                max_depth=depth,
            ),
            encoding="utf-8",
        )
        summary = {
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "change": change,
            "actual_depth": int(predictor.get_depth()),
            "end_groups": int(predictor.get_n_leaves()),
            "displayed_top_10_buyers": found,
            "expected_top_10_buyers_with_random_ties": expected,
            "training_log_loss": float(log_loss(train_labels, train_probabilities, labels=[0, 1])),
            "validation_brier": float(brier_score_loss(validation_labels, probabilities)),
            "validation_log_loss": float(log_loss(validation_labels, probabilities, labels=[0, 1])),
        }
        summaries.append(summary)
        (folder / "settings.json").write_text(
            json.dumps(
                {
                    **summary,
                    "algorithm": "DecisionTreeClassifier",
                    "random_state": 42,
                    "train_sha256": sha256((DATA_DIR / "train.csv").read_bytes()).hexdigest(),
                    "validation_sha256": sha256((DATA_DIR / "validation.csv").read_bytes()).hexdigest(),
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

    lines = [
        "# Decision tree hyperparameter experiment",
        "",
        "A hyperparameter is a setting we choose before training.",
        "",
        "## Changes",
        "",
        "Original: max_depth=3 and min_samples_leaf=10.",
        "",
        "1. Keep the end-group minimum at 10. Try depth limits of 2, 3, and 5.",
        "2. Keep the depth limit at 3. Try end-group minimums of 5, 10, and 20.",
        "",
        "Each candidate changes only one setting from the original.",
        "max_depth limits the number of decisions along a path. It does not force the tree to use all of them.",
        "min_samples_leaf sets the minimum number of training leads in each end group.",
        "A smaller end group allows more specific patterns, but its buyer rate can be less reliable.",
        "",
        f"All candidates use the same {len(training)} training leads, {len(validation)} validation leads, features, encoding, and random seed.",
        "The main ranking check is expected buyers in the top 10 with random selection inside tied scores.",
        "",
        "## Results",
        "",
        "| Depth limit | Minimum end group | Change | Learned depth | End groups | Displayed top-10 buyers | Expected buyers with random ties | Brier | Validation log loss | Training log loss |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['max_depth']} | {row['min_samples_leaf']} | {row['change']} | "
            f"{row['actual_depth']} | {row['end_groups']} | {row['displayed_top_10_buyers']} | "
            f"{row['expected_top_10_buyers_with_random_ties']:.2f} | {row['validation_brier']:.4f} | "
            f"{row['validation_log_loss']:.4f} | {row['training_log_loss']:.4f} |"
        )
    lines.extend([
        "",
        "## How to read this",
        "",
        "More expected buyers is better for the top-10 ranking check.",
        "Lower Brier score and log loss are better for probability errors.",
        "A better training result alone does not show improvement. Check validation results too.",
        "Equal results can mean the settings produced the same tree. A higher limit does not guarantee a larger tree.",
        "The displayed top 10 breaks ties by lead ID. Use expected buyers to avoid rewarding an arbitrary tie order.",
        "The expected buyer count is an average across possible selections, not a fraction of a person.",
        "",
        "Each candidate has its own predictions, ranking, learned tree, and settings file.",
        "The original result files stay in place. No final setting has been selected.",
        "This small validation group has only 6 buyers. Repeated tuning can also fit this group by chance.",
        "The test set was not opened. No data or split files were changed.",
        "",
        "Reference: [decision tree settings](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).",
    ])
    report = "\n".join(lines) + "\n"
    output = experiment / "comparison.md"
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

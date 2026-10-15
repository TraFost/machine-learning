"""Final test of the selected logistic regression model. No tuning here."""

import csv
import json
from hashlib import sha256

import joblib
from sklearn.metrics import brier_score_loss, log_loss

from common.data import DATA_DIR, LABEL, features, read_leads
from common.ranking import ranking_summary
from logistic_regression.train_model import RESULTS_DIR, build_model


def main() -> None:
    # Fixed BEFORE reading test data. Do not choose a different C from test results.
    c = 1.0
    k = 10
    output_dir = RESULTS_DIR / "final_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    training = read_leads("train")
    training_labels = [int(row[LABEL]) for row in training]
    selected = {
        "algorithm": "LogisticRegression",
        "C": c,
        "max_iter": 1000,
        "random_state": 42,
        "training_group": "train.csv only; validation was not added",
        "train_sha256": sha256((DATA_DIR / "train.csv").read_bytes()).hexdigest(),
        "ranking_policy": "top 10 by probability; lead ID breaks displayed ties",
        "main_ranking_check": "expected buyers in top 10 with random selection inside cutoff ties",
        "probability_checks": ["Brier score", "log loss"],
        "selection_reason": "No clear validation ranking benefit from more complex models; keep the simple original model.",
    }
    manifest = output_dir / "selected_model.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != selected:
        raise SystemExit("The final choice or training data changed. Do not reuse this test as a new final check.")
    manifest.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")

    model = build_model(c=c)
    model.fit(features(training), training_labels)
    # Save the complete fitted pipeline, including training-only feature encoding.
    joblib.dump(model, output_dir / "model.joblib")

    # The selected model is now fixed. Open test examples for evaluation only.
    test = read_leads("test")
    train_ids = {row["lead_id"] for row in training}
    validation_ids = {row["lead_id"] for row in read_leads("validation")}
    test_ids = {row["lead_id"] for row in test}
    if train_ids & validation_ids or test_ids & (train_ids | validation_ids):
        raise SystemExit("Data groups overlap. The test would not be separate.")
    if len(test_ids) != len(test) or len(test) < k:
        raise SystemExit("Test data must contain at least 10 unique leads.")

    column = list(model.named_steps["predictor"].classes_).index(1)
    probabilities = [round(float(p), 6) for p in model.predict_proba(features(test))[:, column]]
    labels = [int(row[LABEL]) for row in test]
    rows = [{**row, "predicted_purchase_probability": p} for row, p in zip(test, probabilities)]
    baseline_rate = sum(training_labels) / len(training_labels)
    baseline = [baseline_rate] * len(test)
    ranked, displayed_buyers, expected_buyers = ranking_summary(rows, k=k)
    buyers = sum(labels)
    random_buyers = k * buyers / len(test)

    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "test_ranked.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["rank", "selected_top_10", *ranked[0]])
        writer.writeheader()
        for position, row in enumerate(ranked, 1):
            writer.writerow({"rank": position, "selected_top_10": int(position <= k), **row})

    model_brier = float(brier_score_loss(labels, probabilities))
    model_loss = float(log_loss(labels, probabilities, labels=[0, 1]))
    baseline_brier = float(brier_score_loss(labels, baseline))
    baseline_loss = float(log_loss(labels, baseline, labels=[0, 1]))
    metrics = {
        "test_leads": len(test), "test_buyers": buyers,
        "baseline_purchase_probability": baseline_rate,
        "model_brier": model_brier, "baseline_brier": baseline_brier,
        "model_log_loss": model_loss, "baseline_log_loss": baseline_loss,
        "displayed_top_10_buyers": displayed_buyers,
        "precision_at_10": displayed_buyers / k,
        "recall_at_10": displayed_buyers / buyers if buyers else None,
        "expected_top_10_buyers_with_random_ties": expected_buyers,
        "random_selection_expected_buyers": random_buyers,
        "tie_adjusted_lift_at_10": expected_buyers / random_buyers if random_buyers else None,
        "test_sha256": sha256((DATA_DIR / "test.csv").read_bytes()).hexdigest(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    cutoff = ranked[k - 1]["predicted_purchase_probability"]
    above = [row for row in ranked if row["predicted_purchase_probability"] > cutoff]
    tied = [row for row in ranked if row["predicted_purchase_probability"] == cutoff]
    places = k - len(above)
    lines = [
        "# Final test: logistic regression", "",
        "The model and C=1.0 were selected before test evaluation.",
        f"The model learned again from the same {len(training)} training leads. Validation leads were not added.",
        f"The test contains {len(test)} separate leads and {buyers} buyers.",
        "Test labels were used only to check predictions, not to train or select the model.", "",
        "## Probability errors", "",
        f"The baseline gives every lead the training buyer rate: {baseline_rate:.1%}.",
        "Lower scores are better. These scores are not percentages of correct answers.", "",
        "| Method | Brier score | Log loss |", "|---|---:|---:|",
        f"| Constant baseline | {baseline_brier:.4f} | {baseline_loss:.4f} |",
        f"| Logistic regression, C=1.0 | {model_brier:.4f} | {model_loss:.4f} |", "",
        "## Top-10 ranking", "",
        f"The displayed top 10 contains {displayed_buyers} buyers and {k - displayed_buyers} nonbuyers.",
        f"Precision@10: {displayed_buyers / k:.1%}. This is the buyer rate among the selected leads.",
        f"Buyers found: {displayed_buyers} of {buyers}.",
        f"At the cutoff, {len(tied)} leads share one score and {places} places remain.",
        f"With random selection inside the cutoff tie, expected buyers in the top 10: {expected_buyers:.2f}.",
        f"Randomly choosing 10 of all test leads would find {random_buyers:.2f} buyers on average.",
    ]
    if buyers:
        lines.append(f"Tie-adjusted lift@10: {expected_buyers / random_buyers:.2f} times the random-selection rate.")
    lines.extend([
        "", "## Selected leads", "",
        "| Rank | Lead | Predicted purchase chance | Actually bought? |", "|---:|---|---:|---|",
    ])
    for rank, row in enumerate(ranked[:k], 1):
        lines.append(f"| {rank} | {row['lead_id']} | {row['predicted_purchase_probability']:.1%} | {'Yes' if int(row[LABEL]) else 'No'} |")
    lines.extend([
        "", "## Interpretation", "",
        "This is a final check of the selected model, not a comparison of algorithms on the test set.",
        "Do not tune settings or select another algorithm to improve this test score.",
        "Rerunning the same code with unchanged data repeats the same check; it is not new evidence.",
        "This small synthetic dataset demonstrates the process. It does not establish real sales performance.",
        "The fitted model and encoding are saved in model.joblib. Validation files remain unchanged.",
    ])
    report = "\n".join(lines) + "\n"
    (output_dir / "test_evaluation.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved results: {output_dir}")


if __name__ == "__main__":
    main()

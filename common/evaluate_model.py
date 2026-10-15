"""Step 4: compare saved validation predictions with a simple baseline."""

import csv
from pathlib import Path

from sklearn.metrics import brier_score_loss, log_loss

from common.data import LABEL, read_leads


def main(results_dir: Path, model_name: str) -> None:
    training = read_leads("train")
    validation = read_leads("validation")
    path = results_dir / "validation_predictions.csv"
    if not path.exists():
        raise SystemExit("First run the algorithm training module.")
    with path.open(newline="", encoding="utf-8") as file:
        predictions = list(csv.DictReader(file))

    expected = {lead["lead_id"]: lead for lead in validation}
    if len(predictions) != len(expected) or {
        row["lead_id"] for row in predictions
    } != set(expected):
        raise SystemExit("Predictions do not match validation leads. Run the algorithm training module again.")
    for row in predictions:
        if any(row[key] != value for key, value in expected[row["lead_id"]].items()):
            raise SystemExit("Validation data changed. Run the algorithm training module again.")
    if {lead["lead_id"] for lead in training} & set(expected):
        raise SystemExit("Training and validation groups overlap.")

    labels = [int(row[LABEL]) for row in predictions]
    probabilities = [float(row["predicted_purchase_probability"]) for row in predictions]
    if any(not 0 <= probability <= 1 for probability in probabilities):
        raise SystemExit("Purchase probabilities must be between 0 and 1.")

    # The baseline uses the TRAINING buyer rate, never the validation buyer rate.
    training_buyers = sum(int(row[LABEL]) for row in training)
    baseline_rate = training_buyers / len(training)
    baseline_predictions = [baseline_rate] * len(labels)
    scores = {}
    for name, values in [
        ("Constant baseline", baseline_predictions),
        (model_name, probabilities),
    ]:
        scores[name] = {
            "brier": float(brier_score_loss(labels, values)),
            "log_loss": float(log_loss(labels, values, labels=[0, 1])),
        }

    lines = [
        "# Validation evaluation",
        "",
        f"Training: {len(training)} leads; {training_buyers} buyers ({baseline_rate:.1%}).",
        f"Validation: {len(labels)} leads; {sum(labels)} buyers ({sum(labels) / len(labels):.1%}).",
        "",
        f"Baseline: give every validation lead the same {baseline_rate:.1%} purchase chance.",
        "That chance comes only from the training group.",
        "",
        "## Compare probability errors",
        "",
        "Lower is better for both measures. These are error scores, not percentages of correct answers.",
        "Brier score measures probability errors. Log loss penalizes confident mistakes strongly.",
        "",
        "| Method | Brier score | Log loss |",
        "|---|---:|---:|",
    ]
    for name, score in scores.items():
        lines.append(f"| {name} | {score['brier']:.4f} | {score['log_loss']:.4f} |")

    wins = sum(
        scores[model_name][metric] < scores["Constant baseline"][metric]
        for metric in ["brier", "log_loss"]
    )
    conclusion = {
        2: "The model beats the baseline on both probability-error measures in this validation group.",
        1: "The measures disagree: the model beats the baseline on one probability-error measure.",
        0: "The model does not beat the baseline on either probability-error measure in this validation group.",
    }[wins]
    lines.extend(["", conclusion, "", "## Check predicted chances in groups", "",
                  "| Predicted chance | Leads | Mean predicted chance | Actual buyer rate |",
                  "|---|---:|---:|---:|"])
    for lower, upper in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        members = [
            (p, y) for p, y in zip(probabilities, labels)
            if lower <= p and (p < upper or upper == 1.0)
        ]
        if members:
            predicted = sum(p for p, _ in members) / len(members)
            observed = sum(y for _, y in members) / len(members)
            lines.append(f"| {lower:.0%} to {'including ' if upper == 1 else 'below '}{upper:.0%} | {len(members)} | {predicted:.1%} | {observed:.1%} |")
        else:
            lines.append(f"| {lower:.0%} to {upper:.0%} | 0 | No leads | No leads |")

    lines.extend([
        "",
        "Compare the last two columns. A higher predicted chance than actual buyer rate suggests overestimation in that group.",
        "Groups with few leads give unstable estimates. These bins are a descriptive check, not proof of calibration.",
        "",
        "## What this means",
        "",
        "This checks probability errors. It does not yet measure the quality of a top-leads calling list.",
        "It does not tell us whether more training data or a different algorithm will help.",
        "Only 40 validation leads are available in the current exercise. Do not treat this one result as a general verdict on this algorithm.",
        "The test set was not opened or used. No model was trained or changed by this script.",
        "",
        "References: [probability errors and calibration](https://scikit-learn.org/stable/modules/calibration.html), "
        "[Brier score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html), "
        "[log loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).",
    ])
    report = "\n".join(lines) + "\n"
    output = results_dir / "validation_evaluation.md"
    output.write_text(report, encoding="utf-8")

    baseline_dir = Path(__file__).resolve().parents[1] / "results" / "constant_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    with (baseline_dir / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[*validation[0], "predicted_purchase_probability"])
        writer.writeheader()
        for row in validation:
            writer.writerow({**row, "predicted_purchase_probability": baseline_rate})

    print(report)
    print(f"Saved report: {output}")



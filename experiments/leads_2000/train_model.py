"""Run one algorithm on the larger experiment's training and validation groups."""

import argparse
import csv
import json
from hashlib import sha256
from pathlib import Path

from sklearn.metrics import brier_score_loss, log_loss

from common.data import LABEL, features
from common.ranking import ranking_summary
from decision_tree.train_model import build_model as decision_tree
from gradient_boosting.train_model import build_model as gradient_boosting
from logistic_regression.train_model import build_model as logistic_regression
from random_forest.train_model import build_model as random_forest


ROOT = Path(__file__).parent
BUILDERS = {
    "logistic_regression": logistic_regression,
    "decision_tree": decision_tree,
    "random_forest": random_forest,
    "gradient_boosting": gradient_boosting,
}


def read_group(name: str) -> list[dict]:
    path = ROOT / "data" / f"{name}.csv"
    if not path.exists():
        raise SystemExit("First run: uv run python -m experiments.leads_2000.prepare_data")
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_comparison() -> None:
    collected = []
    for algorithm in BUILDERS:
        path = ROOT / algorithm / "results" / "metrics.json"
        if path.exists():
            collected.append(json.loads(path.read_text(encoding="utf-8")))
    lines = [
        "# Larger experiment: validation comparison", "",
        "Each model learned from the same 1,200 training leads and predicted for the same 400 validation leads.",
        "Only original settings are used. Models are run one at a time.",
        "Top 100 selects 25% of validation leads, as top 10 did in the first experiment.",
        "Top 10 is also shown, but now selects only 2.5% of leads.", "",
        "| Model | Brier score | Log loss | Expected top-10 buyers with random ties | Expected top-100 buyers with random ties | Lift@100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in collected:
        lines.append(f"| {row['algorithm']} | {row['brier']:.4f} | {row['log_loss']:.4f} | "
                     f"{row['top_10']['expected_buyers']:.2f} | {row['top_100']['expected_buyers']:.2f} | {row['top_100']['lift']:.2f} |")
    if collected:
        row = collected[0]
        lines.extend(["", f"Constant training-rate baseline: Brier {row['baseline_brier']:.4f}; log loss {row['baseline_log_loss']:.4f}."])
    lines.extend([
        "", "Lower probability errors are better. More buyers and higher lift are better for ranking.",
        "Expected buyer counts average possible selections within a score tie. They are not literal fractional people.",
        "A new random sample is used. Differences from the first experiment reflect sample size and sample composition.",
        "This is not a controlled learning curve and does not prove that more data caused a specific improvement.",
        "The buying process remains synthetic. Results do not establish real sales performance.",
        ("Random forest was selected for final test evaluation. See random_forest/results/final_test/test_evaluation.md."
         if (ROOT / "random_forest" / "results" / "final_test" / "test_evaluation.md").exists()
         else "The test group stays reserved. No final model has been selected in this larger experiment."),
    ])
    (ROOT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=list(BUILDERS))
    algorithm = parser.parse_args().algorithm
    training, validation = read_group("train"), read_group("validation")
    if {r["lead_id"] for r in training} & {r["lead_id"] for r in validation}:
        raise SystemExit("Training and validation groups overlap.")
    model = BUILDERS[algorithm]()
    model.fit(features(training), [int(row[LABEL]) for row in training])
    predictor = model.named_steps["predictor"]
    column = list(predictor.classes_).index(1)
    probabilities = [round(float(p), 6) for p in model.predict_proba(features(validation))[:, column]]
    rows = [{**row, "predicted_purchase_probability": p} for row, p in zip(validation, probabilities)]
    labels = [int(row[LABEL]) for row in rows]
    rate = sum(int(row[LABEL]) for row in training) / len(training)
    output = ROOT / algorithm / "results"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {
        "algorithm": algorithm,
        "train_leads": len(training), "validation_leads": len(validation),
        "validation_buyers": sum(labels),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "baseline_probability": rate,
        "baseline_brier": float(brier_score_loss(labels, [rate] * len(labels))),
        "baseline_log_loss": float(log_loss(labels, [rate] * len(labels), labels=[0, 1])),
        "hyperparameters": predictor.get_params(),
        **{f"{name}_sha256": sha256((ROOT / "data" / f"{name}.csv").read_bytes()).hexdigest() for name in ["train", "validation"]},
    }
    for k in [10, 100]:
        ranked, found, expected = ranking_summary(rows, k=k)
        random_buyers = k * sum(labels) / len(labels)
        metrics[f"top_{k}"] = {
            "displayed_buyers": found, "displayed_precision": found / k,
            "expected_buyers": expected, "expected_precision": expected / k,
            "random_expected_buyers": random_buyers,
            "lift": expected / random_buyers if random_buyers else None,
        }
        with (output / f"validation_ranked_top_{k}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["rank", f"selected_top_{k}", *ranked[0]])
            writer.writeheader()
            for position, row in enumerate(ranked, 1):
                writer.writerow({"rank": position, f"selected_top_{k}": int(position <= k), **row})
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report = [
        f"# {algorithm}: 2,000-lead experiment", "",
        f"Learned from {len(training)} training leads. Predicted for {len(validation)} validation leads.",
        "Only training features and labels were used to fit the model. Original hyperparameters were used.",
        f"Validation buyers: {sum(labels)} ({sum(labels) / len(labels):.1%}).", "",
        "| Method | Brier score | Log loss |", "|---|---:|---:|",
        f"| Constant baseline | {metrics['baseline_brier']:.4f} | {metrics['baseline_log_loss']:.4f} |",
        f"| {algorithm} | {metrics['brier']:.4f} | {metrics['log_loss']:.4f} |", "",
        "| Selected group | Displayed buyers | Expected buyers with random ties | Random selection: expected buyers | Lift |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in [10, 100]:
        r = metrics[f"top_{k}"]
        report.append(f"| Top {k} | {r['displayed_buyers']} | {r['expected_buyers']:.2f} | {r['random_expected_buyers']:.2f} | {r['lift']:.2f} |")
    report.extend(["", "Score ties use lead ID for display. Expected counts average random selection inside cutoff ties.",
                   "Top 100 is the main ranking comparison because it selects the same 25% share as the first experiment.",
                   "The test set was not opened. No model or setting has been chosen from test results."])
    text = "\n".join(report) + "\n"
    (output / "validation_evaluation.md").write_text(text, encoding="utf-8")
    write_comparison()
    print(text)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

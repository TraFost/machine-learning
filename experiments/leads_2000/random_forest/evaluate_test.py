"""Final test of the selected random forest in the 2,000-lead experiment."""

import csv
import json
from hashlib import sha256
from pathlib import Path

import joblib
from sklearn.metrics import brier_score_loss, log_loss

from common.data import LABEL, features
from common.ranking import ranking_summary
from experiments.leads_2000.train_model import ROOT, read_group
from random_forest.train_model import build_model


def main(
    *, n_estimators: int = 100, follow_up: bool = False,
    max_depth: int = 3, test_group: str = "test", fresh_test: bool = False,
) -> None:
    # Fix the model and selection policy before opening test data.
    if fresh_test and (follow_up or n_estimators != 100 or max_depth != 5 or test_group != "fresh_test_seed44"):
        raise SystemExit("The fresh test is fixed to 100 trees, depth 5, and generation seed 44.")
    if not fresh_test and (max_depth != 3 or test_group != "test"):
        raise SystemExit("Do not change the original test configuration.")
    if not fresh_test and not follow_up and n_estimators != 100:
        raise SystemExit("Keep the original final test fixed. Use a separate follow-up run.")
    model = build_model(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=10)
    training = read_group("train")
    y_train = [int(row[LABEL]) for row in training]
    folder_name = f"follow_up_test_{n_estimators}_trees" if follow_up else "final_test"
    if fresh_test:
        folder_name = "fresh_test_depth_5"
    output = Path(__file__).parent / "results" / folder_name
    output.mkdir(parents=True, exist_ok=True)
    selection = {
        "algorithm": "RandomForestClassifier",
        "hyperparameters": model.named_steps["predictor"].get_params(),
        "training_group": "train.csv only; validation leads are not added",
        "train_sha256": sha256((ROOT / "data" / "train.csv").read_bytes()).hexdigest(),
        "main_ranking_check": "top 100 of 400; random selection inside cutoff ties",
        "additional_ranking_check": "top 10",
        "selection_reason": (
            "Follow-up: 50 trees had the lowest validation probability errors among the tested forest settings."
            if follow_up else
            "User selected random forest after validation comparison; original settings remain fixed."
        ),
    }
    if follow_up:
        selection["evaluation_status"] = "Exploratory reuse of an already inspected test group; not a new independent final test."
    if fresh_test:
        selection["selection_reason"] = "User requested a depth increase to 5, fixed before generating fresh synthetic examples."
        selection["fresh_test"] = {"generation_seed": 44, "leads": 400, "group": test_group}
    manifest = output / "selected_model.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != selection:
        raise SystemExit("The selected model or training data changed. Do not reuse this as a new final test.")
    manifest.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    model.fit(features(training), y_train)
    joblib.dump(model, output / "model.joblib")

    if fresh_test:
        from prepare_data import generate_leads

        # The model settings are recorded and fitted before creating new outcomes.
        fresh_path = ROOT / "data" / f"{test_group}.csv"
        generated = generate_leads(count=400, seed=44)
        for index, row in enumerate(generated, 1):
            row["lead_id"] = f"FRESH44-{index:04}"
        if not fresh_path.exists():
            with fresh_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(generated[0]))
                writer.writeheader()
                writer.writerows(generated)
        else:
            existing = read_group(test_group)
            expected = [{key: str(value) for key, value in row.items()} for row in generated]
            if existing != expected:
                raise SystemExit("The fresh test file changed. Keep the generated examples fixed.")
    test = read_group(test_group)
    validation_ids = {row["lead_id"] for row in read_group("validation")}
    train_ids = {row["lead_id"] for row in training}
    test_ids = {row["lead_id"] for row in test}
    if test_ids & (train_ids | validation_ids) or train_ids & validation_ids:
        raise SystemExit("Data groups overlap.")
    if len(test) != 400 or len(test_ids) != 400:
        raise SystemExit("Expected 400 unique test leads.")
    column = list(model.named_steps["predictor"].classes_).index(1)
    probabilities = [round(float(p), 6) for p in model.predict_proba(features(test))[:, column]]
    rows = [{**row, "predicted_purchase_probability": p} for row, p in zip(test, probabilities)]
    labels = [int(row[LABEL]) for row in test]
    rate = sum(y_train) / len(y_train)
    baseline = [rate] * len(test)
    metrics = {
        "test_leads": len(test), "test_buyers": sum(labels),
        "baseline_probability": rate,
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "baseline_brier": float(brier_score_loss(labels, baseline)),
        "baseline_log_loss": float(log_loss(labels, baseline, labels=[0, 1])),
        "test_sha256": sha256((ROOT / "data" / f"{test_group}.csv").read_bytes()).hexdigest(),
    }
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for k in [10, 100]:
        ranked, found, expected = ranking_summary(rows, k=k)
        random_buyers = k * sum(labels) / len(labels)
        metrics[f"top_{k}"] = {
            "displayed_buyers": found, "precision": found / k,
            "recall": found / sum(labels) if sum(labels) else None,
            "expected_buyers_with_random_ties": expected,
            "random_selection_expected_buyers": random_buyers,
            "tie_adjusted_lift": expected / random_buyers if random_buyers else None,
        }
        with (output / f"test_ranked_top_{k}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["rank", f"selected_top_{k}", *ranked[0]])
            writer.writeheader()
            for rank, row in enumerate(ranked, 1):
                writer.writerow({"rank": rank, f"selected_top_{k}": int(rank <= k), **row})
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    lines = [
        ("# Fresh synthetic test: random forest, depth 5" if fresh_test else
         "# Follow-up test: random forest, 2,000-lead experiment" if follow_up
         else "# Final test: random forest, 2,000-lead experiment"), "",
        ("This run reuses test data that we already inspected. It is not a new independent final check."
         if follow_up else "Random forest and its settings were selected before test evaluation."),
        f"Settings: {n_estimators} trees, maximum depth {max_depth}, minimum end group 10, random seed 42.",
        "Row sampling and feature sampling are the same as the original forest.",
        f"Learned from {len(training)} training leads. Validation leads were not added.",
        f"Test group: {len(test)} leads; {sum(labels)} buyers ({sum(labels) / len(labels):.1%}).",
        "Test outcomes were used to check predictions, not to teach or choose the model.", "",
        "## Probability errors", "", "Lower is better.", "",
        "| Method | Brier score | Log loss |", "|---|---:|---:|",
        f"| Constant training-rate baseline | {metrics['baseline_brier']:.4f} | {metrics['baseline_log_loss']:.4f} |",
        f"| Random forest | {metrics['brier']:.4f} | {metrics['log_loss']:.4f} |", "",
        "## Ranking", "",
        "Top 100 is the main check. It selects 25% of the test group.",
        "Displayed lists use lead ID to order equal scores. Expected counts average random selection inside cutoff ties.", "",
        "| Group | Buyers in displayed list | Precision | Recall | Expected buyers with random ties | Random selection: expected buyers | Lift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in [10, 100]:
        r = metrics[f"top_{k}"]
        recall = f"{r['recall']:.1%}" if r["recall"] is not None else "Undefined"
        lift = f"{r['tie_adjusted_lift']:.2f}" if r["tie_adjusted_lift"] is not None else "Undefined"
        lines.append(f"| Top {k} | {r['displayed_buyers']} | {r['precision']:.1%} | {recall} | "
                     f"{r['expected_buyers_with_random_ties']:.2f} | {r['random_selection_expected_buyers']:.2f} | {lift} |")
    lines.extend([
        "", "## Limits", "",
        "This is a check of the chosen forest. It does not establish which algorithm is best on the test group.",
        "Do not select settings or a different algorithm to improve this test result.",
        "Repeating the same run is not new evidence.",
        "This synthetic experiment demonstrates the process. It does not prove real sales performance.",
        "The fitted pipeline is saved in model.joblib. Validation results remain unchanged.",
    ])
    if fresh_test:
        lines.extend([
            "", "## Fresh data and the depth change", "",
            "400 new examples were generated with seed 44 after the depth-5 configuration was fixed.",
            "The buying process and feature definitions are unchanged. The earlier test files remain in place.",
            "The generation seed 44 is separate from the model's random seed 42.",
            "This model still learns only from the same 1,200 training leads.",
            "Do not compare this score directly with the old test score to claim that more depth helped.",
            "The model depth and test examples both changed. A depth comparison needs the same evaluation examples.",
            "No seed or configuration was selected from these fresh outcomes.",
        ])
    text = "\n".join(lines) + "\n"
    (output / "test_evaluation.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

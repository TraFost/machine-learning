"""Compare a small set of hyperparameters on validation leads only."""

import csv
import json
from hashlib import sha256

from sklearn.metrics import brier_score_loss, log_loss

from common.data import DATA_DIR, LABEL, features, read_leads
from common.ranking import ranking_summary


def run_experiment(
    name, results_dir, build_model, defaults, changes, explanations,
    *, data_dir=None, k=10, notes=(),
):
    source_dir = DATA_DIR if data_dir is None else data_dir
    if data_dir is None:
        training, validation = read_leads("train"), read_leads("validation")
    else:
        groups = []
        for group in ["train", "validation"]:
            with (source_dir / f"{group}.csv").open(newline="", encoding="utf-8") as file:
                groups.append(list(csv.DictReader(file)))
        training, validation = groups
    if len(validation) < k:
        raise SystemExit(f"At least {k} validation leads are required.")
    if {r["lead_id"] for r in training} & {r["lead_id"] for r in validation}:
        raise SystemExit("Training and validation groups overlap.")
    y_train = [int(r[LABEL]) for r in training]
    y_validation = [int(r[LABEL]) for r in validation]
    output_dir = results_dir / "hyperparameters"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [("original", defaults.copy())]
    for setting, value in changes:
        candidates.append((f"{setting}_{str(value).replace('.', 'p')}", {**defaults, setting: value}))
    summaries = []
    for run, settings in candidates:
        model = build_model(**settings)
        model.fit(features(training), y_train)
        predictor = model.named_steps["predictor"]
        column = list(predictor.classes_).index(1)
        train_p = model.predict_proba(features(training))[:, column]
        probabilities = [round(float(p), 6) for p in model.predict_proba(features(validation))[:, column]]
        rows = [{**r, "predicted_purchase_probability": p} for r, p in zip(validation, probabilities)]
        ranked, displayed, expected = ranking_summary(rows, k=k)
        folder = output_dir / run
        folder.mkdir(exist_ok=True)
        with (folder / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (folder / "validation_ranked.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["rank", f"selected_top_{k}", *ranked[0]])
            writer.writeheader()
            for position, r in enumerate(ranked, 1):
                writer.writerow({"rank": position, f"selected_top_{k}": int(position <= k), **r})
        summary = {
            "run": run, **settings,
            f"displayed_top_{k}_buyers": displayed,
            f"expected_top_{k}_buyers_with_random_ties": expected,
            "validation_brier": float(brier_score_loss(y_validation, probabilities)),
            "validation_log_loss": float(log_loss(y_validation, probabilities, labels=[0, 1])),
            "training_log_loss": float(log_loss(y_train, train_p, labels=[0, 1])),
        }
        summaries.append(summary)
        metadata = {
            "algorithm": name, "hyperparameters": predictor.get_params(), **summary,
            **{f"{g}_sha256": sha256((source_dir / f"{g}.csv").read_bytes()).hexdigest() for g in ["train", "validation"]},
        }
        (folder / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        f"# {name}: hyperparameter comparison", "",
        "Each run changes one setting from the original. Other settings stay fixed.",
        "The features, training leads, validation leads, encoding, and seed stay the same.", "",
        "## What the settings mean", "", *[f"- {text}" for text in explanations], "",
        "## Results", "",
        f"Main check: expected buyers in the top {k} with random selection inside ties.",
        "Lower Brier score and log loss are better for probability errors.", "",
        "| Run | " + " | ".join(defaults) + " | Displayed buyers | Expected buyers with random ties | Brier | Validation log loss | Training log loss |",
        "|---|" + "---:|" * (len(defaults) + 5),
    ]
    for row in summaries:
        lines.append(
            "| " + row["run"] + " | " + " | ".join(str(row[key]) for key in defaults)
            + f" | {row[f'displayed_top_{k}_buyers']} | {row[f'expected_top_{k}_buyers_with_random_ties']:.2f}"
            + f" | {row['validation_brier']:.4f} | {row['validation_log_loss']:.4f} | {row['training_log_loss']:.4f} |"
        )
    lines.extend([
        "", "The displayed list uses lead ID to break ties. The expected count avoids rewarding that arbitrary order.",
        "A fractional expected count is an average across possible selections.",
        "A lower training error alone does not show improvement. Check validation results too.",
        "We changed one setting at a time. This does not test all combinations. Settings can interact.",
        "Each run has its own folder. Original results remain in place. No final model was selected.",
        f"This run uses {len(training)} training leads and {len(validation)} validation leads. Validation has {sum(y_validation)} buyers.",
        "Repeated comparisons can fit chance patterns in this group. Keep the search small.",
        "Use a separate final test after selecting a model. Use cross-validation for larger studies.",
        "The test set was not opened. Input data and splits were not changed.", "",
        *notes, "",
        "References: [random forest settings](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)"
        " and [gradient boosting settings](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).",
    ])
    report = "\n".join(lines) + "\n"
    path = output_dir / "comparison.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {path}")

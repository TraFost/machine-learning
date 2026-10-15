"""Train a gradient-boosted trees using the existing training leads."""

import csv
import json
from hashlib import sha256
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

from common.data import DATA_DIR, LABEL, features, read_leads


RESULTS_DIR = Path(__file__).parent / "results"


def build_model(n_estimators: int = 100, learning_rate: float = 0.05, max_depth: int = 2) -> Pipeline:
    return Pipeline(
        [
            ("encode", DictVectorizer(sparse=False)),
            (
                "predictor",
                GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=10,
                    learning_rate=learning_rate,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    training, validation = read_leads("train"), read_leads("validation")
    if {row["lead_id"] for row in training} & {row["lead_id"] for row in validation}:
        raise SystemExit("Training and validation groups overlap.")

    model = build_model()
    model.fit(features(training), [int(row[LABEL]) for row in training])
    booster = model.named_steps["predictor"]
    buyer_column = list(booster.classes_).index(1)
    probabilities = model.predict_proba(features(validation))[:, buyer_column]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "validation_predictions.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[*validation[0], "predicted_purchase_probability"])
        writer.writeheader()
        for row, probability in zip(validation, probabilities):
            writer.writerow({**row, "predicted_purchase_probability": round(float(probability), 6)})

    settings = {
        "algorithm": "GradientBoostingClassifier",
        "n_estimators": 100,
        "max_depth": 2,
        "min_samples_leaf": 10,
        "learning_rate": 0.05,
        "random_state": 42,
        "training_leads": len(training),
        "validation_leads": len(validation),
        "train_sha256": sha256((DATA_DIR / "train.csv").read_bytes()).hexdigest(),
        "validation_sha256": sha256((DATA_DIR / "validation.csv").read_bytes()).hexdigest(),
    }
    (RESULTS_DIR / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    (RESULTS_DIR / "how_it_works.md").write_text(
        "# How gradient-boosted trees learn\n\n"
        "Start with a prediction based on the training buyer rate.\n"
        "Add a small tree that helps reduce the current training errors.\n"
        "Update the combined prediction. Repeat for 100 stages.\n\n"
        "The small trees supply corrections. They are not independent purchase classifiers.\n"
        "The final probability comes from the combined model, not an average vote.\n"
        "The learning rate limits each new tree's contribution.\n\n"
        "Settings: 100 stages, learning_rate=0.05, max_depth=2, min_samples_leaf=10.\n"
        "The same training leads are used at every stage in this version.\n"
        "Only training outcomes guide learning. Validation outcomes are not used to build the trees.\n"
        "No early stopping or hyperparameter search was used.\n"
        "More stages do not guarantee better results on new leads.\n"
        "The test set was not opened.\n\n"
        "Reference: [scikit-learn gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).\n",
        encoding="utf-8",
    )

    print("Model: gradient-boosted trees")
    print("Built 100 stages. Each new tree helps reduce the current training errors.")
    print("Settings: learning_rate=0.05; max_depth=2; min_samples_leaf=10.")
    print("The learning rate limits how much each new tree changes the combined prediction.")
    print("These are starting settings. They have not been tuned.")
    print(f"Learned from {len(training)} training leads.")
    print(f"Predicted purchase chances for {len(validation)} validation leads.")
    print("The test set was not opened.")
    print(f"Saved: {output}")

if __name__ == "__main__":
    main()

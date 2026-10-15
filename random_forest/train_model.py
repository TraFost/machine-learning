"""Train a random forest using the existing training leads."""

import csv
import json
from hashlib import sha256
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

from common.data import DATA_DIR, LABEL, features, read_leads


RESULTS_DIR = Path(__file__).parent / "results"


def build_model(n_estimators: int = 100, max_depth: int = 3, min_samples_leaf: int = 10) -> Pipeline:
    return Pipeline(
        [
            ("encode", DictVectorizer(sparse=False)),
            (
                "predictor",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features="sqrt",
                    bootstrap=True,
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
    forest = model.named_steps["predictor"]
    buyer_column = list(forest.classes_).index(1)
    probabilities = model.predict_proba(features(validation))[:, buyer_column]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "validation_predictions.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[*validation[0], "predicted_purchase_probability"])
        writer.writeheader()
        for row, probability in zip(validation, probabilities):
            writer.writerow({**row, "predicted_purchase_probability": round(float(probability), 6)})

    settings = {
        "algorithm": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 3,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": 42,
        "training_leads": len(training),
        "validation_leads": len(validation),
        "train_sha256": sha256((DATA_DIR / "train.csv").read_bytes()).hexdigest(),
        "validation_sha256": sha256((DATA_DIR / "validation.csv").read_bytes()).hexdigest(),
    }
    (RESULTS_DIR / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    (RESULTS_DIR / "how_it_works.md").write_text(
        "# How this random forest learns\n\n"
        "We use the same 120 training leads and the same three input fields.\n"
        "Encoding converts purchase intent into numeric columns.\n\n"
        "1. Build 100 trees.\n"
        "2. Give each tree a random sample of training rows. A row can appear more than once in a sample.\n"
        "3. At each split, consider a random subset of encoded features.\n"
        "4. Each tree predicts a purchase probability.\n"
        "5. Average the probabilities from all 100 trees.\n\n"
        "Sampling does not create new customer evidence. It reuses the same training leads.\n"
        "The trees have a depth limit of 3 and an end-group minimum of 10.\n"
        "The forest can reduce reliance on one tree. It does not guarantee improvement.\n"
        "These are starting settings. No hyperparameter search was run.\n"
        "The test set was not opened.\n\n"
        "Reference: [scikit-learn random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).\n",
        encoding="utf-8",
    )

    print("Model: random forest")
    print("Built 100 trees and combined their purchase probabilities.")
    print("Each tree has a depth limit of 3 and an end-group minimum of 10.")
    print("Trees receive varied training samples and consider different feature subsets.")
    print("These samples reuse existing leads; they do not add new evidence.")
    print(f"Learned from {len(training)} training leads.")
    print(f"Predicted purchase chances for {len(validation)} validation leads.")
    print("The test set was not opened.")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

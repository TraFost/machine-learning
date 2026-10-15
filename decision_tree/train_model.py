"""Learn a small decision tree from the same training leads."""

import csv
import json
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from common.data import LABEL, features, read_leads


RESULTS_DIR = Path(__file__).parent / "results"


def build_model(max_depth: int = 3, min_samples_leaf: int = 10) -> Pipeline:
    return Pipeline(
        [
            ("encode", DictVectorizer(sparse=False)),
            (
                "predictor",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
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
    predictor = model.named_steps["predictor"]
    buyer_column = list(predictor.classes_).index(1)
    probabilities = model.predict_proba(features(validation))[:, buyer_column]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "validation_predictions.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[*validation[0], "predicted_purchase_probability"])
        writer.writeheader()
        for row, probability in zip(validation, probabilities):
            writer.writerow({**row, "predicted_purchase_probability": round(float(probability), 6)})

    tree = export_text(
        predictor,
        feature_names=list(model.named_steps["encode"].get_feature_names_out()),
        show_weights=True,
    )
    (RESULTS_DIR / "learned_tree.txt").write_text(
        "LEARNED DECISION TREE\n\n"
        "A category field is encoded as a yes/no input.\n"
        "For a yes/no field: <= 0.50 means no; > 0.50 means yes.\n"
        "weights: [nonbuyers, buyers] counts TRAINING leads at each end branch.\n"
        "The buyer share at that end branch supplies the purchase probability.\n"
        "These are training counts, not validation outcomes.\n\n" + tree,
        encoding="utf-8",
    )
    (RESULTS_DIR / "settings.json").write_text(
        json.dumps(
            {"algorithm": "DecisionTreeClassifier", "max_depth": 3, "min_samples_leaf": 10, "random_state": 42},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("Model: decision tree")
    print("The model learns branching questions from training examples.")
    print("Settings: max_depth=3; min_samples_leaf=10; random_state=42.")
    print("The tree can use at most three decisions along each path.")
    print("Each end branch must contain at least 10 training leads.")
    print("These are starting settings. They have not been tuned.")
    print(f"Learned from {len(training)} training leads.")
    print(f"Predicted purchase chances for {len(validation)} validation leads.")
    print("Features and data splits are the same as for logistic regression.")
    print("The test set was not opened.")
    print(f"Saved predictions: {output}")
    print(f"Saved learned branches: {RESULTS_DIR / 'learned_tree.txt'}")


if __name__ == "__main__":
    main()

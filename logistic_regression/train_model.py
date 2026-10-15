"""Step 3: learn from training leads and inspect validation predictions."""

import csv
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


from common.data import DATA_DIR, LABEL, features, read_leads

RESULTS_DIR = Path(__file__).parent / "results"


def build_model(c: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            # Converts category names into numeric inputs the model can use.
            ("encode", DictVectorizer(sparse=False)),
            ("predictor", LogisticRegression(C=c, max_iter=1000, random_state=42)),
        ]
    )


def main() -> None:
    training = read_leads("train")
    validation = read_leads("validation")
    if {lead["lead_id"] for lead in training} & {
        lead["lead_id"] for lead in validation
    }:
        raise SystemExit("Training and validation leads must be separate.")

    model = build_model()

    # Learning: give the algorithm training features AND known outcomes.
    model.fit(features(training), [int(lead[LABEL]) for lead in training])

    # Prediction: supply validation features WITHOUT outcomes.
    # The model keeps the values learned during training; this does not retrain it.
    buyer_column = list(model.named_steps["predictor"].classes_).index(1)
    probabilities = model.predict_proba(features(validation))[:, buyer_column]

    print("STEP 3: train one model, then inspect its predictions\n")
    print("Model: logistic regression")
    print(f"Learned from {len(training)} training leads and their purchase outcomes.")
    print(f"Predicted purchase chances for {len(validation)} separate validation leads.")
    print("The model did not receive their purchase outcomes when predicting.\n")
    print(f"{'Lead':<8} {'Pricing?':<10} {'Demo?':<8} {'Intent':<9} {'Predicted chance':<19} {'Actually bought?'}")
    for lead, probability in list(zip(validation, probabilities))[:12]:
        print(
            f"{lead['lead_id']:<8} "
            f"{('yes' if int(lead['pricing_interest']) else 'no'):<10} "
            f"{('yes' if int(lead['demo_requested']) else 'no'):<8} "
            f"{lead['purchase_intent']:<9} "
            f"{probability:>6.1%}             "
            f"{'yes' if int(lead[LABEL]) else 'no'}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "validation_predictions.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=[*validation[0], "predicted_purchase_probability"]
        )
        writer.writeheader()
        for lead, probability in zip(validation, probabilities):
            writer.writerow({**lead, "predicted_purchase_probability": round(float(probability), 6)})

    print("\nThe predicted chance is an estimate, not a guarantee.")
    print("An individual outcome cannot tell us whether that probability was accurate.")
    print("We have not compared models or measured overall prediction quality yet.")
    print("The test set was not opened or used.")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

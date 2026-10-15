"""Follow-up forest tuning using larger training and validation groups only."""

from pathlib import Path

from common.tuning import run_experiment
from random_forest.train_model import build_model


def main() -> None:
    experiment = Path(__file__).resolve().parents[1]
    run_experiment(
        name="Random forest: 2,000-lead follow-up",
        results_dir=Path(__file__).parent / "results",
        build_model=build_model,
        defaults={"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 10},
        changes=[
            ("n_estimators", 50), ("n_estimators", 200),
            ("max_depth", 2), ("max_depth", 5),
            ("min_samples_leaf", 5), ("min_samples_leaf", 20),
        ],
        explanations=[
            "n_estimators: number of trees. Try 50, 100, and 200.",
            "max_depth: maximum decisions along a tree path. Try 2, 3, and 5.",
            "min_samples_leaf: minimum training examples per end branch. Try 5, 10, and 20.",
            "Row sampling, feature sampling, and random seed remain fixed.",
        ],
        data_dir=experiment / "data",
        k=100,
        notes=[
            "## Follow-up after the first test", "",
            "We already inspected the original forest's test result before starting this tuning exercise.",
            "This report is exploratory validation analysis. It is not a new final test.",
            "The original final_test folder and saved model remain unchanged.",
            "This script does not read test.csv or use test scores to select settings.",
            "A tuned model has no new independent final result in this experiment.",
        ],
    )


if __name__ == "__main__":
    main()

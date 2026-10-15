"""Change one random forest setting at a time."""

from common.tuning import run_experiment
from random_forest.train_model import RESULTS_DIR, build_model


def main():
    run_experiment(
        name="Random forest", results_dir=RESULTS_DIR, build_model=build_model,
        defaults={"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 10},
        changes=[("n_estimators", 50), ("n_estimators", 200), ("max_depth", 2),
                 ("max_depth", 5), ("min_samples_leaf", 5), ("min_samples_leaf", 20)],
        explanations=[
            "n_estimators: number of trees. More trees can make averaging more stable, but require more work.",
            "max_depth: maximum decisions along a path. More depth allows more detailed patterns.",
            "min_samples_leaf: minimum training examples per end branch. Larger groups limit specific patterns.",
            "Feature sampling, row sampling, and random seed stay at their original settings.",
        ],
    )


if __name__ == "__main__":
    main()

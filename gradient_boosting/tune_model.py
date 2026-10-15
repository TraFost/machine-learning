"""Change one gradient boosting setting at a time."""

from common.tuning import run_experiment
from gradient_boosting.train_model import RESULTS_DIR, build_model


def main():
    run_experiment(
        name="Gradient-boosted trees", results_dir=RESULTS_DIR, build_model=build_model,
        defaults={"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2},
        changes=[("n_estimators", 50), ("n_estimators", 200), ("learning_rate", 0.02),
                 ("learning_rate", 0.1), ("max_depth", 1), ("max_depth", 3)],
        explanations=[
            "n_estimators: number of stages. More stages allow more updates, but can fit training details too closely.",
            "learning_rate: how much each new tree contributes. A smaller value makes each update smaller.",
            "max_depth: maximum decisions in each new tree. More depth allows more complex corrections.",
            "Stage count and learning rate work together. We vary them separately to inspect their effects.",
            "The minimum end group stays at 10. The random seed stays at 42.",
        ],
    )


if __name__ == "__main__":
    main()

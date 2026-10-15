"""Fix depth 5, then generate and evaluate 400 fresh synthetic test leads."""

from experiments.leads_2000.random_forest.evaluate_test import main


if __name__ == "__main__":
    main(n_estimators=100, max_depth=5, test_group="fresh_test_seed44", fresh_test=True)

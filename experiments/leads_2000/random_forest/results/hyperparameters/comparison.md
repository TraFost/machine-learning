# Random forest: 2,000-lead follow-up: hyperparameter comparison

Each run changes one setting from the original. Other settings stay fixed.
The features, training leads, validation leads, encoding, and seed stay the same.

## What the settings mean

- n_estimators: number of trees. Try 50, 100, and 200.
- max_depth: maximum decisions along a tree path. Try 2, 3, and 5.
- min_samples_leaf: minimum training examples per end branch. Try 5, 10, and 20.
- Row sampling, feature sampling, and random seed remain fixed.

## Results

Main check: expected buyers in the top 100 with random selection inside ties.
Lower Brier score and log loss are better for probability errors.

| Run | n_estimators | max_depth | min_samples_leaf | Displayed buyers | Expected buyers with random ties | Brier | Validation log loss | Training log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 100 | 3 | 10 | 47 | 47.03 | 0.1547 | 0.4774 | 0.4781 |
| n_estimators_50 | 50 | 3 | 10 | 47 | 47.03 | 0.1545 | 0.4772 | 0.4787 |
| n_estimators_200 | 200 | 3 | 10 | 47 | 47.03 | 0.1547 | 0.4775 | 0.4782 |
| max_depth_2 | 100 | 2 | 10 | 47 | 47.03 | 0.1575 | 0.4857 | 0.4875 |
| max_depth_5 | 100 | 5 | 10 | 47 | 47.03 | 0.1549 | 0.4780 | 0.4766 |
| min_samples_leaf_5 | 100 | 3 | 5 | 47 | 47.03 | 0.1547 | 0.4774 | 0.4781 |
| min_samples_leaf_20 | 100 | 3 | 20 | 47 | 47.03 | 0.1547 | 0.4775 | 0.4782 |

The displayed list uses lead ID to break ties. The expected count avoids rewarding that arbitrary order.
A fractional expected count is an average across possible selections.
A lower training error alone does not show improvement. Check validation results too.
We changed one setting at a time. This does not test all combinations. Settings can interact.
Each run has its own folder. Original results remain in place. No final model was selected.
This run uses 1200 training leads and 400 validation leads. Validation has 95 buyers.
Repeated comparisons can fit chance patterns in this group. Keep the search small.
Use a separate final test after selecting a model. Use cross-validation for larger studies.
The test set was not opened. Input data and splits were not changed.

## Follow-up after the first test

We already inspected the original forest's test result before starting this tuning exercise.
This report is exploratory validation analysis. It is not a new final test.
The original final_test folder and saved model remain unchanged.
This script does not read test.csv or use test scores to select settings.
A tuned model has no new independent final result in this experiment.

References: [random forest settings](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and [gradient boosting settings](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).

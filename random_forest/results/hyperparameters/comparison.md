# Random forest: hyperparameter comparison

Each run changes one setting from the original. Other settings stay fixed.
The features, training leads, validation leads, encoding, and seed stay the same.

## What the settings mean

- n_estimators: number of trees. More trees can make averaging more stable, but require more work.
- max_depth: maximum decisions along a path. More depth allows more detailed patterns.
- min_samples_leaf: minimum training examples per end branch. Larger groups limit specific patterns.
- Feature sampling, row sampling, and random seed stay at their original settings.

## Results

Main check: expected buyers in the top 10 with random selection inside ties.
Lower Brier score and log loss are better for probability errors.

| Run | n_estimators | max_depth | min_samples_leaf | Displayed buyers | Expected buyers with random ties | Brier | Validation log loss | Training log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 100 | 3 | 10 | 3 | 2.71 | 0.1402 | 0.4464 | 0.5153 |
| n_estimators_50 | 50 | 3 | 10 | 3 | 2.71 | 0.1442 | 0.4574 | 0.5167 |
| n_estimators_200 | 200 | 3 | 10 | 3 | 2.71 | 0.1428 | 0.4527 | 0.5135 |
| max_depth_2 | 100 | 2 | 10 | 3 | 2.71 | 0.1367 | 0.4406 | 0.5260 |
| max_depth_5 | 100 | 5 | 10 | 3 | 2.71 | 0.1409 | 0.4481 | 0.5133 |
| min_samples_leaf_5 | 100 | 3 | 5 | 3 | 3.86 | 0.1410 | 0.4461 | 0.5018 |
| min_samples_leaf_20 | 100 | 3 | 20 | 3 | 3.86 | 0.1366 | 0.4494 | 0.5733 |

The displayed list uses lead ID to break ties. The expected count avoids rewarding that arbitrary order.
A fractional expected count is an average across possible selections.
A lower training error alone does not show improvement. Check validation results too.
We changed one setting at a time. This does not test all combinations. Settings can interact.
Each run has its own folder. Original results remain in place. No final model was selected.
Only 40 validation leads and 6 buyers are available in this exercise.
Repeated comparisons can fit chance patterns in this group. Keep the search small.
Use a separate final test after selecting a model. Use cross-validation for larger studies.
The test set was not opened. Input data and splits were not changed.

References: [random forest settings](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and [gradient boosting settings](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).

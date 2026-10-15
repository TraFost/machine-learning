# Gradient-boosted trees: hyperparameter comparison

Each run changes one setting from the original. Other settings stay fixed.
The features, training leads, validation leads, encoding, and seed stay the same.

## What the settings mean

- n_estimators: number of stages. More stages allow more updates, but can fit training details too closely.
- learning_rate: how much each new tree contributes. A smaller value makes each update smaller.
- max_depth: maximum decisions in each new tree. More depth allows more complex corrections.
- Stage count and learning rate work together. We vary them separately to inspect their effects.
- The minimum end group stays at 10. The random seed stays at 42.

## Results

Main check: expected buyers in the top 10 with random selection inside ties.
Lower Brier score and log loss are better for probability errors.

| Run | n_estimators | learning_rate | max_depth | Displayed buyers | Expected buyers with random ties | Brier | Validation log loss | Training log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 100 | 0.05 | 2 | 3 | 2.71 | 0.1425 | 0.4487 | 0.4991 |
| n_estimators_50 | 50 | 0.05 | 2 | 3 | 3.40 | 0.1438 | 0.4537 | 0.5192 |
| n_estimators_200 | 200 | 0.05 | 2 | 3 | 2.71 | 0.1434 | 0.4480 | 0.4885 |
| learning_rate_0p02 | 100 | 0.02 | 2 | 4 | 3.57 | 0.1449 | 0.4574 | 0.5326 |
| learning_rate_0p1 | 100 | 0.1 | 2 | 3 | 2.71 | 0.1434 | 0.4477 | 0.4875 |
| max_depth_1 | 100 | 0.05 | 1 | 3 | 3.86 | 0.1431 | 0.4511 | 0.5299 |
| max_depth_3 | 100 | 0.05 | 3 | 3 | 2.71 | 0.1446 | 0.4503 | 0.4877 |

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

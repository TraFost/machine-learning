# Decision tree hyperparameter experiment

A hyperparameter is a setting we choose before training.

## Changes

Original: max_depth=3 and min_samples_leaf=10.

1. Keep the end-group minimum at 10. Try depth limits of 2, 3, and 5.
2. Keep the depth limit at 3. Try end-group minimums of 5, 10, and 20.

Each candidate changes only one setting from the original.
max_depth limits the number of decisions along a path. It does not force the tree to use all of them.
min_samples_leaf sets the minimum number of training leads in each end group.
A smaller end group allows more specific patterns, but its buyer rate can be less reliable.

All candidates use the same 120 training leads, 40 validation leads, features, encoding, and random seed.
The main ranking check is expected buyers in the top 10 with random selection inside tied scores.

## Results

| Depth limit | Minimum end group | Change | Learned depth | End groups | Displayed top-10 buyers | Expected buyers with random ties | Brier | Validation log loss | Training log loss |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 10 | Original | 3 | 5 | 4 | 3.57 | 0.1532 | 0.4672 | 0.5172 |
| 2 | 10 | Less depth | 2 | 3 | 4 | 3.57 | 0.1505 | 0.4652 | 0.5319 |
| 5 | 10 | More depth | 4 | 6 | 4 | 3.57 | 0.1564 | 0.4772 | 0.5092 |
| 3 | 5 | Smaller end groups | 3 | 6 | 3 | 3.40 | 0.1503 | 0.4670 | 0.5055 |
| 3 | 20 | Larger end groups | 3 | 4 | 4 | 3.57 | 0.1552 | 0.4742 | 0.5224 |

## How to read this

More expected buyers is better for the top-10 ranking check.
Lower Brier score and log loss are better for probability errors.
A better training result alone does not show improvement. Check validation results too.
Equal results can mean the settings produced the same tree. A higher limit does not guarantee a larger tree.
The displayed top 10 breaks ties by lead ID. Use expected buyers to avoid rewarding an arbitrary tie order.
The expected buyer count is an average across possible selections, not a fraction of a person.

Each candidate has its own predictions, ranking, learned tree, and settings file.
The original result files stay in place. No final setting has been selected.
This small validation group has only 6 buyers. Repeated tuning can also fit this group by chance.
The test set was not opened. No data or split files were changed.

Reference: [decision tree settings](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

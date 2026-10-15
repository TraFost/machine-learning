# Compare logistic regression settings

Changed: C, the regularization setting. Original C was 1.0.
Smaller C puts a stronger limit on learned feature weights. Larger C puts a weaker limit on them.
This does not add features, add examples, or switch algorithms.
Each version learns from the same 120 training leads and predicts for the same 40 validation leads.
The encoding, solver, iteration limit, and random seed stay the same.

Main check: expected buyers in the top 10 when tied leads are selected randomly.
Also check probability errors. Lower Brier score and log loss are better.

| C | Weight limit | Buyers in displayed top 10 | Expected buyers with random ties | Brier score | Log loss |
|---|---|---:|---:|---:|---:|
| 0.1 | Stronger limit | 3 | 3.86 | 0.1343 | 0.4441 |
| 1.0 | Original setting | 3 | 3.86 | 0.1373 | 0.4399 |
| 10.0 | Weaker limit | 3 | 3.86 | 0.1412 | 0.4451 |

The displayed list breaks ties by lead ID. The expected count averages across possible random choices inside the cutoff tie.
A fractional expected count is an average, not a fraction of a person.
A setting can change probabilities without changing which leads appear near the top.

Each setting has its own predictions, ranked list, learned weights, and settings file.
The original result files remain in place. No setting has been selected automatically.
Only 40 validation leads are available. Small score differences may not persist on new data.
The test set was not opened. The dataset and split files were not changed.

Reference: [scikit-learn LogisticRegression: C](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

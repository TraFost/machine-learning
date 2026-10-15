# Follow-up test: random forest, 2,000-lead experiment

This run reuses test data that we already inspected. It is not a new independent final check.
Settings: 50 trees, maximum depth 3, minimum end group 10, random seed 42.
Row sampling and feature sampling are the same as the original forest.
Learned from 1200 training leads. Validation leads were not added.
Test group: 400 leads; 87 buyers (21.8%).
Test outcomes were used to check predictions, not to teach or choose the model.

## Probability errors

Lower is better.

| Method | Brier score | Log loss |
|---|---:|---:|
| Constant training-rate baseline | 0.1704 | 0.5244 |
| Random forest | 0.1453 | 0.4534 |

## Ranking

Top 100 is the main check. It selects 25% of the test group.
Displayed lists use lead ID to order equal scores. Expected counts average random selection inside cutoff ties.

| Group | Buyers in displayed list | Precision | Recall | Expected buyers with random ties | Random selection: expected buyers | Lift |
|---|---:|---:|---:|---:|---:|---:|
| Top 10 | 7 | 70.0% | 8.0% | 7.27 | 2.17 | 3.34 |
| Top 100 | 41 | 41.0% | 47.1% | 42.16 | 21.75 | 1.94 |

## Limits

This is a check of the chosen forest. It does not establish which algorithm is best on the test group.
Do not select settings or a different algorithm to improve this test result.
Repeating the same run is not new evidence.
This synthetic experiment demonstrates the process. It does not prove real sales performance.
The fitted pipeline is saved in model.joblib. Validation results remain unchanged.

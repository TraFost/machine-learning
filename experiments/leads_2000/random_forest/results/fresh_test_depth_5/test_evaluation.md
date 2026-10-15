# Fresh synthetic test: random forest, depth 5

Random forest and its settings were selected before test evaluation.
Settings: 100 trees, maximum depth 5, minimum end group 10, random seed 42.
Row sampling and feature sampling are the same as the original forest.
Learned from 1200 training leads. Validation leads were not added.
Test group: 400 leads; 91 buyers (22.8%).
Test outcomes were used to check predictions, not to teach or choose the model.

## Probability errors

Lower is better.

| Method | Brier score | Log loss |
|---|---:|---:|
| Constant training-rate baseline | 0.1758 | 0.5363 |
| Random forest | 0.1503 | 0.4715 |

## Ranking

Top 100 is the main check. It selects 25% of the test group.
Displayed lists use lead ID to order equal scores. Expected counts average random selection inside cutoff ties.

| Group | Buyers in displayed list | Precision | Recall | Expected buyers with random ties | Random selection: expected buyers | Lift |
|---|---:|---:|---:|---:|---:|---:|
| Top 10 | 10 | 100.0% | 11.0% | 8.67 | 2.27 | 3.81 |
| Top 100 | 42 | 42.0% | 46.2% | 42.00 | 22.75 | 1.85 |

## Limits

This is a check of the chosen forest. It does not establish which algorithm is best on the test group.
Do not select settings or a different algorithm to improve this test result.
Repeating the same run is not new evidence.
This synthetic experiment demonstrates the process. It does not prove real sales performance.
The fitted pipeline is saved in model.joblib. Validation results remain unchanged.

## Fresh data and the depth change

400 new examples were generated with seed 44 after the depth-5 configuration was fixed.
The buying process and feature definitions are unchanged. The earlier test files remain in place.
The generation seed 44 is separate from the model's random seed 42.
This model still learns only from the same 1,200 training leads.
Do not compare this score directly with the old test score to claim that more depth helped.
The model depth and test examples both changed. A depth comparison needs the same evaluation examples.
No seed or configuration was selected from these fresh outcomes.

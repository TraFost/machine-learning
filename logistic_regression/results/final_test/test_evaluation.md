# Final test: logistic regression

The model and C=1.0 were selected before test evaluation.
The model learned again from the same 120 training leads. Validation leads were not added.
The test contains 40 separate leads and 8 buyers.
Test labels were used only to check predictions, not to train or select the model.

## Probability errors

The baseline gives every lead the training buyer rate: 29.2%.
Lower scores are better. These scores are not percentages of correct answers.

| Method | Brier score | Log loss |
|---|---:|---:|
| Constant baseline | 0.1684 | 0.5223 |
| Logistic regression, C=1.0 | 0.1546 | 0.4796 |

## Top-10 ranking

The displayed top 10 contains 4 buyers and 6 nonbuyers.
Precision@10: 40.0%. This is the buyer rate among the selected leads.
Buyers found: 4 of 8.
At the cutoff, 6 leads share one score and 6 places remain.
With random selection inside the cutoff tie, expected buyers in the top 10: 4.00.
Randomly choosing 10 of all test leads would find 2.00 buyers on average.
Tie-adjusted lift@10: 2.00 times the random-selection rate.

## Selected leads

| Rank | Lead | Predicted purchase chance | Actually bought? |
|---:|---|---:|---|
| 1 | L-029 | 65.7% | No |
| 2 | L-072 | 60.6% | Yes |
| 3 | L-087 | 60.6% | Yes |
| 4 | L-189 | 60.6% | No |
| 5 | L-002 | 55.8% | Yes |
| 6 | L-008 | 55.8% | No |
| 7 | L-051 | 55.8% | Yes |
| 8 | L-088 | 55.8% | No |
| 9 | L-164 | 55.8% | No |
| 10 | L-198 | 55.8% | No |

## Interpretation

This is a final check of the selected model, not a comparison of algorithms on the test set.
Do not tune settings or select another algorithm to improve this test score.
Rerunning the same code with unchanged data repeats the same check; it is not new evidence.
This small synthetic dataset demonstrates the process. It does not establish real sales performance.
The fitted model and encoding are saved in model.joblib. Validation files remain unchanged.

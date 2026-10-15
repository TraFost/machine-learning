# Larger experiment: validation comparison

Each model learned from the same 1,200 training leads and predicted for the same 400 validation leads.
Only original settings are used. Models are run one at a time.
Top 100 selects 25% of validation leads, as top 10 did in the first experiment.
Top 10 is also shown, but now selects only 2.5% of leads.

| Model | Brier score | Log loss | Expected top-10 buyers with random ties | Expected top-100 buyers with random ties | Lift@100 |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.1575 | 0.4849 | 6.50 | 45.35 | 1.91 |
| decision_tree | 0.1550 | 0.4799 | 7.00 | 46.58 | 1.96 |
| random_forest | 0.1547 | 0.4774 | 7.00 | 47.03 | 1.98 |
| gradient_boosting | 0.1551 | 0.4789 | 7.00 | 47.03 | 1.98 |

Constant training-rate baseline: Brier 0.1811; log loss 0.5483.

Lower probability errors are better. More buyers and higher lift are better for ranking.
Expected buyer counts average possible selections within a score tie. They are not literal fractional people.
A new random sample is used. Differences from the first experiment reflect sample size and sample composition.
This is not a controlled learning curve and does not prove that more data caused a specific improvement.
The buying process remains synthetic. Results do not establish real sales performance.
Random forest was selected for final test evaluation. See random_forest/results/final_test/test_evaluation.md.

# random_forest: 2,000-lead experiment

Learned from 1200 training leads. Predicted for 400 validation leads.
Only training features and labels were used to fit the model. Original hyperparameters were used.
Validation buyers: 95 (23.8%).

| Method | Brier score | Log loss |
|---|---:|---:|
| Constant baseline | 0.1811 | 0.5483 |
| random_forest | 0.1547 | 0.4774 |

| Selected group | Displayed buyers | Expected buyers with random ties | Random selection: expected buyers | Lift |
|---|---:|---:|---:|---:|
| Top 10 | 7 | 7.00 | 2.38 | 2.95 |
| Top 100 | 47 | 47.03 | 23.75 | 1.98 |

Score ties use lead ID for display. Expected counts average random selection inside cutoff ties.
Top 100 is the main ranking comparison because it selects the same 25% share as the first experiment.
The test set was not opened. No model or setting has been chosen from test results.

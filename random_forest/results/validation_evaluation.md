# Validation evaluation

Training: 120 leads; 35 buyers (29.2%).
Validation: 40 leads; 6 buyers (15.0%).

Baseline: give every validation lead the same 29.2% purchase chance.
That chance comes only from the training group.

## Compare probability errors

Lower is better for both measures. These are error scores, not percentages of correct answers.
Brier score measures probability errors. Log loss penalizes confident mistakes strongly.

| Method | Brier score | Log loss |
|---|---:|---:|
| Constant baseline | 0.1476 | 0.4779 |
| Random forest | 0.1402 | 0.4464 |

The model beats the baseline on both probability-error measures in this validation group.

## Check predicted chances in groups

| Predicted chance | Leads | Mean predicted chance | Actual buyer rate |
|---|---:|---:|---:|
| 0% to below 20% | 9 | 17.0% | 0.0% |
| 20% to below 40% | 17 | 25.0% | 5.9% |
| 40% to below 60% | 14 | 55.1% | 35.7% |
| 60% to 80% | 0 | No leads | No leads |
| 80% to 100% | 0 | No leads | No leads |

Compare the last two columns. A higher predicted chance than actual buyer rate suggests overestimation in that group.
Groups with few leads give unstable estimates. These bins are a descriptive check, not proof of calibration.

## What this means

This checks probability errors. It does not yet measure the quality of a top-leads calling list.
It does not tell us whether more training data or a different algorithm will help.
Only 40 validation leads are available in the current exercise. Do not treat this one result as a general verdict on this algorithm.
The test set was not opened or used. No model was trained or changed by this script.

References: [probability errors and calibration](https://scikit-learn.org/stable/modules/calibration.html), [Brier score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html), [log loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).

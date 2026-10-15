# Lead scoring learning lab

## Question

After the first conversation, will a lead purchase within 30 days?

We use 200 synthetic leads. The data is invented and includes chance. All outcome
windows are assumed complete. Synthetic results do not prove real business value.

## Folder layout

```text
poc/
  data/                        Shared examples and split files
  prepare_data.py              Create synthetic examples
  split_data.py                Separate examples before training
  common/                      Shared inputs and evaluation code
  logistic_regression/
    train_model.py
    tune_model.py
    evaluate_model.py
    evaluate_ranking.py
    results/                   Predictions, reports, regularization experiments
  decision_tree/
    train_model.py
    evaluate_model.py
    evaluate_ranking.py
    results/                   Predictions, reports, and learned_tree.txt
  random_forest/
    train_model.py
    evaluate_model.py
    evaluate_ranking.py
    results/                   Predictions, reports, and settings
  gradient_boosting/
    train_model.py
    evaluate_model.py
    evaluate_ranking.py
    results/                   Predictions, reports, and settings
  results/                     Shared baseline and model comparison report
```

Code and results for each algorithm are in that algorithm's folder. Shared data
stays in one place so we can compare algorithms on the same examples.

Run commands from the `poc` folder. Use `-m` for modules inside algorithm folders.
The old root-level training and evaluation commands have been replaced.

## Data fields

| Field | Role |
|---|---|
| lead_id | Tracking only; not a feature |
| pricing_interest | Feature from the first conversation |
| demo_requested | Feature from the first conversation |
| purchase_intent | Feature from the first conversation |
| purchased_within_30_days | Label observed afterward |

For yes/no fields, 1 means yes and 0 means no.

## Prepare data (already done)

```powershell
uv run python prepare_data.py
uv run python split_data.py
```

The split has 120 training leads, 40 validation leads, and 40 test leads.
Only the training group teaches the model. Validation helps us choose. Test is the
final check after our choices. Do not regenerate or change splits to improve scores.

## Logistic regression

```powershell
uv run python -m logistic_regression.train_model
uv run python -m logistic_regression.evaluate_model
uv run python -m logistic_regression.evaluate_ranking
uv run python -m logistic_regression.tune_model
```

Logistic regression learns feature weights and returns a purchase probability.
The original setting is C=1.0. The tuning exercise tries C=0.1, 1.0, and 10.0.
Smaller C puts a stronger limit on feature weights. Each setting has a subfolder
inside `logistic_regression/results/regularization/`.

## Decision tree

```powershell
uv run python -m decision_tree.train_model
uv run python -m decision_tree.evaluate_model
uv run python -m decision_tree.evaluate_ranking
```

The tree learns branching questions from the same training data.

- max_depth=3: at most three decisions along each path.
- min_samples_leaf=10: each end branch must have at least 10 training leads.
- random_state=42: a fixed seed for repeatable training.

These are starting settings. They have not been tuned. The tree uses the same
feature encoding as logistic regression. Encoding is learned only from training.
Read `decision_tree/results/learned_tree.txt` to inspect the learned branches.

### Try decision tree hyperparameters

```powershell
uv run python -m decision_tree.tune_model
```

The script changes one setting at a time from the original tree:

- Keep `min_samples_leaf=10`; try `max_depth=2`, `3`, and `5`.
- Keep `max_depth=3`; try `min_samples_leaf=5`, `10`, and `20`.

All candidates use the same training and validation data. Read
`decision_tree/results/hyperparameters/comparison.md`. Each setting has its own
folder with predictions, ranking, learned tree, and settings. The original tree
results remain in place. No final setting is selected automatically.

## Random forest

```powershell
uv run python -m random_forest.train_model
uv run python -m random_forest.evaluate_model
uv run python -m random_forest.evaluate_ranking
```

A random forest combines predictions from many trees. This first version uses:

- `n_estimators=100`: build 100 trees.
- `max_depth=3`: limit decisions along each tree path.
- `min_samples_leaf=10`: require at least 10 training examples per end branch.
- `max_features="sqrt"`: consider a subset of encoded features at each split.
- `bootstrap=True`: train each tree with a random sample of existing training rows.
- `random_state=42`: keep the run repeatable.

These are starting settings. The forest uses the same data and input fields as
the earlier models. Sampling training leads does not create new evidence.
Read `random_forest/results/how_it_works.md` for the learning steps.

## Gradient-boosted trees (current exercise)

```powershell
uv run python -m gradient_boosting.train_model
uv run python -m gradient_boosting.evaluate_model
uv run python -m gradient_boosting.evaluate_ranking
```

This model builds trees in stages. Each new tree helps reduce the current training
errors. The final prediction combines these corrections. It is not an average of
independent tree probabilities, as in random forest.

Starting settings:

- `n_estimators=100`: build 100 stages.
- `learning_rate=0.05`: limit each new tree's contribution.
- `max_depth=2`: use small trees, with at most two decisions along a path.
- `min_samples_leaf=10`: require at least 10 training leads per end branch.
- `random_state=42`: make the run repeatable.

This version uses only training data to build the trees. It does not use early
stopping or tune settings. Read `gradient_boosting/results/how_it_works.md`.

## Try forest and boosting hyperparameters

```powershell
uv run python -m random_forest.tune_model
uv run python -m gradient_boosting.tune_model
```

Each script tries the original settings and six changes. Each run changes only
one setting from the original. This is a small comparison, not a full grid search.

| Model | Settings and values tried |
|---|---|
| Random forest | Trees: 50, 100, 200; depth: 2, 3, 5; minimum end group: 5, 10, 20 |
| Gradient boosting | Stages: 50, 100, 200; learning rate: 0.02, 0.05, 0.1; depth: 1, 2, 3 |

Read each algorithm's `results/hyperparameters/comparison.md` and `summary.csv`.
Each run has its own predictions, ranking, and settings. Original results remain
in place. The shared model comparison still shows the original models.

Repeated comparisons can fit chance patterns in the same small validation group.
Keep the search small. No final model is selected, and the test set is not opened.

## Compare models

```powershell
uv run python -m common.compare_models
```

Read `results/model_comparison.md`.

- Brier score and log loss measure probability errors. Lower is better.
- Precision@10 is the buyer rate in the selected top 10.
- The ranked CSV shows position and whether each lead is selected.
- selected_top_10=1 means selected. It does not mean the lead bought.
- Tied scores use lead ID for a repeatable displayed order.
- Reports also show an expected buyer count for random selection inside ties.

A displayed list can look better just because of tie ordering. Compare the
expected result too. No final algorithm has been selected.

## Final test: selected logistic regression

We selected logistic regression with `C=1.0` after validation comparisons. The
more complex models did not show a clear benefit for our main ranking check.

```powershell
uv run python -m logistic_regression.evaluate_test
```

This trains again on the same 120 training leads, then evaluates the 40 test leads.
Validation leads are not added to training. The script does not tune settings or
compare other algorithms on test data.

Results are saved in `logistic_regression/results/final_test/`:

- `selected_model.json`: fixed model choice and training record.
- `model.joblib`: fitted model and feature encoding.
- `test_predictions.csv`: purchase probabilities and actual outcomes.
- `test_ranked.csv`: ordered test leads and top-10 selection.
- `test_evaluation.md`: baseline comparison and ranking results.
- `metrics.json`: recorded scores and test data fingerprint.

Do not change the model to improve this test result. Repeating the same run does
not create new evidence. A later real-data exercise needs its own evaluation plan.

## Important limits

A separate larger experiment is available in `experiments/leads_2000/`.
Read its README for the sequential commands. It keeps the original 200-lead
data and results in place, and reserves a separate test group.

Training scripts train again when run. The final test script also saves a model
file. Running scripts again replaces their result files. Only the final test
script opens the test group for model evaluation.

Current evidence is small: 40 validation leads and 6 buyers. Do not use one result
as a general claim about an algorithm. Ranking quality and accurate probabilities
are different checks.

# Separate experiment: 2,000 synthetic leads

Use the same buying process, features, and target as the first PoC. Use a new
generation seed, 43. Keep the original 200-lead files in place.

Run from `poc`, one command at a time:

```powershell
uv run python -m experiments.leads_2000.prepare_data
uv run python -m experiments.leads_2000.train_model logistic_regression
uv run python -m experiments.leads_2000.train_model decision_tree
uv run python -m experiments.leads_2000.train_model random_forest
uv run python -m experiments.leads_2000.train_model gradient_boosting
```

Preparation is run once. It creates 1,200 training leads, 400 validation leads,
and 400 reserved test leads. Each model uses its original settings and is run
separately. The script reuses model builders from the algorithm folders.

Each algorithm has its own result folder inside this experiment. Read
`comparison.md` for the combined validation results. Re-running a model replaces
its result files. The test group is not evaluated during these comparisons.

Top 10 now means 2.5% of validation leads. We also report top 100, which keeps the
same selection share as top 10 of 40 in the first PoC. This helps avoid confusing
a change in calling capacity with an improvement in the model.

This new sample can show how model results change with a larger dataset. It
does not isolate sample size as the only cause. A controlled learning-curve
experiment would keep evaluation examples fixed while changing training size.

## Final test: random forest

We selected random forest after the validation comparison. Keep its original
settings: 100 trees, maximum depth 3, minimum end group 10, and random seed 42.

```powershell
uv run python -m experiments.leads_2000.random_forest.evaluate_test
```

This trains again on the same 1,200 training leads and evaluates the 400 test
leads. It does not add validation leads to training or tune settings.
Read `random_forest/results/final_test/test_evaluation.md`.

That folder also contains the selected settings, fitted model, test predictions,
ranked lists for top 10 and top 100, and recorded metrics. Do not change settings
to improve the test score. Repeating the run is not new evidence.

## Follow-up: random forest hyperparameters

```powershell
uv run python -m experiments.leads_2000.random_forest.tune_model
```

This uses only the same 1,200 training leads and 400 validation leads. The main
ranking check is top 100. Compare the original settings with six changes:

- Trees: 50 and 200, compared with 100.
- Maximum depth: 2 and 5, compared with 3.
- Minimum end group: 5 and 20, compared with 10.

Each run changes one setting from the original. Read
`random_forest/results/hyperparameters/comparison.md`. Each run has its own
predictions, ranked list, and settings. Original validation and test files remain
in place.

This is a follow-up experiment after seeing the original test result. The tuning
script does not read test data. No tuned model has a new independent final test.

### Check the tuned candidate on the reused test group

The 50-tree candidate had slightly lower validation probability errors and the
same top-100 ranking result. Check it with:

```powershell
uv run python -m experiments.leads_2000.random_forest.evaluate_follow_up
```

This is an exploratory reuse of test data we already inspected. It is not a new
independent final check. Results go to `results/follow_up_test_50_trees/` inside
this experiment's random forest folder. Original `final_test` files stay in place.

## Depth 5 with fresh synthetic test examples

```powershell
uv run python -m experiments.leads_2000.random_forest.evaluate_fresh_test
```

This fixes 100 trees, maximum depth 5, and minimum end group 10 before creating
400 new examples. It uses the same buying process and generation seed 44.
The model still learns only from the same 1,200 training leads.

`min_samples_leaf=10` means a split is allowed only if both new end groups have
at least 10 training examples. Groups may contain more than 10. For this forest,
repeated draws of one row do not count as separate unique examples for this limit.

The new test file is `data/fresh_test_seed44.csv`. Results go to
`random_forest/results/fresh_test_depth_5/`. Earlier test files stay in place.
Reruns repeat the same examples. The script does not select another seed after
seeing results.

Changing both depth and test examples does not isolate the effect of depth.
Do not claim that deeper trees are better based on a comparison with the old test.

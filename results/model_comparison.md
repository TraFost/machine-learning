# Model comparison on the same validation leads

All models learn from the same training leads and use the same features.
Logistic regression uses its original C=1.0 setting.
Decision tree uses max_depth=3 and min_samples_leaf=10. These are starting settings.
Random forest uses 100 trees, max_depth=3, and min_samples_leaf=10. These are starting settings.
Gradient boosting uses 100 stages, learning_rate=0.05, max_depth=2, and min_samples_leaf=10.
The original models are compared here. Separate tuning reports show the other settings.

| Model | Displayed top-10 buyers | Expected buyers with random ties | Brier score | Log loss |
|---|---:|---:|---:|---:|
| Logistic regression | 3 | 3.86 | 0.1373 | 0.4399 |
| Decision tree | 4 | 3.57 | 0.1532 | 0.4672 |
| Random forest | 3 | 2.71 | 0.1402 | 0.4464 |
| Gradient-boosted trees | 3 | 2.71 | 0.1425 | 0.4487 |

More buyers is better. Lower Brier score and log loss are better.
The displayed top 10 uses lead ID to order equal scores. That order is not learned by the model.
Expected buyers averages possible random choices inside a tie. Use this for a fairer ranking comparison.
The tree can show more buyers in one displayed list without having a better expected result.
Different algorithms do not guarantee better results. This is one small synthetic validation group.
No final model has been selected. The test set was not opened.

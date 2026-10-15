# How gradient-boosted trees learn

Start with a prediction based on the training buyer rate.
Add a small tree that helps reduce the current training errors.
Update the combined prediction. Repeat for 100 stages.

The small trees supply corrections. They are not independent purchase classifiers.
The final probability comes from the combined model, not an average vote.
The learning rate limits each new tree's contribution.

Settings: 100 stages, learning_rate=0.05, max_depth=2, min_samples_leaf=10.
The same training leads are used at every stage in this version.
Only training outcomes guide learning. Validation outcomes are not used to build the trees.
No early stopping or hyperparameter search was used.
More stages do not guarantee better results on new leads.
The test set was not opened.

Reference: [scikit-learn gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).

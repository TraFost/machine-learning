# How this random forest learns

We use the same 120 training leads and the same three input fields.
Encoding converts purchase intent into numeric columns.

1. Build 100 trees.
2. Give each tree a random sample of training rows. A row can appear more than once in a sample.
3. At each split, consider a random subset of encoded features.
4. Each tree predicts a purchase probability.
5. Average the probabilities from all 100 trees.

Sampling does not create new customer evidence. It reuses the same training leads.
The trees have a depth limit of 3 and an end-group minimum of 10.
The forest can reduce reliance on one tree. It does not guarantee improvement.
These are starting settings. No hyperparameter search was run.
The test set was not opened.

Reference: [scikit-learn random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

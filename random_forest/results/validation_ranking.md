# Validation ranking: contact the top 10 leads

There are 40 validation leads and 6 actual buyers.
We select 10 leads using predicted probability, highest first.
When scores tie, lead ID decides the displayed order. Purchase outcomes never decide the order.

| Lead | Predicted purchase chance | Actually bought? |
|---|---:|---|
| L-054 | 58.7% | No |
| L-158 | 58.7% | No |
| L-181 | 58.7% | No |
| L-148 | 57.1% | Yes |
| L-010 | 54.4% | No |
| L-025 | 54.4% | No |
| L-076 | 54.4% | No |
| L-078 | 54.4% | Yes |
| L-083 | 54.4% | Yes |
| L-098 | 54.4% | No |

This displayed list contains 3 buyers among 10 leads.
Precision@10: 30.0%. This is the buyer rate in the selected list.
Buyers found: 3 of 6 actual buyers.

## Check the tied scores

At the cutoff, 7 leads share a 54.4% score, but only 6 places remain.
Different selections within this tie could give 2 to 3 buyers in the top 10.
With random selection inside the tie, the expected buyer count is 2.71.
An expected count is an average across possible selections, not a fraction of a person.

## Compare with random selection

Randomly choosing 10 of these 40 leads would find 1.50 buyers on average.
The model's tie-adjusted expected buyer rate is 1.81 times the random-selection rate. This is lift@10.

This checks ordering, not whether predicted percentages are accurate.
It uses observed validation outcomes for evaluation; those outcomes were unknown to the model.
It does not prove contacting these leads causes more purchases.
This small synthetic validation group is not enough for a reliable business conclusion.
The test set was not opened. No model was trained or changed.

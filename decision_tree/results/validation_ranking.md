# Validation ranking: contact the top 10 leads

There are 40 validation leads and 6 actual buyers.
We select 10 leads using predicted probability, highest first.
When scores tie, lead ID decides the displayed order. Purchase outcomes never decide the order.

| Lead | Predicted purchase chance | Actually bought? |
|---|---:|---|
| L-010 | 65.2% | No |
| L-025 | 65.2% | No |
| L-050 | 65.2% | No |
| L-054 | 65.2% | No |
| L-068 | 65.2% | Yes |
| L-076 | 65.2% | No |
| L-078 | 65.2% | Yes |
| L-083 | 65.2% | Yes |
| L-089 | 65.2% | Yes |
| L-098 | 65.2% | No |

This displayed list contains 4 buyers among 10 leads.
Precision@10: 40.0%. This is the buyer rate in the selected list.
Buyers found: 4 of 6 actual buyers.

## Check the tied scores

At the cutoff, 14 leads share a 65.2% score, but only 10 places remain.
Different selections within this tie could give 1 to 5 buyers in the top 10.
With random selection inside the tie, the expected buyer count is 3.57.
An expected count is an average across possible selections, not a fraction of a person.

## Compare with random selection

Randomly choosing 10 of these 40 leads would find 1.50 buyers on average.
The model's tie-adjusted expected buyer rate is 2.38 times the random-selection rate. This is lift@10.

This checks ordering, not whether predicted percentages are accurate.
It uses observed validation outcomes for evaluation; those outcomes were unknown to the model.
It does not prove contacting these leads causes more purchases.
This small synthetic validation group is not enough for a reliable business conclusion.
The test set was not opened. No model was trained or changed.

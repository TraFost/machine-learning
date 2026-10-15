# Original and follow-up forest checks

The 50-tree setting was chosen from validation results. All other model settings stayed the same.
The test group was already inspected before tuning. This comparison is exploratory, not a new final test.

| Trees | Top-100 buyers | Expected buyers with random ties | Brier | Log loss |
|---:|---:|---:|---:|---:|
| 100 | 41 | 42.16 | 0.1452 | 0.4529 |
| 50 | 41 | 42.16 | 0.1453 | 0.4534 |

The original final_test files remain unchanged.
Do not keep adjusting settings to improve this reused test result.

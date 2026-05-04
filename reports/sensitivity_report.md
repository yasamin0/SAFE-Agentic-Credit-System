# Sensitivity Analysis Report

Evaluates how SAFE decisions change under variations in weights, thresholds, sensitive feature choice, and perturbation assumptions.

## Scenario Table

| scenario                          |   prediction_threshold |   approval_threshold |   w_auc |   w_fair |   w_rob | sensitive_feature   |      auc |   fairness_aggregate |   robustness_aggregate |   safe_score | decision   |   delta_vs_base |
|:----------------------------------|-----------------------:|---------------------:|--------:|---------:|--------:|:--------------------|---------:|---------------------:|-----------------------:|-------------:|:-----------|----------------:|
| sensitive_feature=foreign_worker  |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | foreign_worker      | 0.776667 |             0.729906 |               0.956571 |     0.789267 | APPROVED   |      0.0968259  |
| weights=(0.30,0.30,0.40)          |                   0.55 |                 0.75 |     0.3 |      0.3 |     0.4 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.776504 | APPROVED   |      0.0840633  |
| weights=(0.50,0.30,0.20)          |                   0.55 |                 0.75 |     0.5 |      0.3 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.740524 | REJECTED   |      0.0480825  |
| prediction_threshold=0.6          |                   0.6  |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.548611 |               0.956571 |     0.69862  | REJECTED   |      0.00617851 |
| approval_threshold=0.7            |                   0.55 |                 0.7  |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.692441 | REJECTED   |      0          |
| approval_threshold=0.75           |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.692441 | REJECTED   |      0          |
| approval_threshold=0.8            |                   0.55 |                 0.8  |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.692441 | REJECTED   |      0          |
| base                              |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.692441 | REJECTED   |      0          |
| prediction_threshold=0.55         |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.692441 | REJECTED   |      0          |
| sensitive_feature=personal_status |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.692441 | REJECTED   |      0          |
| weights=(0.30,0.50,0.20)          |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.692441 | REJECTED   |      0          |
| prediction_threshold=0.45         |                   0.45 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.322917 |               0.956571 |     0.585772 | REJECTED   |     -0.106669   |
| prediction_threshold=0.5          |                   0.5  |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.322917 |               0.956571 |     0.585772 | REJECTED   |     -0.106669   |
| sensitive_feature=age             |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | age                 | 0.776667 |             0        |               0.956571 |     0.424314 | REJECTED   |     -0.268127   |

## Main Effects

| factor               |   mean_effect_range |
|:---------------------|--------------------:|
| prediction_threshold |           0.0902778 |
| w_fair               |           0.0687984 |
| w_rob                |           0.0359808 |
| approval_threshold   |           0         |

## Pairwise Interactions

| factor_a             | factor_b           |   interaction_strength |
|:---------------------|:-------------------|-----------------------:|
| prediction_threshold | w_fair             |            0.0073172   |
| w_fair               | w_rob              |            1.23358e-16 |
| prediction_threshold | approval_threshold |            1.11022e-16 |
| prediction_threshold | w_rob              |            9.25186e-17 |
| approval_threshold   | w_fair             |            0           |
| approval_threshold   | w_rob              |            0           |

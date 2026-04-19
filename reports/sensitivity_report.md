# Sensitivity Analysis Report

Evaluates how SAFE decisions change under variations in weights, thresholds, sensitive feature choice, and perturbation assumptions.

## Scenario Table

| scenario                          |   prediction_threshold |   approval_threshold |   w_auc |   w_fair |   w_rob | sensitive_feature   |   auc |   fairness_aggregate |   robustness_aggregate |   safe_score | decision   |   delta_vs_base |
|:----------------------------------|-----------------------:|---------------------:|--------:|---------:|--------:|:--------------------|------:|---------------------:|-----------------------:|-------------:|:-----------|----------------:|
| weights=(0.30,0.30,0.40)          |                   0.55 |                 0.75 |     0.3 |      0.3 |     0.4 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.781409 | APPROVED   |       0.0847286 |
| sensitive_feature=foreign_worker  |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | foreign_worker      |  0.78 |             0.706034 |               0.963573 |     0.779732 | APPROVED   |       0.0830518 |
| weights=(0.50,0.30,0.20)          |                   0.55 |                 0.75 |     0.5 |      0.3 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.744694 | REJECTED   |       0.0480139 |
| approval_threshold=0.7            |                   0.55 |                 0.7  |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| approval_threshold=0.75           |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| approval_threshold=0.8            |                   0.55 |                 0.8  |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| base                              |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| prediction_threshold=0.5          |                   0.5  |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| prediction_threshold=0.55         |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| prediction_threshold=0.6          |                   0.6  |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| sensitive_feature=personal_status |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| weights=(0.30,0.50,0.20)          |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| prediction_threshold=0.45         |                   0.45 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.481944 |               0.963573 |     0.667687 | REJECTED   |      -0.0289931 |
| sensitive_feature=age             |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | age                 |  0.78 |             0        |               0.963573 |     0.426715 | REJECTED   |      -0.269965  |

## Main Effects

| factor               |   mean_effect_range |
|:---------------------|--------------------:|
| w_fair               |           0.0509132 |
| w_rob                |           0.0367147 |
| prediction_threshold |           0.0231944 |
| approval_threshold   |           0         |

## Pairwise Interactions

| factor_a             | factor_b           |   interaction_strength |
|:---------------------|:-------------------|-----------------------:|
| prediction_threshold | w_fair             |            0.00144965  |
| prediction_threshold | w_rob              |            1.11022e-16 |
| w_fair               | w_rob              |            4.93432e-17 |
| prediction_threshold | approval_threshold |            2.77556e-17 |
| approval_threshold   | w_fair             |            0           |
| approval_threshold   | w_rob              |            0           |

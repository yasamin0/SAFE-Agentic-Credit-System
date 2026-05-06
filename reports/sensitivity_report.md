# Sensitivity Analysis Report

Evaluates how SAFE decisions change under variations in weights, thresholds, sensitive feature choice, and perturbation assumptions.

## Scenario Table

| scenario                          |   prediction_threshold |   approval_threshold |   w_auc |   w_fair |   w_rob | sensitive_feature   |      auc |   fairness_aggregate |   robustness_aggregate |   safe_score | decision   |   delta_vs_base |
|:----------------------------------|-----------------------:|---------------------:|--------:|---------:|--------:|:--------------------|---------:|---------------------:|-----------------------:|-------------:|:-----------|----------------:|
| sensitive_feature=foreign_worker  |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | foreign_worker      | 0.808571 |             0.735242 |               0.965548 |     0.803302 | APPROVED   |       0.0976557 |
| weights=(0.30,0.30,0.40)          |                   0.55 |                 0.75 |     0.3 |      0.3 |     0.4 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.79077  | APPROVED   |       0.0851234 |
| weights=(0.50,0.30,0.20)          |                   0.55 |                 0.75 |     0.5 |      0.3 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.759374 | APPROVED   |       0.0537282 |
| prediction_threshold=0.6          |                   0.6  |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.604167 |               0.965548 |     0.737764 | REJECTED   |       0.0321181 |
| approval_threshold=0.7            |                   0.55 |                 0.7  |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.705646 | APPROVED   |       0         |
| approval_threshold=0.75           |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.705646 | REJECTED   |       0         |
| approval_threshold=0.8            |                   0.55 |                 0.8  |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.705646 | REJECTED   |       0         |
| base                              |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.705646 | REJECTED   |       0         |
| prediction_threshold=0.55         |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.705646 | REJECTED   |       0         |
| sensitive_feature=personal_status |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.705646 | REJECTED   |       0         |
| weights=(0.30,0.50,0.20)          |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.705646 | REJECTED   |       0         |
| prediction_threshold=0.45         |                   0.45 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.404412 |               0.965548 |     0.637887 | REJECTED   |      -0.0677594 |
| prediction_threshold=0.5          |                   0.5  |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.404412 |               0.965548 |     0.637887 | REJECTED   |      -0.0677594 |
| sensitive_feature=age             |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | age                 | 0.808571 |             0        |               0.965548 |     0.435681 | REJECTED   |      -0.269965  |

## Main Effects

| factor               |   mean_effect_range |
|:---------------------|--------------------:|
| prediction_threshold |           0.079902  |
| w_fair               |           0.0640682 |
| w_rob                |           0.0313953 |
| approval_threshold   |           0         |

## Pairwise Interactions

| factor_a             | factor_b           |   interaction_strength |
|:---------------------|:-------------------|-----------------------:|
| prediction_threshold | w_fair             |            0.00558789  |
| prediction_threshold | w_rob              |            1.66533e-16 |
| prediction_threshold | approval_threshold |            2.77556e-17 |
| w_fair               | w_rob              |            1.23358e-17 |
| approval_threshold   | w_fair             |            0           |
| approval_threshold   | w_rob              |            0           |

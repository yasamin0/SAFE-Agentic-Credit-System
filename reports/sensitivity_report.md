# Sensitivity Analysis Report

Evaluates how SAFE decisions change under variations in weights, thresholds, sensitive feature choice, and perturbation assumptions.

## Scenario Table

| scenario                          |   prediction_threshold |   approval_threshold |   w_rga |   w_rgr |   w_rge |   w_fair | sensitive_feature   |    aurga |   rgr_aggregate |    aurge |   fairness_aggregate |   safe_score | decision   |   delta_vs_base |
|:----------------------------------|-----------------------:|---------------------:|--------:|--------:|--------:|---------:|:--------------------|---------:|----------------:|---------:|---------------------:|-------------:|:-----------|----------------:|
| sensitive_feature=foreign_worker  |                   0.55 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | foreign_worker      | 0.709529 |        0.928066 | 0.971187 |             0.713316 |     0.830524 | APPROVED   |      0.0433464  |
| weights=(0.20,0.30,0.25,0.25)     |                   0.55 |                 0.75 |    0.2  |    0.3  |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.798105 | APPROVED   |      0.0109268  |
| approval_threshold=0.7            |                   0.55 |                 0.7  |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |      0          |
| approval_threshold=0.75           |                   0.55 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |      0          |
| approval_threshold=0.8            |                   0.55 |                 0.8  |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | REJECTED   |      0          |
| base                              |                   0.55 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |      0          |
| prediction_threshold=0.55         |                   0.55 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |      0          |
| prediction_threshold=0.6          |                   0.6  |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |      0          |
| sensitive_feature=personal_status |                   0.55 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |      0          |
| weights=(0.25,0.25,0.25,0.25)     |                   0.55 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |      0          |
| weights=(0.20,0.20,0.30,0.30)     |                   0.55 |                 0.75 |    0.2  |    0.2  |    0.3  |     0.3  | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.780854 | APPROVED   |     -0.00632388 |
| weights=(0.30,0.25,0.20,0.25)     |                   0.55 |                 0.75 |    0.3  |    0.25 |    0.2  |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.774095 | APPROVED   |     -0.0130829  |
| prediction_threshold=0.45         |                   0.45 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.459559 |     0.767085 | APPROVED   |     -0.0200929  |
| prediction_threshold=0.5          |                   0.5  |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.415441 |     0.756056 | APPROVED   |     -0.0311223  |
| sensitive_feature=age             |                   0.55 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | age                 | 0.709529 |        0.928066 | 0.971187 |             0        |     0.652195 | REJECTED   |     -0.134983   |

## Main Effects

| factor               |   mean_effect_range |
|:---------------------|--------------------:|
| w_fair               |           0.114264  |
| w_rge                |           0.0482471 |
| w_rgr                |           0.043935  |
| prediction_threshold |           0.0311223 |
| w_rga                |           0.0220814 |
| approval_threshold   |           0         |

## Pairwise Interactions

| factor_a             | factor_b           |   interaction_strength |
|:---------------------|:-------------------|-----------------------:|
| prediction_threshold | w_fair             |            0.00343711  |
| prediction_threshold | w_rgr              |            0.00170718  |
| prediction_threshold | w_rge              |            0.00170718  |
| prediction_threshold | w_rga              |            0.00170718  |
| approval_threshold   | w_rgr              |            1.11022e-16 |
| w_rga                | w_rgr              |            1.11022e-16 |
| w_rgr                | w_rge              |            1.11022e-16 |
| prediction_threshold | approval_threshold |            8.32667e-17 |
| approval_threshold   | w_rga              |            7.40149e-17 |
| approval_threshold   | w_rge              |            7.40149e-17 |
| w_rga                | w_rge              |            6.16791e-17 |
| approval_threshold   | w_fair             |            2.96059e-17 |
| w_rga                | w_fair             |          nan           |
| w_rgr                | w_fair             |          nan           |
| w_rge                | w_fair             |          nan           |

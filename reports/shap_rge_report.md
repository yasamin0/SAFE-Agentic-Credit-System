# RGE vs SHAP Comparison Report

This report compares RGE feature ranking with SHAP feature ranking.

SHAP values were computed using XGBoost built-in TreeSHAP contributions.

- Spearman correlation: 0.9359
- Compared features: 61

## Top Features by RGE

| feature                                       |   rge_importance |   mean_abs_shap |   rge_rank |   shap_rank |
|:----------------------------------------------|-----------------:|----------------:|-----------:|------------:|
| duration                                      |       0.0477528  |       0.412601  |          1 |           2 |
| checking_status_no checking                   |       0.0446363  |       0.645755  |          2 |           1 |
| credit_amount                                 |       0.0386087  |       0.366573  |          3 |           3 |
| age                                           |       0.0196214  |       0.189198  |          4 |           6 |
| savings_status_less_than_100                  |       0.0186985  |       0.20909   |          5 |           5 |
| credit_history_critical/other existing credit |       0.0133838  |       0.25522   |          6 |           4 |
| installment_commitment                        |       0.00826577 |       0.14586   |          7 |           8 |
| checking_status_less_than_0                   |       0.00803319 |       0.188635  |          8 |           7 |
| other_payment_plans_bank                      |       0.00664074 |       0.139764  |          9 |           9 |
| employment_less_than_1                        |       0.00522323 |       0.104016  |         10 |          11 |
| other_payment_plans_none                      |       0.00512478 |       0.0816926 |         11 |          16 |
| other_parties_guarantor                       |       0.00495416 |       0.0756539 |         12 |          18 |
| purpose_used car                              |       0.00397618 |       0.12105   |         13 |          10 |
| property_magnitude_real estate                |       0.00391904 |       0.0827734 |         14 |          15 |
| property_magnitude_no known property          |       0.00377146 |       0.0968452 |         15 |          13 |

## Top Features by SHAP

| feature                                       |   rge_importance |   mean_abs_shap |   rge_rank |   shap_rank |
|:----------------------------------------------|-----------------:|----------------:|-----------:|------------:|
| checking_status_no checking                   |       0.0446363  |       0.645755  |          2 |           1 |
| duration                                      |       0.0477528  |       0.412601  |          1 |           2 |
| credit_amount                                 |       0.0386087  |       0.366573  |          3 |           3 |
| credit_history_critical/other existing credit |       0.0133838  |       0.25522   |          6 |           4 |
| savings_status_less_than_100                  |       0.0186985  |       0.20909   |          5 |           5 |
| age                                           |       0.0196214  |       0.189198  |          4 |           6 |
| checking_status_less_than_0                   |       0.00803319 |       0.188635  |          8 |           7 |
| installment_commitment                        |       0.00826577 |       0.14586   |          7 |           8 |
| other_payment_plans_bank                      |       0.00664074 |       0.139764  |          9 |           9 |
| purpose_used car                              |       0.00397618 |       0.12105   |         13 |          10 |
| employment_less_than_1                        |       0.00522323 |       0.104016  |         10 |          11 |
| employment_4less_than_=Xless_than_7           |       0.00291043 |       0.0996811 |         17 |          12 |
| property_magnitude_no known property          |       0.00377146 |       0.0968452 |         15 |          13 |
| purpose_new car                               |       0.0031094  |       0.0873753 |         16 |          14 |
| property_magnitude_real estate                |       0.00391904 |       0.0827734 |         14 |          15 |

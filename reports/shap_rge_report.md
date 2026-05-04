# RGE vs SHAP Comparison Report

This report compares RGE feature ranking with SHAP feature ranking.

SHAP values were computed using XGBoost built-in TreeSHAP contributions.

- Spearman correlation: 0.9838
- Compared features: 61

## Top Features by RGE

| feature                                       |   rge_importance |   mean_abs_shap |   rge_rank |   shap_rank |
|:----------------------------------------------|-----------------:|----------------:|-----------:|------------:|
| checking_status_no checking                   |       0.0838279  |       0.516695  |          1 |           1 |
| duration                                      |       0.0447539  |       0.283202  |          2 |           2 |
| credit_amount                                 |       0.015712   |       0.155965  |          3 |           4 |
| credit_history_critical/other existing credit |       0.0118309  |       0.159786  |          4 |           3 |
| savings_status_less_than_100                  |       0.00980749 |       0.123919  |          5 |           5 |
| other_payment_plans_bank                      |       0.00775343 |       0.0708847 |          6 |           8 |
| checking_status_less_than_0                   |       0.00695987 |       0.109477  |          7 |           6 |
| installment_commitment                        |       0.00480458 |       0.0654136 |          8 |          10 |
| other_payment_plans_none                      |       0.00448047 |       0.0593103 |          9 |          13 |
| property_magnitude_real estate                |       0.00426532 |       0.0519489 |         10 |          14 |
| employment_less_than_1                        |       0.00332291 |       0.0703868 |         11 |           9 |
| property_magnitude_no known property          |       0.00327274 |       0.0744427 |         12 |           7 |
| purpose_used car                              |       0.00271645 |       0.0593975 |         13 |          12 |
| age                                           |       0.00265341 |       0.0635839 |         14 |          11 |
| other_parties_guarantor                       |       0.00204778 |       0.026952  |         15 |          20 |

## Top Features by SHAP

| feature                                       |   rge_importance |   mean_abs_shap |   rge_rank |   shap_rank |
|:----------------------------------------------|-----------------:|----------------:|-----------:|------------:|
| checking_status_no checking                   |       0.0838279  |       0.516695  |          1 |           1 |
| duration                                      |       0.0447539  |       0.283202  |          2 |           2 |
| credit_history_critical/other existing credit |       0.0118309  |       0.159786  |          4 |           3 |
| credit_amount                                 |       0.015712   |       0.155965  |          3 |           4 |
| savings_status_less_than_100                  |       0.00980749 |       0.123919  |          5 |           5 |
| checking_status_less_than_0                   |       0.00695987 |       0.109477  |          7 |           6 |
| property_magnitude_no known property          |       0.00327274 |       0.0744427 |         12 |           7 |
| other_payment_plans_bank                      |       0.00775343 |       0.0708847 |          6 |           8 |
| employment_less_than_1                        |       0.00332291 |       0.0703868 |         11 |           9 |
| installment_commitment                        |       0.00480458 |       0.0654136 |          8 |          10 |
| age                                           |       0.00265341 |       0.0635839 |         14 |          11 |
| purpose_used car                              |       0.00271645 |       0.0593975 |         13 |          12 |
| other_payment_plans_none                      |       0.00448047 |       0.0593103 |          9 |          13 |
| property_magnitude_real estate                |       0.00426532 |       0.0519489 |         10 |          14 |
| personal_status_male single                   |       0.00168313 |       0.0472774 |         16 |          15 |

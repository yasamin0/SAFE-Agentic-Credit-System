# RGE vs SHAP Comparison Report

Selected operational model: Voting Ensemble

- SHAP status: completed
- SHAP method: kernel_explainer
- Spearman correlation: 0.9025383395029084

## Top RGE-SHAP Comparison Rows

| feature                                       |   rge_similarity_after_removal |   rge_importance |   importance_rank_least_to_most |   mean_abs_shap | shap_method      | model           |   rge_rank |   shap_rank |
|:----------------------------------------------|-------------------------------:|-----------------:|--------------------------------:|----------------:|:-----------------|:----------------|-----------:|------------:|
| checking_status_no checking                   |                       0.962999 |       0.037001   |                              61 |      0.0820248  | kernel_explainer | Voting Ensemble |          1 |           1 |
| duration                                      |                       0.976822 |       0.0231778  |                              60 |      0.0395935  | kernel_explainer | Voting Ensemble |          2 |           2 |
| credit_amount                                 |                       0.982739 |       0.0172609  |                              59 |      0.0299873  | kernel_explainer | Voting Ensemble |          3 |           4 |
| credit_history_critical/other existing credit |                       0.988129 |       0.0118705  |                              58 |      0.0387015  | kernel_explainer | Voting Ensemble |          4 |           3 |
| savings_status_less_than_100                  |                       0.990147 |       0.00985324 |                              57 |      0.0286613  | kernel_explainer | Voting Ensemble |          5 |           5 |
| checking_status_less_than_0                   |                       0.990826 |       0.00917381 |                              56 |      0.0283681  | kernel_explainer | Voting Ensemble |          6 |           6 |
| installment_commitment                        |                       0.993955 |       0.0060452  |                              55 |      0.0231122  | kernel_explainer | Voting Ensemble |          7 |           7 |
| age                                           |                       0.994906 |       0.0050944  |                              54 |      0.0101524  | kernel_explainer | Voting Ensemble |          8 |          15 |
| other_payment_plans_none                      |                       0.996082 |       0.00391767 |                              53 |      0.0109444  | kernel_explainer | Voting Ensemble |          9 |          11 |
| property_magnitude_real estate                |                       0.99616  |       0.0038403  |                              52 |      0.0118483  | kernel_explainer | Voting Ensemble |         10 |          10 |
| purpose_used car                              |                       0.996292 |       0.00370829 |                              51 |      0.0104935  | kernel_explainer | Voting Ensemble |         11 |          13 |
| other_payment_plans_bank                      |                       0.996329 |       0.00367149 |                              50 |      0.00895918 | kernel_explainer | Voting Ensemble |         12 |          17 |
| personal_status_male single                   |                       0.996499 |       0.00350083 |                              49 |      0.0130375  | kernel_explainer | Voting Ensemble |         13 |           9 |
| purpose_new car                               |                       0.997325 |       0.00267509 |                              48 |      0.0108433  | kernel_explainer | Voting Ensemble |         14 |          12 |
| property_magnitude_no known property          |                       0.997327 |       0.00267348 |                              47 |      0.0151609  | kernel_explainer | Voting Ensemble |         15 |           8 |
| employment_less_than_1                        |                       0.997378 |       0.00262176 |                              46 |      0.00906943 | kernel_explainer | Voting Ensemble |         16 |          16 |
| employment_4less_than_=Xless_than_7           |                       0.997394 |       0.00260565 |                              45 |      0.00849808 | kernel_explainer | Voting Ensemble |         17 |          18 |
| other_parties_guarantor                       |                       0.997411 |       0.00258866 |                              44 |      0.00626567 | kernel_explainer | Voting Ensemble |         18 |          25 |
| purpose_education                             |                       0.998435 |       0.0015651  |                              43 |      0.00646248 | kernel_explainer | Voting Ensemble |         19 |          23 |
| credit_history_no credits/all paid            |                       0.998641 |       0.00135925 |                              42 |      0.00635258 | kernel_explainer | Voting Ensemble |         20 |          24 |

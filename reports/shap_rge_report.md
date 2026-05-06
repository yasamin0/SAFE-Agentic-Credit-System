# RGE vs SHAP Comparison Report

Selected operational model: Stacking Ensemble

- SHAP status: completed
- SHAP method: kernel_explainer
- Spearman correlation: 0.8902696985721839

## Top RGE-SHAP Comparison Rows

| feature                                       |   rge_similarity_after_removal |   rge_importance |   importance_rank_least_to_most |   mean_abs_shap | shap_method      | model             |   rge_rank |   shap_rank |
|:----------------------------------------------|-------------------------------:|-----------------:|--------------------------------:|----------------:|:-----------------|:------------------|-----------:|------------:|
| checking_status_no checking                   |                       0.969578 |       0.0304221  |                              61 |      0.0648984  | kernel_explainer | Stacking Ensemble |          1 |           1 |
| duration                                      |                       0.980553 |       0.0194471  |                              60 |      0.0340796  | kernel_explainer | Stacking Ensemble |          2 |           2 |
| credit_amount                                 |                       0.984587 |       0.0154131  |                              59 |      0.0239618  | kernel_explainer | Stacking Ensemble |          3 |           6 |
| checking_status_less_than_0                   |                       0.988527 |       0.0114735  |                              58 |      0.0268981  | kernel_explainer | Stacking Ensemble |          4 |           4 |
| credit_history_critical/other existing credit |                       0.988596 |       0.0114042  |                              57 |      0.0315383  | kernel_explainer | Stacking Ensemble |          5 |           3 |
| savings_status_less_than_100                  |                       0.990007 |       0.00999274 |                              56 |      0.0268637  | kernel_explainer | Stacking Ensemble |          6 |           5 |
| installment_commitment                        |                       0.992984 |       0.00701605 |                              55 |      0.0203844  | kernel_explainer | Stacking Ensemble |          7 |           7 |
| personal_status_male single                   |                       0.996099 |       0.00390099 |                              54 |      0.0120176  | kernel_explainer | Stacking Ensemble |          8 |           9 |
| other_payment_plans_none                      |                       0.996165 |       0.00383549 |                              53 |      0.00995714 | kernel_explainer | Stacking Ensemble |          9 |          11 |
| purpose_used car                              |                       0.996318 |       0.00368229 |                              52 |      0.00839137 | kernel_explainer | Stacking Ensemble |         10 |          13 |
| property_magnitude_real estate                |                       0.996391 |       0.00360882 |                              51 |      0.0105223  | kernel_explainer | Stacking Ensemble |         11 |          10 |
| property_magnitude_no known property          |                       0.99647  |       0.00353038 |                              50 |      0.0151697  | kernel_explainer | Stacking Ensemble |         12 |           8 |
| other_payment_plans_bank                      |                       0.996808 |       0.0031923  |                              49 |      0.00824496 | kernel_explainer | Stacking Ensemble |         13 |          14 |
| age                                           |                       0.996937 |       0.00306326 |                              48 |      0.00578441 | kernel_explainer | Stacking Ensemble |         14 |          23 |
| purpose_new car                               |                       0.997087 |       0.00291264 |                              47 |      0.00770479 | kernel_explainer | Stacking Ensemble |         15 |          15 |
| other_parties_guarantor                       |                       0.997549 |       0.00245093 |                              46 |      0.00635429 | kernel_explainer | Stacking Ensemble |         16 |          21 |
| employment_4less_than_=Xless_than_7           |                       0.997771 |       0.00222871 |                              45 |      0.00873425 | kernel_explainer | Stacking Ensemble |         17 |          12 |
| employment_less_than_1                        |                       0.998224 |       0.00177623 |                              44 |      0.0069828  | kernel_explainer | Stacking Ensemble |         18 |          18 |
| savings_status_no known savings               |                       0.998343 |       0.00165699 |                              43 |      0.00761449 | kernel_explainer | Stacking Ensemble |         19 |          17 |
| purpose_education                             |                       0.998405 |       0.00159543 |                              42 |      0.00567044 | kernel_explainer | Stacking Ensemble |         20 |          24 |

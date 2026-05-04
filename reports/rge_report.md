# Rank Graduation Explainability Report

This report implements the paper-style RGE explainability analysis.

## Method
- Original predictions are computed on the clean test set.
- Each feature is removed individually to estimate RGE importance.
- Features are ordered from least important to most important.
- Features are progressively removed in this order to create the RGE curve.
- AURGE is the area under the RGE curve.

## Results
- AURGE: 0.9658
- Number of processed features: 61

## Most Important Features by RGE
| feature                                       |   rge_similarity_after_removal |   rge_importance |   importance_rank_least_to_most |
|:----------------------------------------------|-------------------------------:|-----------------:|--------------------------------:|
| duration                                      |                       0.952247 |       0.0477528  |                              61 |
| checking_status_no checking                   |                       0.955364 |       0.0446363  |                              60 |
| credit_amount                                 |                       0.961391 |       0.0386087  |                              59 |
| age                                           |                       0.980379 |       0.0196214  |                              58 |
| savings_status_less_than_100                  |                       0.981301 |       0.0186985  |                              57 |
| credit_history_critical/other existing credit |                       0.986616 |       0.0133838  |                              56 |
| installment_commitment                        |                       0.991734 |       0.00826577 |                              55 |
| checking_status_less_than_0                   |                       0.991967 |       0.00803319 |                              54 |
| other_payment_plans_bank                      |                       0.993359 |       0.00664074 |                              53 |
| employment_less_than_1                        |                       0.994777 |       0.00522323 |                              52 |

## Least Important Features by RGE
| feature                                      |   rge_similarity_after_removal |   rge_importance |   importance_rank_least_to_most |
|:---------------------------------------------|-------------------------------:|-----------------:|--------------------------------:|
| purpose_retraining                           |                       1        |      0           |                               1 |
| foreign_worker_yes                           |                       1        |      0           |                               2 |
| own_telephone_yes                            |                       1        |      0           |                               3 |
| job_unemp/unskilled non res                  |                       1        |      0           |                               4 |
| savings_status_500less_than_=Xless_than_1000 |                       1        |      0           |                               5 |
| purpose_domestic appliance                   |                       1        |      0           |                               6 |
| personal_status_male mar/wid                 |                       1        |      0           |                               7 |
| purpose_other                                |                       1        |      0           |                               8 |
| other_payment_plans_stores                   |                       0.999992 |      7.54201e-06 |                               9 |
| purpose_business                             |                       0.999933 |      6.72049e-05 |                              10 |

## Output Files
- RGE feature importance CSV: rge_feature_importance.csv
- RGE curve CSV: rge_curve.csv
- RGE curve plot: rge_curve.png
- RGE importance plot: rge_feature_importance.png

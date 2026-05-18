# Rank Graduation Explainability Report

RGE measures ranking change when features are removed.

## Results
- AURGE: 0.9712
- Number of processed features: 61

## Most Important Features by RGE
| feature                                       |   rge_similarity_after_removal |   rge_importance |   importance_rank_least_to_most |
|:----------------------------------------------|-------------------------------:|-----------------:|--------------------------------:|
| checking_status_no checking                   |                       0.962999 |       0.037001   |                              61 |
| duration                                      |                       0.976822 |       0.0231778  |                              60 |
| credit_amount                                 |                       0.982739 |       0.0172609  |                              59 |
| credit_history_critical/other existing credit |                       0.988129 |       0.0118705  |                              58 |
| savings_status_less_than_100                  |                       0.990147 |       0.00985324 |                              57 |
| checking_status_less_than_0                   |                       0.990826 |       0.00917381 |                              56 |
| installment_commitment                        |                       0.993955 |       0.0060452  |                              55 |
| age                                           |                       0.994906 |       0.0050944  |                              54 |
| other_payment_plans_none                      |                       0.996082 |       0.00391767 |                              53 |
| property_magnitude_real estate                |                       0.99616  |       0.0038403  |                              52 |

## Least Important Features by RGE
| feature                                      |   rge_similarity_after_removal |   rge_importance |   importance_rank_least_to_most |
|:---------------------------------------------|-------------------------------:|-----------------:|--------------------------------:|
| purpose_domestic appliance                   |                       1        |      0           |                               1 |
| savings_status_500less_than_=Xless_than_1000 |                       0.999995 |      5.21495e-06 |                               2 |
| personal_status_male mar/wid                 |                       0.999987 |      1.34592e-05 |                               3 |
| other_payment_plans_stores                   |                       0.999975 |      2.54192e-05 |                               4 |
| purpose_retraining                           |                       0.999967 |      3.31672e-05 |                               5 |
| job_unemp/unskilled non res                  |                       0.999966 |      3.36437e-05 |                               6 |
| purpose_business                             |                       0.999928 |      7.18869e-05 |                               7 |
| other_parties_co applicant                   |                       0.99989  |      0.000109569 |                               8 |
| credit_history_all paid                      |                       0.999888 |      0.000112298 |                               9 |
| job_unskilled resident                       |                       0.99988  |      0.000120416 |                              10 |

## Output Files
- RGE feature importance CSV: rge_feature_importance.csv
- RGE curve CSV: rge_curve.csv
- RGE curve plot: rge_curve.png
- RGE importance plot: rge_feature_importance.png

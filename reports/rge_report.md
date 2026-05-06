# Rank Graduation Explainability Report

RGE measures ranking change when features are removed.

## Results
- AURGE: 0.9725
- Number of processed features: 61

## Most Important Features by RGE
| feature                                       |   rge_similarity_after_removal |   rge_importance |   importance_rank_least_to_most |
|:----------------------------------------------|-------------------------------:|-----------------:|--------------------------------:|
| checking_status_no checking                   |                       0.969578 |       0.0304221  |                              61 |
| duration                                      |                       0.980553 |       0.0194471  |                              60 |
| credit_amount                                 |                       0.984587 |       0.0154131  |                              59 |
| checking_status_less_than_0                   |                       0.988527 |       0.0114735  |                              58 |
| credit_history_critical/other existing credit |                       0.988596 |       0.0114042  |                              57 |
| savings_status_less_than_100                  |                       0.990007 |       0.00999274 |                              56 |
| installment_commitment                        |                       0.992984 |       0.00701605 |                              55 |
| personal_status_male single                   |                       0.996099 |       0.00390099 |                              54 |
| other_payment_plans_none                      |                       0.996165 |       0.00383549 |                              53 |
| purpose_used car                              |                       0.996318 |       0.00368229 |                              52 |

## Least Important Features by RGE
| feature                                      |   rge_similarity_after_removal |   rge_importance |   importance_rank_least_to_most |
|:---------------------------------------------|-------------------------------:|-----------------:|--------------------------------:|
| purpose_domestic appliance                   |                       0.999999 |      1.40386e-06 |                               1 |
| savings_status_500less_than_=Xless_than_1000 |                       0.999992 |      8.20037e-06 |                               2 |
| personal_status_male mar/wid                 |                       0.999986 |      1.35141e-05 |                               3 |
| purpose_retraining                           |                       0.999971 |      2.87634e-05 |                               4 |
| other_payment_plans_stores                   |                       0.99996  |      3.98786e-05 |                               5 |
| purpose_business                             |                       0.999928 |      7.19638e-05 |                               6 |
| job_unemp/unskilled non res                  |                       0.999919 |      8.11e-05    |                               7 |
| job_unskilled resident                       |                       0.999884 |      0.00011565  |                               8 |
| checking_status_>=200                        |                       0.999884 |      0.000116174 |                               9 |
| other_parties_co applicant                   |                       0.999867 |      0.000132569 |                              10 |

## Output Files
- RGE feature importance CSV: rge_feature_importance.csv
- RGE curve CSV: rge_curve.csv
- RGE curve plot: rge_curve.png
- RGE importance plot: rge_feature_importance.png

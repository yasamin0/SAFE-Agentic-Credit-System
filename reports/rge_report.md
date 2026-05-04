# Rank Graduation Explainability Report

This report implements the paper-style RGE explainability analysis.

## Method
- Original predictions are computed on the clean test set.
- Each feature is removed individually to estimate RGE importance.
- Features are ordered from least important to most important.
- Features are progressively removed in this order to create the RGE curve.
- AURGE is the area under the RGE curve.

## Results
- AURGE: 0.9816
- Number of processed features: 61

## Most Important Features by RGE
| feature                                       |   rge_similarity_after_removal |   rge_importance |   importance_rank_least_to_most |
|:----------------------------------------------|-------------------------------:|-----------------:|--------------------------------:|
| checking_status_no checking                   |                       0.916172 |       0.0838279  |                              61 |
| duration                                      |                       0.955246 |       0.0447539  |                              60 |
| credit_amount                                 |                       0.984288 |       0.015712   |                              59 |
| credit_history_critical/other existing credit |                       0.988169 |       0.0118309  |                              58 |
| savings_status_less_than_100                  |                       0.990193 |       0.00980749 |                              57 |
| other_payment_plans_bank                      |                       0.992247 |       0.00775343 |                              56 |
| checking_status_less_than_0                   |                       0.99304  |       0.00695987 |                              55 |
| installment_commitment                        |                       0.995195 |       0.00480458 |                              54 |
| other_payment_plans_none                      |                       0.99552  |       0.00448047 |                              53 |
| property_magnitude_real estate                |                       0.995735 |       0.00426532 |                              52 |

## Least Important Features by RGE
| feature                                    |   rge_similarity_after_removal |   rge_importance |   importance_rank_least_to_most |
|:-------------------------------------------|-------------------------------:|-----------------:|--------------------------------:|
| checking_status_0less_than_=Xless_than_200 |                              1 |                0 |                               1 |
| purpose_domestic appliance                 |                              1 |                0 |                              21 |
| purpose_business                           |                              1 |                0 |                              20 |
| credit_history_existing paid               |                              1 |                0 |                              19 |
| checking_status_>=200                      |                              1 |                0 |                              18 |
| purpose_furniture/equipment                |                              1 |                0 |                              17 |
| num_dependents                             |                              1 |                0 |                              16 |
| purpose_other                              |                              1 |                0 |                              14 |
| purpose_repairs                            |                              1 |                0 |                              13 |
| purpose_retraining                         |                              1 |                0 |                              12 |

## Output Files
- RGE feature importance CSV: rge_feature_importance.csv
- RGE curve CSV: rge_curve.csv
- RGE curve plot: rge_curve.png
- RGE importance plot: rge_feature_importance.png

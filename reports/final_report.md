# Final SAFE Agentic Credit Scoring Report

## User Controls
- Data source: CSV (data/raw/raw_credit_data.csv)
- Prediction threshold: 0.55
- Approval threshold: 0.75
- Weights: AUC=0.300, Fairness=0.500, Robustness=0.200
- Sensitive feature: personal_status
- Drop sensitive from model: False
- Decision rule: APPROVED if SAFE_SCORE >= APPROVAL_THRESHOLD else REJECTED, where SAFE_SCORE = W_AUC*AUC + W_FAIR*FAIRNESS_AGG + W_ROB*ROBUSTNESS_AGG

## Accuracy
- AUC: 0.7767

## Classification Metrics
- PR-AUC: 0.5615
- Precision: 0.6562
- Recall: 0.3500
- F1 Score: 0.4565
- Brier Score: 0.1689

Confusion matrix:
|          |   pred_0 |   pred_1 |
|:---------|---------:|---------:|
| actual_0 |      129 |       11 |
| actual_1 |       39 |       21 |

Calibration curve data:
|   mean_predicted_probability |   fraction_of_positives |
|-----------------------------:|------------------------:|
|                    0.0725791 |                0.103448 |
|                    0.144692  |                0.104167 |
|                    0.245052  |                0.228571 |
|                    0.364269  |                0.285714 |
|                    0.441174  |                0.448276 |
|                    0.566911  |                0.68     |
|                    0.66095   |                0.666667 |
|                    0.709972  |                0        |

## Fairness Aggregation
- SPD gap: 0.3611
- EOD gap: 0.3431
- AOD gap: 0.3382
- Disparate impact ratio: 0.1875
- Fairness aggregate: 0.5363

## Robustness Aggregation
- Noise AUC ratio: 1.0000
- Dropout AUC ratio: 0.8708
- Missingness AUC ratio: 0.9989
- Robustness aggregate: 0.9566

## Rank-Based Robustness: RGR / AURGR
- AURGR Gaussian Noise: 0.9814
- AURGR Percentile Swapping: 0.8475
- RGR Aggregate: 0.9145
- Gaussian RGR curve CSV: rgr_gaussian_curve.csv
- Percentile Swapping RGR curve CSV: rgr_swapping_curve.csv
- Gaussian RGR plot: rgr_gaussian_curve.png
- Percentile Swapping RGR plot: rgr_swapping_curve.png

Interpretation:
- RGR measures whether the ranking of model predictions remains stable after perturbing the input data.
- A higher AURGR means the model is more robust across increasing perturbation intensities.
- Gaussian noise tests sensitivity to continuous random noise.
- Percentile swapping tests sensitivity to stronger distributional perturbations.

## Ensemble Auditing
Individual auditor scores:
| auditor             |    score |
|:--------------------|---------:|
| performance_auditor | 0.776667 |
| fairness_auditor    | 0.536254 |
| robustness_auditor  | 0.956571 |

- Final ensemble SAFE score: 0.6924
- Ensemble rule: weighted aggregation of independent performance, fairness, and robustness auditors.

## Mitigation Experiment
- Mitigation type: group-aware threshold adjustment
- Disadvantaged group detected: male mar/wid
- Baseline fairness aggregate: 0.5363
- Mitigated fairness aggregate: 0.5363
- Baseline SAFE score: 0.6924
- Mitigated SAFE score: 0.6502

### Group Table
| group              |   n |   positive_rate |      tpr |       fpr |
|:-------------------|----:|----------------:|---------:|----------:|
| female div/dep/mar |  61 |       0.163934  | 0.35     | 0.0731707 |
| male div/sep       |   9 |       0.444444  | 0.666667 | 0.333333  |
| male mar/wid       |  12 |       0.0833333 | 0.333333 | 0         |
| male single        | 118 |       0.144068  | 0.323529 | 0.0714286 |

## Sensitivity Analysis Summary
Top scenarios by SAFE score:
| scenario                         |   prediction_threshold |   approval_threshold |   w_auc |   w_fair |   w_rob | sensitive_feature   |      auc |   fairness_aggregate |   robustness_aggregate |   safe_score | decision   |   delta_vs_base |
|:---------------------------------|-----------------------:|---------------------:|--------:|---------:|--------:|:--------------------|---------:|---------------------:|-----------------------:|-------------:|:-----------|----------------:|
| sensitive_feature=foreign_worker |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | foreign_worker      | 0.776667 |             0.729906 |               0.956571 |     0.789267 | APPROVED   |      0.0968259  |
| weights=(0.30,0.30,0.40)         |                   0.55 |                 0.75 |     0.3 |      0.3 |     0.4 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.776504 | APPROVED   |      0.0840633  |
| weights=(0.50,0.30,0.20)         |                   0.55 |                 0.75 |     0.5 |      0.3 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.740524 | REJECTED   |      0.0480825  |
| prediction_threshold=0.6         |                   0.6  |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.548611 |               0.956571 |     0.69862  | REJECTED   |      0.00617851 |
| approval_threshold=0.7           |                   0.55 |                 0.7  |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.692441 | REJECTED   |      0          |
| approval_threshold=0.75          |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.692441 | REJECTED   |      0          |
| approval_threshold=0.8           |                   0.55 |                 0.8  |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.692441 | REJECTED   |      0          |
| base                             |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.692441 | REJECTED   |      0          |

## Interaction / Effects Summary
- Baseline SAFE score: 0.6924
- Best scenario from sensitivity analysis: sensitive_feature=foreign_worker
- Best scenario SAFE score: 0.7893
- Strongest observed effect beyond baseline: sensitive_feature=foreign_worker
- Effect size vs baseline: 0.0968
- Interpretation: the governance decision is sensitive to policy weights and sensitive-feature choice, while threshold changes had weaker effects in this run.

## Global Interaction Analysis
Top main effects on SAFE score:
| factor               |   mean_effect_range |
|:---------------------|--------------------:|
| prediction_threshold |           0.0902778 |
| w_fair               |           0.0687984 |
| w_rob                |           0.0359808 |
| approval_threshold   |           0         |

Top pairwise interactions:
| factor_a             | factor_b           |   interaction_strength |
|:---------------------|:-------------------|-----------------------:|
| prediction_threshold | w_fair             |            0.0073172   |
| w_fair               | w_rob              |            1.23358e-16 |
| prediction_threshold | approval_threshold |            1.11022e-16 |
| prediction_threshold | w_rob              |            9.25186e-17 |
| approval_threshold   | w_fair             |            0           |
| approval_threshold   | w_rob              |            0           |

Interpretation:
- Main effects show which single factor most strongly changes SAFE score on average.
- Pairwise interactions show which pairs of factors jointly influence the SAFE decision beyond their separate average effects.

## Rank-Based Explainability: RGE / AURGE
- AURGE: 0.9816
- RGE feature importance CSV: rge_feature_importance.csv
- RGE curve CSV: rge_curve.csv
- RGE curve plot: rge_curve.png
- RGE importance plot: rge_feature_importance.png

Interpretation:
- RGE measures how much the model prediction ranking changes when features are removed.
- Features are first ordered from least important to most important.
- The RGE curve is created by progressively removing features in this order.
- A higher AURGE means the model ranking remains more stable during progressive feature removal.

Top 10 most important processed features by RGE:
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

Top 10 least important processed features by RGE:
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

## Explainability Snapshot: XGBoost Feature Importance
Top 10 most important processed features by XGBoost importance:
| feature                                       |   importance |
|:----------------------------------------------|-------------:|
| checking_status_no checking                   |    0.12994   |
| savings_status_less_than_100                  |    0.0542038 |
| checking_status_less_than_0                   |    0.0360756 |
| property_magnitude_real estate                |    0.032887  |
| credit_history_critical/other existing credit |    0.0327943 |
| other_payment_plans_bank                      |    0.0321772 |
| credit_history_all paid                       |    0.0315327 |
| property_magnitude_no known property          |    0.0314187 |
| duration                                      |    0.0300228 |
| other_parties_guarantor                       |    0.0267429 |

## SAFE AI Paper Metrics: Multi-Model Compliance Comparison
- AURGA: 0.7012
- AURGR Gaussian Noise: 0.9814
- AURGR Percentile Swapping: 0.8475
- AURGE: 0.9816
- SHAP-RGE Spearman correlation: 0.983841540319698

Model metrics comparison:
| Model               |    AURGA |   AURGR_Gaussian |   AURGR_Swapping |    AURGR |    AURGE |
|:--------------------|---------:|-----------------:|-----------------:|---------:|---------:|
| Logistic Regression | 0.656765 |         0.996301 |         0.846401 | 0.921351 | 0.974049 |
| Random Forest       | 0.730336 |         0.99553  |         0.923426 | 0.959478 | 0.982013 |
| XGBoost             | 0.701229 |         0.981441 |         0.847468 | 0.914455 | 0.981596 |
| Voting Ensemble     | 0.720947 |         0.99164  |         0.864491 | 0.928066 | 0.971187 |
| Stacking Ensemble   | 0.713829 |         0.994305 |         0.869086 | 0.931696 | 0.972548 |
| Random Baseline     | 0.5      |         1        |         1        | 1        | 1        |

Compliance score comparison:
| Model               |    AURGA |   AURGR_Gaussian |   AURGR_Swapping |    AURGR |    AURGE |   Compliance_Arithmetic |   Compliance_Geometric |   Compliance_RMS |   Compliance_TOPSIS |
|:--------------------|---------:|-----------------:|-----------------:|---------:|---------:|------------------------:|-----------------------:|-----------------:|--------------------:|
| Random Forest       | 0.730336 |         0.99553  |         0.923426 | 0.959478 | 0.982013 |                0.890609 |               0.88286  |         0.897838 |            0.880599 |
| Voting Ensemble     | 0.720947 |         0.99164  |         0.864491 | 0.928066 | 0.971187 |                0.8734   |               0.866154 |         0.880203 |            0.797964 |
| Stacking Ensemble   | 0.713829 |         0.994305 |         0.869086 | 0.931696 | 0.972548 |                0.872691 |               0.864821 |         0.880049 |            0.795747 |
| XGBoost             | 0.701229 |         0.981441 |         0.847468 | 0.914455 | 0.981596 |                0.86576  |               0.857008 |         0.873972 |            0.744791 |
| Logistic Regression | 0.656765 |         0.996301 |         0.846401 | 0.921351 | 0.974049 |                0.850722 |               0.83844  |         0.861974 |            0.624381 |
| Random Baseline     | 0.5      |         1        |         1        | 1        | 1        |                0.833333 |               0.793701 |         0.866025 |            0.218483 |

Interpretation:
- AURGA evaluates rank-based accuracy under progressive data removal.
- AURGR evaluates rank-based robustness under increasing perturbation intensity.
- AURGE evaluates rank-based explainability under progressive feature removal.
- The final Compliance Score combines AURGA, AURGR, and AURGE using Arithmetic Mean, Geometric Mean, RMS, and TOPSIS.

## Auditor Notes
- Multi-metric fairness and robustness aggregation are enabled.
- Sensitivity analysis covers thresholds, weights, alternative sensitive features, and perturbation settings.

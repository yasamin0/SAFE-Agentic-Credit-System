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
- AUC: 0.7800

## Fairness Aggregation
- SPD gap: 0.3611
- EOD gap: 0.3333
- AOD gap: 0.3333
- Disparate impact ratio: 0.1875
- Fairness aggregate: 0.5399

## Robustness Aggregation
- Noise AUC ratio: 0.9966
- Dropout AUC ratio: 0.8941
- Missingness AUC ratio: 1.0000
- Robustness aggregate: 0.9636

## Rank-Based Robustness: RGR / AURGR
- AURGR Gaussian Noise: 0.9664
- AURGR Percentile Swapping: 0.8162
- RGR Aggregate: 0.8913
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
| performance_auditor | 0.78     |
| fairness_auditor    | 0.539931 |
| robustness_auditor  | 0.963573 |

- Final ensemble SAFE score: 0.6967
- Ensemble rule: weighted aggregation of independent performance, fairness, and robustness auditors.

## Mitigation Experiment
- Mitigation type: group-aware threshold adjustment
- Disadvantaged group detected: male mar/wid
- Baseline fairness aggregate: 0.5399
- Mitigated fairness aggregate: 0.5399
- Baseline SAFE score: 0.6967
- Mitigated SAFE score: 0.6655

### Group Table
| group              |   n |   positive_rate |      tpr |       fpr |
|:-------------------|----:|----------------:|---------:|----------:|
| female div/dep/mar |  61 |       0.245902  | 0.45     | 0.146341  |
| male div/sep       |   9 |       0.444444  | 0.666667 | 0.333333  |
| male mar/wid       |  12 |       0.0833333 | 0.333333 | 0         |
| male single        | 118 |       0.20339   | 0.470588 | 0.0952381 |

## Sensitivity Analysis Summary
Top scenarios by SAFE score:
| scenario                         |   prediction_threshold |   approval_threshold |   w_auc |   w_fair |   w_rob | sensitive_feature   |   auc |   fairness_aggregate |   robustness_aggregate |   safe_score | decision   |   delta_vs_base |
|:---------------------------------|-----------------------:|---------------------:|--------:|---------:|--------:|:--------------------|------:|---------------------:|-----------------------:|-------------:|:-----------|----------------:|
| weights=(0.30,0.30,0.40)         |                   0.55 |                 0.75 |     0.3 |      0.3 |     0.4 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.781409 | APPROVED   |       0.0847286 |
| sensitive_feature=foreign_worker |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | foreign_worker      |  0.78 |             0.706034 |               0.963573 |     0.779732 | APPROVED   |       0.0830518 |
| weights=(0.50,0.30,0.20)         |                   0.55 |                 0.75 |     0.5 |      0.3 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.744694 | REJECTED   |       0.0480139 |
| approval_threshold=0.7           |                   0.55 |                 0.7  |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| approval_threshold=0.75          |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| approval_threshold=0.8           |                   0.55 |                 0.8  |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| base                             |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| prediction_threshold=0.5         |                   0.5  |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |

## Interaction / Effects Summary
- Baseline SAFE score: 0.6967
- Best scenario from sensitivity analysis: weights=(0.30,0.30,0.40)
- Best scenario SAFE score: 0.7814
- Strongest observed effect beyond baseline: weights=(0.30,0.30,0.40)
- Effect size vs baseline: 0.0847
- Interpretation: the governance decision is sensitive to policy weights and sensitive-feature choice, while threshold changes had weaker effects in this run.

## Global Interaction Analysis
Top main effects on SAFE score:
| factor               |   mean_effect_range |
|:---------------------|--------------------:|
| w_fair               |           0.0509132 |
| w_rob                |           0.0367147 |
| prediction_threshold |           0.0231944 |
| approval_threshold   |           0         |

Top pairwise interactions:
| factor_a             | factor_b           |   interaction_strength |
|:---------------------|:-------------------|-----------------------:|
| prediction_threshold | w_fair             |            0.00144965  |
| prediction_threshold | w_rob              |            1.11022e-16 |
| w_fair               | w_rob              |            8.63507e-17 |
| prediction_threshold | approval_threshold |            2.77556e-17 |
| approval_threshold   | w_fair             |            0           |
| approval_threshold   | w_rob              |            0           |

Interpretation:
- Main effects show which single factor most strongly changes SAFE score on average.
- Pairwise interactions show which pairs of factors jointly influence the SAFE decision beyond their separate average effects.

## Rank-Based Explainability: RGE / AURGE
- AURGE: 0.9658
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

Top 10 least important processed features by RGE:
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

## Explainability Snapshot: XGBoost Feature Importance
Top 10 most important processed features by XGBoost importance:
| feature                                     |   importance |
|:--------------------------------------------|-------------:|
| checking_status_no checking                 |    0.119638  |
| property_magnitude_no known property        |    0.0324392 |
| credit_history_all paid                     |    0.027828  |
| property_magnitude_real estate              |    0.0261974 |
| savings_status_no known savings             |    0.0248817 |
| savings_status_less_than_100                |    0.0246545 |
| other_payment_plans_bank                    |    0.0245591 |
| other_parties_guarantor                     |    0.0235592 |
| savings_status_100less_than_=Xless_than_500 |    0.0232596 |
| duration                                    |    0.0227374 |

## SAFE AI Paper Metrics: Multi-Model Compliance Comparison
- AURGA: 0.7252
- AURGR Gaussian Noise: 0.9664
- AURGR Percentile Swapping: 0.8162
- AURGE: 0.9658
- SHAP-RGE Spearman correlation: 0.9359205941635693

Model metrics comparison:
| Model               |    AURGA |   AURGR_Gaussian |   AURGR_Swapping |    AURGR |    AURGE |
|:--------------------|---------:|-----------------:|-----------------:|---------:|---------:|
| Logistic Regression | 0.64564  |         0.996214 |         0.86436  | 0.930287 | 0.96933  |
| Random Forest       | 0.727355 |         0.990898 |         0.879695 | 0.935296 | 0.96643  |
| XGBoost             | 0.725155 |         0.966399 |         0.816184 | 0.891291 | 0.965834 |
| Voting Ensemble     | 0.720947 |         0.99164  |         0.864491 | 0.928066 | 0.971187 |
| Stacking Ensemble   | 0.713829 |         0.994305 |         0.869086 | 0.931696 | 0.972548 |
| Random Baseline     | 0.5      |         1        |         1        | 1        | 1        |

Compliance score comparison:
| Model               |    AURGA |   AURGR_Gaussian |   AURGR_Swapping |    AURGR |    AURGE |   Compliance_Arithmetic |   Compliance_Geometric |   Compliance_RMS |   Compliance_TOPSIS |
|:--------------------|---------:|-----------------:|-----------------:|---------:|---------:|------------------------:|-----------------------:|-----------------:|--------------------:|
| Random Forest       | 0.727355 |         0.990898 |         0.879695 | 0.935296 | 0.96643  |                0.876361 |               0.869538 |         0.882763 |            0.814536 |
| Voting Ensemble     | 0.720947 |         0.99164  |         0.864491 | 0.928066 | 0.971187 |                0.8734   |               0.866154 |         0.880203 |            0.798708 |
| Stacking Ensemble   | 0.713829 |         0.994305 |         0.869086 | 0.931696 | 0.972548 |                0.872691 |               0.864821 |         0.880049 |            0.798014 |
| XGBoost             | 0.725155 |         0.966399 |         0.816184 | 0.891291 | 0.965834 |                0.86076  |               0.854642 |         0.866619 |            0.732732 |
| Logistic Regression | 0.64564  |         0.996214 |         0.86436  | 0.930287 | 0.96933  |                0.848419 |               0.835013 |         0.860598 |            0.601374 |
| Random Baseline     | 0.5      |         1        |         1        | 1        | 1        |                0.833333 |               0.793701 |         0.866025 |            0.265298 |

Interpretation:
- AURGA evaluates rank-based accuracy under progressive data removal.
- AURGR evaluates rank-based robustness under increasing perturbation intensity.
- AURGE evaluates rank-based explainability under progressive feature removal.
- The final Compliance Score combines AURGA, AURGR, and AURGE using Arithmetic Mean, Geometric Mean, RMS, and TOPSIS.

## Auditor Notes
- Multi-metric fairness and robustness aggregation are enabled.
- Sensitivity analysis covers thresholds, weights, alternative sensitive features, and perturbation settings.

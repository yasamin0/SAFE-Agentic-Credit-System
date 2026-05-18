# Final SAFE Agentic Credit Scoring Report

## User Controls
- Data source: CSV (data/raw/raw_credit_data.csv)
- Prediction threshold: 0.55
- Approval threshold: 0.75
- Weights: RGA=0.250, RGR=0.250, RGE=0.250, Fairness=0.250
- Sensitive feature: personal_status
- Drop sensitive from model: False
- Decision rule: APPROVED if SAFE_SCORE >= APPROVAL_THRESHOLD else REJECTED, where SAFE_SCORE = W_RGA*AURGA + W_RGR*RGR_AGG + W_RGE*AURGE + W_FAIR*FAIRNESS_AGG

## SAFE Model Selection

The system first trained multiple candidate models and selected the top candidates by cross-validation AUC. It then computed paper-based SAFE governance metrics for the top candidates, including AURGA, RGR Aggregate, AURGE, Fairness Aggregate, and paper-based SAFE score.

Selected operational governance model: Voting Ensemble

SAFE model selection comparison:
| model             |   cv_auc |   test_auc |    aurga |   rgr_aggregate |    aurge |   fairness_aggregate |   paper_safe_score |   baseline_safe_score | decision   |
|:------------------|---------:|-----------:|---------:|----------------:|---------:|---------------------:|-------------------:|----------------------:|:-----------|
| Voting Ensemble   | 0.782489 |   0.804762 | 0.709529 |        0.928066 | 0.971187 |             0.539931 |           0.787178 |              0.787178 | APPROVED   |
| Stacking Ensemble | 0.779621 |   0.808571 | 0.701846 |        0.931696 | 0.972548 |             0.539931 |           0.786505 |              0.786505 | APPROVED   |
| XGBoost           | 0.78137  |   0.776667 | 0.690258 |        0.914455 | 0.981596 |             0.536254 |           0.780641 |              0.780641 | APPROVED   |
| Random Forest     | 0.788969 |   0.805833 | 0.7211   |        0.959478 | 0.982013 |             0.388889 |           0.76287  |              0.76287  | APPROVED   |

SAFE model selection artifacts:
- SAFE model selection CSV: safe_model_selection_comparison.csv
- SAFE model selection plot: safe_model_selection_comparison.png
- SAFE model selection report: safe_model_selection_report.md

## Top Models SHAP-RGE Comparison

The system also compares RGE-based feature importance with SHAP-based feature importance for the top four selected candidate models. This makes the explainability comparison broader than the selected operational model alone.

Top-model SHAP-RGE summary:
| model             | status    | shap_method      |   sample_size |   rge_shap_spearman | error   |
|:------------------|:----------|:-----------------|--------------:|--------------------:|:--------|
| Voting Ensemble   | completed | kernel_explainer |           100 |            0.899788 |         |
| Stacking Ensemble | completed | kernel_explainer |           100 |            0.887943 |         |
| XGBoost           | completed | xgboost_treeshap |           100 |            0.982526 |         |
| Random Forest     | completed | tree_explainer   |           100 |            0.914754 |         |

Top-model SHAP-RGE artifacts:
- Comparison CSV: top_models_shap_rge_comparison.csv
- Report: top_models_shap_rge_report.md

## Accuracy
- AUC: 0.8048

## Classification Metrics
- PR-AUC: 0.6486
- Precision: 0.6591
- Recall: 0.4833
- F1 Score: 0.5577
- Brier Score: 0.1592

Confusion matrix:
|          |   pred_0 |   pred_1 |
|:---------|---------:|---------:|
| actual_0 |      125 |       15 |
| actual_1 |       31 |       29 |

Calibration curve data:
|   mean_predicted_probability |   fraction_of_positives |
|-----------------------------:|------------------------:|
|                    0.0615604 |                0.0625   |
|                    0.153323  |                0.102564 |
|                    0.253982  |                0.125    |
|                    0.356361  |                0.5      |
|                    0.447575  |                0.310345 |
|                    0.552489  |                0.533333 |
|                    0.661947  |                0.571429 |
|                    0.739855  |                0.705882 |
|                    0.823064  |                0.8      |
|                    0.901698  |                1        |

## Fairness Aggregation
- SPD gap: 0.3611
- EOD gap: 0.3333
- AOD gap: 0.3333
- Disparate impact ratio: 0.1875
- Fairness aggregate: 0.5399

## Robustness Aggregation
- Noise AUC ratio: 0.9941
- Dropout AUC ratio: 0.9036
- Missingness AUC ratio: 0.9988
- Robustness aggregate: 0.9655

## Rank-Based Robustness: RGR / AURGR
- AURGR Gaussian Noise: 0.9916
- AURGR Percentile Swapping: 0.8645
- RGR Aggregate: 0.9281
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
| performance_auditor | 0.804762 |
| fairness_auditor    | 0.539931 |
| robustness_auditor  | 0.965483 |

- Legacy auditor table is reported for transparency only.
- Final paper-based SAFE score: 0.7872
- Final SAFE rule: W_RGA*AURGA + W_RGR*RGR_Aggregate + W_RGE*AURGE + W_FAIR*Fairness_Aggregate.

## Mitigation Experiment
- Mitigation type: group-aware threshold search
- Disadvantaged group detected: male mar/wid
- Base threshold: 0.5500
- Selected threshold delta: 0.1500
- Selected adjusted threshold: 0.4000
- Baseline fairness aggregate: 0.5399
- Mitigated fairness aggregate: 0.6650
- Baseline paper-based SAFE score: 0.7872
- Mitigated paper-based SAFE score: 0.8185
- Mitigation report: mitigation_report.md
- Mitigation threshold search CSV: mitigation_threshold_search.csv
- Baseline group table CSV: mitigation_group_table_before.csv
- Mitigated group table CSV: mitigation_group_table_after.csv

### Baseline Group Table
| group              |   n |   positive_rate |      tpr |       fpr |
|:-------------------|----:|----------------:|---------:|----------:|
| female div/dep/mar |  61 |       0.278689  | 0.6      | 0.121951  |
| male div/sep       |   9 |       0.444444  | 0.666667 | 0.333333  |
| male mar/wid       |  12 |       0.0833333 | 0.333333 | 0         |
| male single        | 118 |       0.186441  | 0.411765 | 0.0952381 |

### Mitigated Group Table
| group              |   n |   positive_rate |      tpr |       fpr |
|:-------------------|----:|----------------:|---------:|----------:|
| female div/dep/mar |  61 |        0.278689 | 0.6      | 0.121951  |
| male div/sep       |   9 |        0.444444 | 0.666667 | 0.333333  |
| male mar/wid       |  12 |        0.416667 | 0.666667 | 0.333333  |
| male single        | 118 |        0.186441 | 0.411765 | 0.0952381 |

### Top Mitigation Candidates
|   delta |   base_threshold |   adjusted_threshold_for_disadvantaged_group | disadvantaged_group   | ranking_metrics_status            |   fairness_aggregate |   spd_gap |   eod_gap |   aod_gap |   dir_ratio |   positive_rate_gap |   safe_score |
|--------:|-----------------:|---------------------------------------------:|:----------------------|:----------------------------------|---------------------:|----------:|----------:|----------:|------------:|--------------------:|-------------:|
|    0.15 |             0.55 |                                         0.4  | male mar/wid          | unchanged_by_threshold_mitigation |             0.665022 |  0.258004 |  0.254902 |  0.246499 |    0.419492 |            0.258004 |     0.818451 |
|    0.2  |             0.55 |                                         0.35 | male mar/wid          | unchanged_by_threshold_mitigation |             0.665022 |  0.258004 |  0.254902 |  0.246499 |    0.419492 |            0.258004 |     0.818451 |
|    0.02 |             0.55 |                                         0.53 | male mar/wid          | unchanged_by_threshold_mitigation |             0.637051 |  0.277778 |  0.254902 |  0.294118 |    0.375    |            0.277778 |     0.811458 |
|    0.05 |             0.55 |                                         0.5  | male mar/wid          | unchanged_by_threshold_mitigation |             0.637051 |  0.277778 |  0.254902 |  0.294118 |    0.375    |            0.277778 |     0.811458 |
|    0.08 |             0.55 |                                         0.47 | male mar/wid          | unchanged_by_threshold_mitigation |             0.637051 |  0.277778 |  0.254902 |  0.294118 |    0.375    |            0.277778 |     0.811458 |
|    0.1  |             0.55 |                                         0.45 | male mar/wid          | unchanged_by_threshold_mitigation |             0.637051 |  0.277778 |  0.254902 |  0.294118 |    0.375    |            0.277778 |     0.811458 |
|    0    |             0.55 |                                         0.55 | male mar/wid          | unchanged_by_threshold_mitigation |             0.539931 |  0.361111 |  0.333333 |  0.333333 |    0.1875   |            0.361111 |     0.787178 |
### Group Table
| group              |   n |   positive_rate |      tpr |       fpr |
|:-------------------|----:|----------------:|---------:|----------:|
| female div/dep/mar |  61 |       0.278689  | 0.6      | 0.121951  |
| male div/sep       |   9 |       0.444444  | 0.666667 | 0.333333  |
| male mar/wid       |  12 |       0.0833333 | 0.333333 | 0         |
| male single        | 118 |       0.186441  | 0.411765 | 0.0952381 |

## Sensitivity Analysis Summary
Top scenarios by SAFE score:
| scenario                         |   prediction_threshold |   approval_threshold |   w_rga |   w_rgr |   w_rge |   w_fair | sensitive_feature   |    aurga |   rgr_aggregate |    aurge |   fairness_aggregate |   safe_score | decision   |   delta_vs_base |
|:---------------------------------|-----------------------:|---------------------:|--------:|--------:|--------:|---------:|:--------------------|---------:|----------------:|---------:|---------------------:|-------------:|:-----------|----------------:|
| sensitive_feature=foreign_worker |                   0.55 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | foreign_worker      | 0.709529 |        0.928066 | 0.971187 |             0.713316 |     0.830524 | APPROVED   |       0.0433464 |
| weights=(0.20,0.30,0.25,0.25)    |                   0.55 |                 0.75 |    0.2  |    0.3  |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.798105 | APPROVED   |       0.0109268 |
| approval_threshold=0.7           |                   0.55 |                 0.7  |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |       0         |
| approval_threshold=0.75          |                   0.55 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |       0         |
| approval_threshold=0.8           |                   0.55 |                 0.8  |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | REJECTED   |       0         |
| base                             |                   0.55 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |       0         |
| prediction_threshold=0.55        |                   0.55 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |       0         |
| prediction_threshold=0.6         |                   0.6  |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |       0         |

## Interaction / Effects Summary
- Baseline paper-based SAFE score: 0.7872
- Best scenario from sensitivity analysis: sensitive_feature=foreign_worker
- Best scenario SAFE score: 0.8305
- Strongest observed effect beyond baseline: sensitive_feature=foreign_worker
- Effect size vs baseline: 0.0433
- Interpretation: the governance decision is sensitive to policy weights and sensitive-feature choice, while threshold changes had weaker effects in this run.

## Global Interaction Analysis
Top main effects on SAFE score:
| factor               |   mean_effect_range |
|:---------------------|--------------------:|
| w_fair               |           0.114264  |
| w_rge                |           0.0482471 |
| w_rgr                |           0.043935  |
| prediction_threshold |           0.0311223 |

Top pairwise interactions:
| factor_a             | factor_b   |   interaction_strength |
|:---------------------|:-----------|-----------------------:|
| prediction_threshold | w_fair     |            0.00343711  |
| prediction_threshold | w_rgr      |            0.00170718  |
| prediction_threshold | w_rge      |            0.00170718  |
| prediction_threshold | w_rga      |            0.00170718  |
| approval_threshold   | w_rgr      |            1.11022e-16 |
| w_rga                | w_rgr      |            1.11022e-16 |

Interpretation:
- Main effects show which single factor most strongly changes SAFE score on average.
- Pairwise interactions show which pairs of factors jointly influence the SAFE decision beyond their separate average effects.

## Rank-Based Explainability: RGE / AURGE
- AURGE: 0.9712
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

Top 10 least important processed features by RGE:
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

## Explainability Snapshot: XGBoost Feature Importance
Top 10 most important processed features by XGBoost importance:
| feature                                    |   importance |
|:-------------------------------------------|-------------:|
| checking_status_0less_than_=Xless_than_200 |            0 |
| personal_status_male mar/wid               |            0 |
| other_parties_co applicant                 |            0 |
| other_parties_guarantor                    |            0 |
| other_parties_none                         |            0 |
| property_magnitude_car                     |            0 |
| property_magnitude_life insurance          |            0 |
| property_magnitude_no known property       |            0 |
| property_magnitude_real estate             |            0 |
| other_payment_plans_bank                   |            0 |

## SAFE AI Paper Metrics: Multi-Model Compliance Comparison
- AURGA: 0.7095
- AURGR Gaussian Noise: 0.9916
- AURGR Percentile Swapping: 0.8645
- AURGE: 0.9712
- SHAP-RGE Spearman correlation: 0.908143839238498

Model metrics comparison:
| Model               |    AURGA |   AURGR_Gaussian |   AURGR_Swapping |    AURGR |    AURGE |
|:--------------------|---------:|-----------------:|-----------------:|---------:|---------:|
| Logistic Regression | 0.640236 |         0.996301 |         0.846401 | 0.921351 | 0.974049 |
| Random Forest       | 0.7211   |         0.99553  |         0.923426 | 0.959478 | 0.982013 |
| XGBoost             | 0.690258 |         0.981441 |         0.847468 | 0.914455 | 0.981596 |
| Voting Ensemble     | 0.709529 |         0.99164  |         0.864491 | 0.928066 | 0.971187 |
| Stacking Ensemble   | 0.701846 |         0.994305 |         0.869086 | 0.931696 | 0.972548 |
| Random Baseline     | 0.5      |         1        |         1        | 1        | 1        |

Compliance score comparison:
| Model               |    AURGA |   AURGR_Gaussian |   AURGR_Swapping |    AURGR |    AURGE |   AURGR_for_compliance |   AURGE_for_compliance |   Compliance_Arithmetic |   Compliance_Geometric |   Compliance_RMS |   Compliance_TOPSIS |
|:--------------------|---------:|-----------------:|-----------------:|---------:|---------:|-----------------------:|-----------------------:|------------------------:|-----------------------:|-----------------:|--------------------:|
| Random Forest       | 0.7211   |         0.99553  |         0.923426 | 0.959478 | 0.982013 |               0.959478 |               0.982013 |                0.88753  |               0.879122 |         0.895346 |            1        |
| Voting Ensemble     | 0.709529 |         0.99164  |         0.864491 | 0.928066 | 0.971187 |               0.928066 |               0.971187 |                0.869594 |               0.861557 |         0.877105 |            0.949168 |
| Stacking Ensemble   | 0.701846 |         0.994305 |         0.869086 | 0.931696 | 0.972548 |               0.931696 |               0.972548 |                0.868696 |               0.859955 |         0.87683  |            0.946124 |
| XGBoost             | 0.690258 |         0.981441 |         0.847468 | 0.914455 | 0.981596 |               0.914455 |               0.981596 |                0.862103 |               0.852515 |         0.871056 |            0.916847 |
| Logistic Regression | 0.640236 |         0.996301 |         0.846401 | 0.921351 | 0.974049 |               0.921351 |               0.974049 |                0.845212 |               0.831346 |         0.857819 |            0.850837 |
| Random Baseline     | 0.5      |         1        |         1        | 1        | 1        |               0.5      |               0.5      |                0.5      |               0.5      |         0.5      |            0        |

Interpretation:
- AURGA evaluates rank-based accuracy under progressive data removal.
- AURGR evaluates rank-based robustness under increasing perturbation intensity.
- AURGE evaluates rank-based explainability under progressive feature removal.
- The final Compliance Score combines AURGA, AURGR, and AURGE using Arithmetic Mean, Geometric Mean, RMS, and TOPSIS.

## Auditor Notes
- Multi-metric fairness and robustness aggregation are enabled.
- Sensitivity analysis covers thresholds, weights, alternative sensitive features, and perturbation settings.

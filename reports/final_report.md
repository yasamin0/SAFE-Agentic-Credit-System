# Final SAFE Agentic Credit Scoring Report

## User Controls
- Data source: CSV (data/raw/raw_credit_data.csv)
- Prediction threshold: 0.55
- Approval threshold: 0.75
- Weights: AUC=0.300, Fairness=0.500, Robustness=0.200
- Sensitive feature: personal_status
- Drop sensitive from model: False
- Decision rule: APPROVED if SAFE_SCORE >= APPROVAL_THRESHOLD else REJECTED, where SAFE_SCORE = W_AUC*AUC + W_FAIR*FAIRNESS_AGG + W_ROB*ROBUSTNESS_AGG

## SAFE Model Selection

The system first trained multiple candidate models and selected the top candidates by cross-validation AUC. It then computed core SAFE governance metrics for the top candidates, including test AUC, fairness aggregate, robustness aggregate, and baseline SAFE score.

Selected operational governance model: Stacking Ensemble

SAFE model selection comparison:
| model             |   cv_auc |   test_auc |   fairness_aggregate |   robustness_aggregate |   baseline_safe_score | decision   |
|:------------------|---------:|-----------:|---------------------:|-----------------------:|----------------------:|:-----------|
| Stacking Ensemble | 0.779621 |   0.808571 |             0.539931 |               0.965548 |              0.705646 | REJECTED   |
| Voting Ensemble   | 0.782489 |   0.804762 |             0.539931 |               0.965483 |              0.70449  | REJECTED   |
| XGBoost           | 0.78137  |   0.776667 |             0.536254 |               0.956571 |              0.692441 | REJECTED   |
| Random Forest     | 0.788969 |   0.805833 |             0.388889 |               0.958438 |              0.627882 | REJECTED   |

SAFE model selection artifacts:
- SAFE model selection CSV: safe_model_selection_comparison.csv
- SAFE model selection plot: safe_model_selection_comparison.png
- SAFE model selection report: safe_model_selection_report.md

## Top Models SHAP-RGE Comparison

The system also compares RGE-based feature importance with SHAP-based feature importance for the top four selected candidate models. This makes the explainability comparison broader than the selected operational model alone.

Top-model SHAP-RGE summary:
| model             | status    | shap_method      |   sample_size |   rge_shap_spearman | error   |
|:------------------|:----------|:-----------------|--------------:|--------------------:|:--------|
| Stacking Ensemble | completed | kernel_explainer |           100 |            0.899577 |         |
| Voting Ensemble   | completed | kernel_explainer |           100 |            0.897462 |         |
| XGBoost           | completed | xgboost_treeshap |           100 |            0.982526 |         |
| Random Forest     | completed | tree_explainer   |           100 |            0.914754 |         |

Top-model SHAP-RGE artifacts:
- Comparison CSV: top_models_shap_rge_comparison.csv
- Report: top_models_shap_rge_report.md

## Accuracy
- AUC: 0.8086

## Classification Metrics
- PR-AUC: 0.6560
- Precision: 0.6842
- Recall: 0.4333
- F1 Score: 0.5306
- Brier Score: 0.1547

Confusion matrix:
|          |   pred_0 |   pred_1 |
|:---------|---------:|---------:|
| actual_0 |      128 |       12 |
| actual_1 |       34 |       26 |

Calibration curve data:
|   mean_predicted_probability |   fraction_of_positives |
|-----------------------------:|------------------------:|
|                    0.0819856 |               0.0540541 |
|                    0.146486  |               0.109091  |
|                    0.230525  |               0.235294  |
|                    0.341038  |               0.357143  |
|                    0.438169  |               0.411765  |
|                    0.539621  |               0.538462  |
|                    0.658464  |               0.7       |
|                    0.750915  |               0.75      |
|                    0.843729  |               1         |

## Fairness Aggregation
- SPD gap: 0.3611
- EOD gap: 0.3333
- AOD gap: 0.3333
- Disparate impact ratio: 0.1875
- Fairness aggregate: 0.5399

## Robustness Aggregation
- Noise AUC ratio: 0.9962
- Dropout AUC ratio: 0.9024
- Missingness AUC ratio: 0.9981
- Robustness aggregate: 0.9655

## Rank-Based Robustness: RGR / AURGR
- AURGR Gaussian Noise: 0.9943
- AURGR Percentile Swapping: 0.8691
- RGR Aggregate: 0.9317
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
| performance_auditor | 0.808571 |
| fairness_auditor    | 0.539931 |
| robustness_auditor  | 0.965548 |

- Final ensemble SAFE score: 0.7056
- Ensemble rule: weighted aggregation of independent performance, fairness, and robustness auditors.

## Mitigation Experiment
- Mitigation type: group-aware threshold search
- Disadvantaged group detected: male mar/wid
- Base threshold: 0.5500
- Selected threshold delta: 0.2000
- Selected adjusted threshold: 0.3500
- Baseline fairness aggregate: 0.5399
- Mitigated fairness aggregate: 0.6304
- Baseline SAFE score: 0.7056
- Mitigated SAFE score: 0.7509
- Mitigation report: mitigation_report.md
- Mitigation threshold search CSV: mitigation_threshold_search.csv
- Baseline group table CSV: mitigation_group_table_before.csv
- Mitigated group table CSV: mitigation_group_table_after.csv

### Baseline Group Table
| group              |   n |   positive_rate |      tpr |       fpr |
|:-------------------|----:|----------------:|---------:|----------:|
| female div/dep/mar |  61 |       0.229508  | 0.5      | 0.097561  |
| male div/sep       |   9 |       0.444444  | 0.666667 | 0.333333  |
| male mar/wid       |  12 |       0.0833333 | 0.333333 | 0         |
| male single        | 118 |       0.161017  | 0.382353 | 0.0714286 |

### Mitigated Group Table
| group              |   n |   positive_rate |      tpr |       fpr |
|:-------------------|----:|----------------:|---------:|----------:|
| female div/dep/mar |  61 |        0.229508 | 0.5      | 0.097561  |
| male div/sep       |   9 |        0.444444 | 0.666667 | 0.333333  |
| male mar/wid       |  12 |        0.25     | 0.666667 | 0.111111  |
| male single        | 118 |        0.161017 | 0.382353 | 0.0714286 |

### Top Mitigation Candidates
|   delta |   base_threshold |   adjusted_threshold_for_disadvantaged_group | disadvantaged_group   |   auc_probability_based |   fairness_aggregate |   spd_gap |   eod_gap |   aod_gap |   dir_ratio |   positive_rate_gap |   safe_score |
|--------:|-----------------:|---------------------------------------------:|:----------------------|------------------------:|---------------------:|----------:|----------:|----------:|------------:|--------------------:|-------------:|
|    0.2  |             0.55 |                                         0.35 | male mar/wid          |                0.808571 |             0.630359 |  0.283427 |  0.284314 |  0.273109 |    0.362288 |            0.283427 |     0.750861 |
|    0.05 |             0.55 |                                         0.5  | male mar/wid          |                0.808571 |             0.621431 |  0.283427 |  0.284314 |  0.308824 |    0.362288 |            0.283427 |     0.746396 |
|    0.08 |             0.55 |                                         0.47 | male mar/wid          |                0.808571 |             0.621431 |  0.283427 |  0.284314 |  0.308824 |    0.362288 |            0.283427 |     0.746396 |
|    0.1  |             0.55 |                                         0.45 | male mar/wid          |                0.808571 |             0.621431 |  0.283427 |  0.284314 |  0.308824 |    0.362288 |            0.283427 |     0.746396 |
|    0.15 |             0.55 |                                         0.4  | male mar/wid          |                0.808571 |             0.621431 |  0.283427 |  0.284314 |  0.308824 |    0.362288 |            0.283427 |     0.746396 |
|    0    |             0.55 |                                         0.55 | male mar/wid          |                0.808571 |             0.539931 |  0.361111 |  0.333333 |  0.333333 |    0.1875   |            0.361111 |     0.705646 |
|    0.02 |             0.55 |                                         0.53 | male mar/wid          |                0.808571 |             0.539931 |  0.361111 |  0.333333 |  0.333333 |    0.1875   |            0.361111 |     0.705646 |
### Group Table
| group              |   n |   positive_rate |      tpr |       fpr |
|:-------------------|----:|----------------:|---------:|----------:|
| female div/dep/mar |  61 |       0.229508  | 0.5      | 0.097561  |
| male div/sep       |   9 |       0.444444  | 0.666667 | 0.333333  |
| male mar/wid       |  12 |       0.0833333 | 0.333333 | 0         |
| male single        | 118 |       0.161017  | 0.382353 | 0.0714286 |

## Sensitivity Analysis Summary
Top scenarios by SAFE score:
| scenario                         |   prediction_threshold |   approval_threshold |   w_auc |   w_fair |   w_rob | sensitive_feature   |      auc |   fairness_aggregate |   robustness_aggregate |   safe_score | decision   |   delta_vs_base |
|:---------------------------------|-----------------------:|---------------------:|--------:|---------:|--------:|:--------------------|---------:|---------------------:|-----------------------:|-------------:|:-----------|----------------:|
| sensitive_feature=foreign_worker |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | foreign_worker      | 0.808571 |             0.735242 |               0.965548 |     0.803302 | APPROVED   |       0.0976557 |
| weights=(0.30,0.30,0.40)         |                   0.55 |                 0.75 |     0.3 |      0.3 |     0.4 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.79077  | APPROVED   |       0.0851234 |
| weights=(0.50,0.30,0.20)         |                   0.55 |                 0.75 |     0.5 |      0.3 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.759374 | APPROVED   |       0.0537282 |
| prediction_threshold=0.6         |                   0.6  |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.604167 |               0.965548 |     0.737764 | REJECTED   |       0.0321181 |
| approval_threshold=0.7           |                   0.55 |                 0.7  |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.705646 | APPROVED   |       0         |
| approval_threshold=0.75          |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.705646 | REJECTED   |       0         |
| approval_threshold=0.8           |                   0.55 |                 0.8  |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.705646 | REJECTED   |       0         |
| base                             |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.705646 | REJECTED   |       0         |

## Interaction / Effects Summary
- Baseline SAFE score: 0.7056
- Best scenario from sensitivity analysis: sensitive_feature=foreign_worker
- Best scenario SAFE score: 0.8033
- Strongest observed effect beyond baseline: sensitive_feature=foreign_worker
- Effect size vs baseline: 0.0977
- Interpretation: the governance decision is sensitive to policy weights and sensitive-feature choice, while threshold changes had weaker effects in this run.

## Global Interaction Analysis
Top main effects on SAFE score:
| factor               |   mean_effect_range |
|:---------------------|--------------------:|
| prediction_threshold |           0.079902  |
| w_fair               |           0.0640682 |
| w_rob                |           0.0313953 |
| approval_threshold   |           0         |

Top pairwise interactions:
| factor_a             | factor_b           |   interaction_strength |
|:---------------------|:-------------------|-----------------------:|
| prediction_threshold | w_fair             |            0.00558789  |
| prediction_threshold | w_rob              |            1.66533e-16 |
| prediction_threshold | approval_threshold |            2.77556e-17 |
| w_fair               | w_rob              |            1.23358e-17 |
| approval_threshold   | w_fair             |            0           |
| approval_threshold   | w_rob              |            0           |

Interpretation:
- Main effects show which single factor most strongly changes SAFE score on average.
- Pairwise interactions show which pairs of factors jointly influence the SAFE decision beyond their separate average effects.

## Rank-Based Explainability: RGE / AURGE
- AURGE: 0.9725
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

Top 10 least important processed features by RGE:
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
- AURGA: 0.7018
- AURGR Gaussian Noise: 0.9943
- AURGR Percentile Swapping: 0.8691
- AURGE: 0.9725
- SHAP-RGE Spearman correlation: 0.8902696985721839

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

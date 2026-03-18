# Final SAFE Agentic Credit Scoring Report

## User Controls
- Data source: CSV (raw_credit_data.csv)
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

## Explainability Snapshot
Top 10 most important processed features:
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

## Auditor Notes
- Multi-metric fairness and robustness aggregation are enabled.
- Sensitivity analysis covers thresholds, weights, alternative sensitive features, and perturbation settings.

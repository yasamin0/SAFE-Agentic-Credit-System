# System Card — SAFE Agentic Credit Scoring

## Final Governance Decision
**Baseline Decision:** REJECTED

**Baseline SAFE Score:** 0.706

**Mitigated Decision:** APPROVED

**Mitigated SAFE Score:** 0.751

**Approval Threshold:** 0.750

**Governance Conclusion:** The deployment decision remains **REJECTED** under the baseline governance rule. The mitigation result is reported separately as post-processing evidence.

## Decision Logic
The baseline governance decision is approved only if:

`Baseline SAFE Score >= Approval Threshold`

Baseline result:

`0.706 >= 0.750` → **REJECTED**

Mitigated result:

`0.751 >= 0.750` → **APPROVED**

Mitigation interpretation:

Mitigation improved the SAFE Score from 0.7056 to 0.7509. The mitigated decision is APPROVED.
## SAFE Score Formula
`SAFE Score = W_AUC*AUC + W_FAIR*Fairness_Aggregate + W_ROB*Robustness_Aggregate`

Current weights:
- W_AUC = 0.300
- W_FAIR = 0.500
- W_ROB = 0.200

Current computation:
- AUC = 0.8086
- Fairness Aggregate = 0.5399
- Robustness Aggregate = 0.9655
- Final SAFE Score = 0.7056

## Main Reason for Decision
The weakest core dimension is **Fairness Aggregate**.

In this run, the model is rejected because the weighted SAFE score is below the approval threshold.

## Additional Performance Metrics
- PR-AUC: 0.6560
- Precision: 0.6842
- Recall: 0.4333
- F1 Score: 0.5306
- Brier Score: 0.1547

## Fairness Extension
Fairness is kept as an extension for credit lending.

The system evaluates:
- SPD
- EOD
- AOD
- DIR
- Fairness Aggregate
- Group-aware mitigation result

Fairness Aggregate: 0.5399

## Mitigation Result
- Mitigation type: group-aware threshold search
- Selected threshold delta: 0.2000
- Selected adjusted threshold: 0.3500
- Baseline SAFE Score: 0.7056
- Baseline Decision: REJECTED
- Mitigated AUC: 0.8086
- Mitigated Fairness Aggregate: 0.6304
- Mitigated SAFE Score: 0.7509
- Mitigated Decision: APPROVED
- Mitigation summary: Mitigation improved the SAFE Score from 0.7056 to 0.7509. The mitigated decision is APPROVED.

## SAFE AI Paper Metrics
- AURGA: 0.7018
- RGR Aggregate: 0.9317
- AURGE: 0.9725
- SHAP-RGE Spearman Correlation: 0.8903

## Configuration
- Prediction threshold from configuration: 0.55
- Approval threshold from configuration: 0.75
- Sensitive feature: personal_status
- Drop sensitive from model: False
- Random state: 42

## Sensitivity Snapshot
# Sensitivity Analysis Report

Evaluates how SAFE decisions change under variations in weights, thresholds, sensitive feature choice, and perturbation assumptions.

## Scenario Table

| scenario                          |   prediction_threshold |   approval_threshold |   w_auc |   w_fair |   w_rob | sensitive_feature   |      auc |   fairness_aggregate |   robustness_aggregate |   safe_score | decision   |   delta_vs_base |
|:----------------------------------|-----------------------:|---------------------:|--------:|---------:|--------:|:--------------------|---------:|---------------------:|-----------------------:|-------------:|:-----------|----------------:|
| sensitive_feature=foreign_worker  |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | foreign_worker      | 0.808571 |             0.735242 |               0.965548 |     0.803302 | APPROVED   |       0.0976557 |
| weights=(0.30,0.30,0.40)          |                   0.55 |                 0.75 |     0.3 |      0.3 |     0.4 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.79077  | APPROVED   |       0.0851234 |
| weights=(0.50,0.30,0.20)          |                   0.55 |                 0.75 |     0.5 |      0.3 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.759374 | APPROVED   |       0.0537282 |
| prediction_threshold=0.6          |                   0.6  |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.604167 |               0.965548 |     0.737764 | REJECTED   |       0.0321181 |
| approval_threshold=0.7            |                   0.55 |                 0.7  |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.705646 | APPROVED   |       0         |
| approval_threshold=0.75           |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.705646 | REJECTED   |       0         |
| approval_threshold=0.8            |                   0.55 |                 0.8  |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.705646 | REJECTED   |       0         |
| base                              |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.705646 | REJECTED   |       0         |
| prediction_threshold=0.55         |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.705646 | REJECTED   |       0         |
| sensitive_feature=personal_status |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.808571 |             0.539931 |               0.965548 |     0.705646 | REJECTED   |       0         |

## Governance Note
This card separates two concepts:
1. **SAFE Score**, which is the project governance score using AUC, fairness, and robustness.
2. **Compliance Score**, which is the SAFE AI paper-style score using AURGA, AURGR, AURGE, and TOPSIS.

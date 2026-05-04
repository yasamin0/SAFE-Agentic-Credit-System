# System Card — SAFE Agentic Credit Scoring

## Final Governance Decision
**Decision:** REJECTED

**Final SAFE Score:** 0.692

**Approval Threshold:** 0.750

## Decision Logic
The model is approved only if:

`SAFE Score >= Approval Threshold`

Current result:

`0.692 >= 0.750` → **REJECTED**

## SAFE Score Formula
`SAFE Score = W_AUC*AUC + W_FAIR*Fairness_Aggregate + W_ROB*Robustness_Aggregate`

Current weights:
- W_AUC = 0.300
- W_FAIR = 0.500
- W_ROB = 0.200

Current computation:
- AUC = 0.7767
- Fairness Aggregate = 0.5363
- Robustness Aggregate = 0.9566
- Final SAFE Score = 0.6925

## Main Reason for Decision
The weakest core dimension is **Fairness Aggregate**.

In this run, the model is rejected because the weighted SAFE score is below the approval threshold.

## Additional Performance Metrics
- PR-AUC: 0.5615
- Precision: 0.6562
- Recall: 0.3500
- F1 Score: 0.4565
- Brier Score: 0.1689

## Fairness Extension
Fairness is kept as an extension for credit lending.

The system evaluates:
- SPD
- EOD
- AOD
- DIR
- Fairness Aggregate
- Group-aware mitigation result

Fairness Aggregate: 0.5363

## Mitigation Result
- Mitigated AUC: 0.6357
- Mitigated Fairness Aggregate: 0.5363
- Mitigated SAFE Score: 0.6502
- Mitigated Decision: REJECTED

## SAFE AI Paper Metrics
- AURGA: 0.6903
- RGR Aggregate: 0.9145
- AURGE: 0.9816
- SHAP-RGE Spearman Correlation: 0.9838

## Configuration
- Prediction threshold: 0.55
- Sensitive feature: personal_status
- Drop sensitive from model: False
- Random state: 42

## Sensitivity Snapshot
# Sensitivity Analysis Report

Evaluates how SAFE decisions change under variations in weights, thresholds, sensitive feature choice, and perturbation assumptions.

## Scenario Table

| scenario                          |   prediction_threshold |   approval_threshold |   w_auc |   w_fair |   w_rob | sensitive_feature   |      auc |   fairness_aggregate |   robustness_aggregate |   safe_score | decision   |   delta_vs_base |
|:----------------------------------|-----------------------:|---------------------:|--------:|---------:|--------:|:--------------------|---------:|---------------------:|-----------------------:|-------------:|:-----------|----------------:|
| sensitive_feature=foreign_worker  |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | foreign_worker      | 0.776667 |             0.729906 |               0.956571 |     0.789267 | APPROVED   |      0.0968259  |
| weights=(0.30,0.30,0.40)          |                   0.55 |                 0.75 |     0.3 |      0.3 |     0.4 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.776504 | APPROVED   |      0.0840633  |
| weights=(0.50,0.30,0.20)          |                   0.55 |                 0.75 |     0.5 |      0.3 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.740524 | REJECTED   |      0.0480825  |
| prediction_threshold=0.6          |                   0.6  |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.548611 |               0.956571 |     0.69862  | REJECTED   |      0.00617851 |
| approval_threshold=0.7            |                   0.55 |                 0.7  |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.692441 | REJECTED   |      0          |
| approval_threshold=0.75           |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.692441 | REJECTED   |      0          |
| approval_threshold=0.8            |                   0.55 |                 0.8  |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.692441 | REJECTED   |      0          |
| base                              |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.692441 | REJECTED   |      0          |
| prediction_threshold=0.55         |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.692441 | REJECTED   |      0          |
| sensitive_feature=personal_status |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     | 0.776667 |             0.536254 |               0.956571 |     0.692441 | REJECTED   |      0          |

## Governance Note
This card separates two concepts:
1. **SAFE Score**, which is the project governance score using AUC, fairness, and robustness.
2. **Compliance Score**, which is the SAFE AI paper-style score using AURGA, AURGR, AURGE, and TOPSIS.

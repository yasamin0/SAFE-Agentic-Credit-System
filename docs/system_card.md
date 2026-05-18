# System Card — SAFE Agentic Credit Scoring

## Final Governance Decision
**Baseline Decision:** APPROVED

**Baseline SAFE Score:** 0.787

**Mitigated Decision:** APPROVED

**Mitigated SAFE Score:** 0.819

**Approval Threshold:** 0.750

**Governance Conclusion:** The deployment decision remains **APPROVED** under the baseline governance rule. The mitigation result is reported separately as post-processing evidence.

## Decision Logic
The baseline governance decision is approved only if:

`Baseline SAFE Score >= Approval Threshold`

Baseline result:

`0.787 >= 0.750` → **APPROVED**

Mitigated result:

`0.819 >= 0.750` → **APPROVED**

Mitigation interpretation:

Mitigation improved the SAFE Score from 0.7872 to 0.8185. The mitigated decision is APPROVED.
## SAFE Score Formula
`SAFE Score = W_RGA*AURGA + W_RGR*RGR_Aggregate + W_RGE*AURGE + W_FAIR*Fairness_Aggregate`

Current weights:
- W_RGA = 0.250
- W_RGR = 0.250
- W_RGE = 0.250
- W_FAIR = 0.250

Current computation:
- AURGA = 0.7095
- RGR Aggregate = 0.9281
- AURGE = 0.9712
- Fairness Aggregate = 0.5399
- Final SAFE Score = 0.7872

## Main Reason for Decision
The weakest core dimension is **Fairness Aggregate**.

In this run, the model is rejected because the weighted SAFE score is below the approval threshold.

## Additional Performance Metrics
- PR-AUC: 0.6486
- Precision: 0.6591
- Recall: 0.4833
- F1 Score: 0.5577
- Brier Score: 0.1592

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
- Selected threshold delta: 0.1500
- Selected adjusted threshold: 0.4000
- Baseline SAFE Score: 0.7872
- Baseline Decision: APPROVED
- Mitigated AUC: 0.8048
- Mitigated Fairness Aggregate: 0.6650
- Mitigated SAFE Score: 0.8185
- Mitigated Decision: APPROVED
- Mitigation summary: Mitigation improved the SAFE Score from 0.7872 to 0.8185. The mitigated decision is APPROVED.

## SAFE AI Paper Metrics
- AURGA: 0.7095
- RGR Aggregate: 0.9281
- AURGE: 0.9712
- SHAP-RGE Spearman Correlation: 0.9081

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

| scenario                          |   prediction_threshold |   approval_threshold |   w_rga |   w_rgr |   w_rge |   w_fair | sensitive_feature   |    aurga |   rgr_aggregate |    aurge |   fairness_aggregate |   safe_score | decision   |   delta_vs_base |
|:----------------------------------|-----------------------:|---------------------:|--------:|--------:|--------:|---------:|:--------------------|---------:|----------------:|---------:|---------------------:|-------------:|:-----------|----------------:|
| sensitive_feature=foreign_worker  |                   0.55 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | foreign_worker      | 0.709529 |        0.928066 | 0.971187 |             0.713316 |     0.830524 | APPROVED   |      0.0433464  |
| weights=(0.20,0.30,0.25,0.25)     |                   0.55 |                 0.75 |    0.2  |    0.3  |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.798105 | APPROVED   |      0.0109268  |
| approval_threshold=0.7            |                   0.55 |                 0.7  |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |      0          |
| approval_threshold=0.75           |                   0.55 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |      0          |
| approval_threshold=0.8            |                   0.55 |                 0.8  |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | REJECTED   |      0          |
| base                              |                   0.55 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |      0          |
| prediction_threshold=0.55         |                   0.55 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |      0          |
| prediction_threshold=0.6          |                   0.6  |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |      0          |
| sensitive_feature=personal_status |                   0.55 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |      0          |
| weights=(0.25,0.25,0.25,0.25)     |                   0.55 |                 0.75 |    0.25 |    0.25 |    0.25 |     0.25 | personal_status     | 0.709529 |        0.928066 | 0.971187 |             0.539931 |     0.787178 | APPROVED   |      0          |

## Governance Note
This card uses the final SAFE score requested for this project:
1. **RGA / AURGA** for rank-based accuracy.
2. **RGR Aggregate** for rank-based robustness.
3. **RGE / AURGE** for rank-based explainability.
4. **Fairness Aggregate** for credit-lending fairness.

All four dimensions use equal weights.

# System Card — SAFE Agentic Credit Scoring

## Decision
**REJECTED**

## Final SAFE Score
**0.697**

## Decision Rule
- SAFE Score = 0.300*AUC + 0.500*Fairness_Aggregate + 0.200*Robustness_Aggregate
- Approval threshold = 0.75
- Policy = APPROVED if SAFE Score >= threshold else REJECTED

## Metrics Used
- Baseline AUC: 0.780
- Baseline Fairness Aggregate: 0.540
- Baseline Robustness Aggregate: 0.964
- Mitigated SAFE Score: 0.665
- Mitigated Decision: REJECTED

## Reproducibility Controls
- Prediction threshold: 0.55
- Sensitive feature: personal_status
- Drop sensitive from model: False
- Random state: 42

## Rationale
The final SAFE score balances predictive performance with multi-metric fairness and multi-scenario robustness.

## Mitigation Result
- Baseline decision: REJECTED
- Mitigated decision: REJECTED
- Interpretation: mitigation is reported separately to show whether fairness-aware post-processing improves the governance outcome under a fixed policy.

## Sensitivity Snapshot
# Sensitivity Analysis Report

Evaluates how SAFE decisions change under variations in weights, thresholds, sensitive feature choice, and perturbation assumptions.

## Scenario Table

| scenario                          |   prediction_threshold |   approval_threshold |   w_auc |   w_fair |   w_rob | sensitive_feature   |   auc |   fairness_aggregate |   robustness_aggregate |   safe_score | decision   |   delta_vs_base |
|:----------------------------------|-----------------------:|---------------------:|--------:|---------:|--------:|:--------------------|------:|---------------------:|-----------------------:|-------------:|:-----------|----------------:|
| weights=(0.30,0.30,0.40)          |                   0.55 |                 0.75 |     0.3 |      0.3 |     0.4 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.781409 | APPROVED   |       0.0847286 |
| sensitive_feature=foreign_worker  |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | foreign_worker      |  0.78 |             0.706034 |               0.963573 |     0.779732 | APPROVED   |       0.0830518 |
| weights=(0.50,0.30,0.20)          |                   0.55 |                 0.75 |     0.5 |      0.3 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.744694 | REJECTED   |       0.0480139 |
| approval_threshold=0.7            |                   0.55 |                 0.7  |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| approval_threshold=0.75           |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| approval_threshold=0.8            |                   0.55 |                 0.8  |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| base                              |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| prediction_threshold=0.5          |                   0.5  |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| prediction_threshold=0.55         |                   0.55 |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |
| prediction_threshold=0.6          |                   0.6  |                 0.75 |     0.3 |      0.5 |     0.2 | personal_status     |  0.78 |             0.539931 |               0.963573 |     0.69668  | REJECTED   |       0         |

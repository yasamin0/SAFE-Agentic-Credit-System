# Final SAFE Agentic Credit Scoring Report

## User Controls
- Data source: CSV (raw_credit_data.csv)
- Prediction threshold: 0.55
- Approval threshold: 0.78
- Weights: AUC=0.300, Fairness=0.500, Robustness=0.200
- Sensitive feature: personal_status

## Accuracy
- AUC: 0.7800

## Robustness (Noise Stress Test)
- Robustness score (AUC_noisy / AUC_base): 0.9789

## Fairness (Group Gaps)
- SPD gap (max-min positive rate): 0.3611
- EOD gap (max-min TPR): 0.3333
- Fairness score (1 - avg gap): 0.6528

### Group Table
| group              |   n |   positive_rate |      tpr |
|:-------------------|----:|----------------:|---------:|
| female div/dep/mar |  61 |       0.245902  | 0.45     |
| male div/sep       |   9 |       0.444444  | 0.666667 |
| male mar/wid       |  12 |       0.0833333 | 0.333333 |
| male single        | 118 |       0.20339   | 0.470588 |

## Auditor Notes
- If fairness score is low: consider threshold tuning or mitigation techniques.
- If robustness score is low: test more perturbations/shifts.

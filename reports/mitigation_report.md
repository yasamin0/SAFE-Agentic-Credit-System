# Mitigation Experiment Report

This report evaluates group-aware threshold mitigation.

## Method
The experiment first identifies the group with the lowest baseline positive prediction rate. It then evaluates several threshold reductions for that group while keeping the other group thresholds unchanged. The selected mitigation is the candidate with the highest SAFE score.

## Selected Mitigation
- Disadvantaged group: male mar/wid
- Base threshold from configuration: 0.5500
- Selected adjusted threshold: 0.4000
- Selected delta: 0.1500
- Baseline AUC: 0.7767
- Mitigated AUC: 0.7767
- Baseline fairness aggregate: 0.5363
- Mitigated fairness aggregate: 0.5945
- Baseline SAFE score: 0.6924
- Mitigated SAFE score: 0.7216

## Baseline Fairness Components
- SPD gap: 0.3611
- EOD gap: 0.3431
- AOD gap: 0.3382
- DIR ratio: 0.1875

## Mitigated Fairness Components
- SPD gap: 0.3004
- EOD gap: 0.3431
- AOD gap: 0.3025
- DIR ratio: 0.3242

## Baseline Group Table

| group              |   n |   positive_rate |      tpr |       fpr |
|:-------------------|----:|----------------:|---------:|----------:|
| female div/dep/mar |  61 |       0.163934  | 0.35     | 0.0731707 |
| male div/sep       |   9 |       0.444444  | 0.666667 | 0.333333  |
| male mar/wid       |  12 |       0.0833333 | 0.333333 | 0         |
| male single        | 118 |       0.144068  | 0.323529 | 0.0714286 |

## Mitigated Group Table

| group              |   n |   positive_rate |      tpr |       fpr |
|:-------------------|----:|----------------:|---------:|----------:|
| female div/dep/mar |  61 |        0.163934 | 0.35     | 0.0731707 |
| male div/sep       |   9 |        0.444444 | 0.666667 | 0.333333  |
| male mar/wid       |  12 |        0.25     | 0.333333 | 0.222222  |
| male single        | 118 |        0.144068 | 0.323529 | 0.0714286 |

## Threshold Search Results

|   delta |   base_threshold |   adjusted_threshold_for_disadvantaged_group | disadvantaged_group   |   auc_probability_based |   fairness_aggregate |   spd_gap |   eod_gap |   aod_gap |   dir_ratio |   positive_rate_gap |   safe_score |
|--------:|-----------------:|---------------------------------------------:|:----------------------|------------------------:|---------------------:|----------:|----------:|----------:|------------:|--------------------:|-------------:|
|    0.15 |             0.55 |                                         0.4  | male mar/wid          |                0.776667 |             0.594529 |  0.300377 |  0.343137 |  0.302521 |    0.324153 |            0.300377 |     0.721579 |
|    0.2  |             0.55 |                                         0.35 | male mar/wid          |                0.776667 |             0.580641 |  0.300377 |  0.343137 |  0.358077 |    0.324153 |            0.300377 |     0.714634 |
|    0    |             0.55 |                                         0.55 | male mar/wid          |                0.776667 |             0.536254 |  0.361111 |  0.343137 |  0.338235 |    0.1875   |            0.361111 |     0.692441 |
|    0.02 |             0.55 |                                         0.53 | male mar/wid          |                0.776667 |             0.536254 |  0.361111 |  0.343137 |  0.338235 |    0.1875   |            0.361111 |     0.692441 |
|    0.05 |             0.55 |                                         0.5  | male mar/wid          |                0.776667 |             0.536254 |  0.361111 |  0.343137 |  0.338235 |    0.1875   |            0.361111 |     0.692441 |
|    0.08 |             0.55 |                                         0.47 | male mar/wid          |                0.776667 |             0.536254 |  0.361111 |  0.343137 |  0.338235 |    0.1875   |            0.361111 |     0.692441 |
|    0.1  |             0.55 |                                         0.45 | male mar/wid          |                0.776667 |             0.536254 |  0.361111 |  0.343137 |  0.338235 |    0.1875   |            0.361111 |     0.692441 |

## Output Files
- Threshold search CSV: mitigation_threshold_search.csv
- Baseline group table CSV: mitigation_group_table_before.csv
- Mitigated group table CSV: mitigation_group_table_after.csv

# Mitigation Experiment Report

This report evaluates group-aware threshold mitigation.

## Method
The experiment first identifies the group with the lowest baseline positive prediction rate. It then evaluates several threshold reductions for that group while keeping the other group thresholds unchanged. The selected mitigation is the candidate with the highest SAFE score.

## Selected Mitigation
- Disadvantaged group: male mar/wid
- Base threshold from configuration: 0.5500
- Selected adjusted threshold: 0.3500
- Selected delta: 0.2000
- Baseline AUC: 0.8086
- Mitigated AUC: 0.8086
- Baseline fairness aggregate: 0.5399
- Mitigated fairness aggregate: 0.6304
- Baseline SAFE score: 0.7056
- Mitigated SAFE score: 0.7509

## Baseline Fairness Components
- SPD gap: 0.3611
- EOD gap: 0.3333
- AOD gap: 0.3333
- DIR ratio: 0.1875

## Mitigated Fairness Components
- SPD gap: 0.2834
- EOD gap: 0.2843
- AOD gap: 0.2731
- DIR ratio: 0.3623

## Baseline Group Table

| group              |   n |   positive_rate |      tpr |       fpr |
|:-------------------|----:|----------------:|---------:|----------:|
| female div/dep/mar |  61 |       0.229508  | 0.5      | 0.097561  |
| male div/sep       |   9 |       0.444444  | 0.666667 | 0.333333  |
| male mar/wid       |  12 |       0.0833333 | 0.333333 | 0         |
| male single        | 118 |       0.161017  | 0.382353 | 0.0714286 |

## Mitigated Group Table

| group              |   n |   positive_rate |      tpr |       fpr |
|:-------------------|----:|----------------:|---------:|----------:|
| female div/dep/mar |  61 |        0.229508 | 0.5      | 0.097561  |
| male div/sep       |   9 |        0.444444 | 0.666667 | 0.333333  |
| male mar/wid       |  12 |        0.25     | 0.666667 | 0.111111  |
| male single        | 118 |        0.161017 | 0.382353 | 0.0714286 |

## Threshold Search Results

|   delta |   base_threshold |   adjusted_threshold_for_disadvantaged_group | disadvantaged_group   |   auc_probability_based |   fairness_aggregate |   spd_gap |   eod_gap |   aod_gap |   dir_ratio |   positive_rate_gap |   safe_score |
|--------:|-----------------:|---------------------------------------------:|:----------------------|------------------------:|---------------------:|----------:|----------:|----------:|------------:|--------------------:|-------------:|
|    0.2  |             0.55 |                                         0.35 | male mar/wid          |                0.808571 |             0.630359 |  0.283427 |  0.284314 |  0.273109 |    0.362288 |            0.283427 |     0.750861 |
|    0.05 |             0.55 |                                         0.5  | male mar/wid          |                0.808571 |             0.621431 |  0.283427 |  0.284314 |  0.308824 |    0.362288 |            0.283427 |     0.746396 |
|    0.08 |             0.55 |                                         0.47 | male mar/wid          |                0.808571 |             0.621431 |  0.283427 |  0.284314 |  0.308824 |    0.362288 |            0.283427 |     0.746396 |
|    0.1  |             0.55 |                                         0.45 | male mar/wid          |                0.808571 |             0.621431 |  0.283427 |  0.284314 |  0.308824 |    0.362288 |            0.283427 |     0.746396 |
|    0.15 |             0.55 |                                         0.4  | male mar/wid          |                0.808571 |             0.621431 |  0.283427 |  0.284314 |  0.308824 |    0.362288 |            0.283427 |     0.746396 |
|    0    |             0.55 |                                         0.55 | male mar/wid          |                0.808571 |             0.539931 |  0.361111 |  0.333333 |  0.333333 |    0.1875   |            0.361111 |     0.705646 |
|    0.02 |             0.55 |                                         0.53 | male mar/wid          |                0.808571 |             0.539931 |  0.361111 |  0.333333 |  0.333333 |    0.1875   |            0.361111 |     0.705646 |

## Output Files
- Threshold search CSV: mitigation_threshold_search.csv
- Baseline group table CSV: mitigation_group_table_before.csv
- Mitigated group table CSV: mitigation_group_table_after.csv

# Mitigation Experiment Report

This report evaluates group-aware threshold mitigation using the final paper-based SAFE score.

## Method
The experiment first identifies the group with the lowest baseline positive prediction rate. It then evaluates several threshold reductions for that group while keeping the other group thresholds unchanged. The selected mitigation is the candidate with the highest paper-based SAFE score.

## Selected Mitigation
- Disadvantaged group: male mar/wid
- Base threshold from configuration: 0.5500
- Selected adjusted threshold: 0.4000
- Selected delta: 0.1500
- Baseline AUC: 0.8048
- Mitigated AUC: 0.8048
- AURGA: 0.7095
- RGR Aggregate: 0.9281
- AURGE: 0.9712
- Baseline fairness aggregate: 0.5399
- Mitigated fairness aggregate: 0.6650
- Baseline paper-based SAFE score: 0.7872
- Mitigated paper-based SAFE score: 0.8185

## Baseline Fairness Components
- SPD gap: 0.3611
- EOD gap: 0.3333
- AOD gap: 0.3333
- DIR ratio: 0.1875

## Mitigated Fairness Components
- SPD gap: 0.2580
- EOD gap: 0.2549
- AOD gap: 0.2465
- DIR ratio: 0.4195

## Baseline Group Table

| group              |   n |   positive_rate |      tpr |       fpr |
|:-------------------|----:|----------------:|---------:|----------:|
| female div/dep/mar |  61 |       0.278689  | 0.6      | 0.121951  |
| male div/sep       |   9 |       0.444444  | 0.666667 | 0.333333  |
| male mar/wid       |  12 |       0.0833333 | 0.333333 | 0         |
| male single        | 118 |       0.186441  | 0.411765 | 0.0952381 |

## Mitigated Group Table

| group              |   n |   positive_rate |      tpr |       fpr |
|:-------------------|----:|----------------:|---------:|----------:|
| female div/dep/mar |  61 |        0.278689 | 0.6      | 0.121951  |
| male div/sep       |   9 |        0.444444 | 0.666667 | 0.333333  |
| male mar/wid       |  12 |        0.416667 | 0.666667 | 0.333333  |
| male single        | 118 |        0.186441 | 0.411765 | 0.0952381 |

## Threshold Search Results

|   delta |   base_threshold |   adjusted_threshold_for_disadvantaged_group | disadvantaged_group   | ranking_metrics_status            |   fairness_aggregate |   spd_gap |   eod_gap |   aod_gap |   dir_ratio |   positive_rate_gap |   safe_score |
|--------:|-----------------:|---------------------------------------------:|:----------------------|:----------------------------------|---------------------:|----------:|----------:|----------:|------------:|--------------------:|-------------:|
|    0.15 |             0.55 |                                         0.4  | male mar/wid          | unchanged_by_threshold_mitigation |             0.665022 |  0.258004 |  0.254902 |  0.246499 |    0.419492 |            0.258004 |     0.818451 |
|    0.2  |             0.55 |                                         0.35 | male mar/wid          | unchanged_by_threshold_mitigation |             0.665022 |  0.258004 |  0.254902 |  0.246499 |    0.419492 |            0.258004 |     0.818451 |
|    0.02 |             0.55 |                                         0.53 | male mar/wid          | unchanged_by_threshold_mitigation |             0.637051 |  0.277778 |  0.254902 |  0.294118 |    0.375    |            0.277778 |     0.811458 |
|    0.05 |             0.55 |                                         0.5  | male mar/wid          | unchanged_by_threshold_mitigation |             0.637051 |  0.277778 |  0.254902 |  0.294118 |    0.375    |            0.277778 |     0.811458 |
|    0.08 |             0.55 |                                         0.47 | male mar/wid          | unchanged_by_threshold_mitigation |             0.637051 |  0.277778 |  0.254902 |  0.294118 |    0.375    |            0.277778 |     0.811458 |
|    0.1  |             0.55 |                                         0.45 | male mar/wid          | unchanged_by_threshold_mitigation |             0.637051 |  0.277778 |  0.254902 |  0.294118 |    0.375    |            0.277778 |     0.811458 |
|    0    |             0.55 |                                         0.55 | male mar/wid          | unchanged_by_threshold_mitigation |             0.539931 |  0.361111 |  0.333333 |  0.333333 |    0.1875   |            0.361111 |     0.787178 |

## Output Files
- Threshold search CSV: mitigation_threshold_search.csv
- Baseline group table CSV: mitigation_group_table_before.csv
- Mitigated group table CSV: mitigation_group_table_after.csv

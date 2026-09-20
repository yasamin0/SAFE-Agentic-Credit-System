
# SAFE Weight Sensitivity Analysis

The SAFE score was recalculated under five policy-weighting scenarios
without retraining the predictive models.

The purpose of this analysis is to evaluate whether the composite SAFE
score is highly dependent on the original equal-weight assumption.

## Weighting Scenarios

| Scenario | RGA | RGR | RGE | Fairness |
|---|---:|---:|---:|---:|
| Equal weights | 0.25 | 0.25 | 0.25 | 0.25 |
| RGA priority | 0.40 | 0.20 | 0.20 | 0.20 |
| RGR priority | 0.20 | 0.40 | 0.20 | 0.20 |
| RGE priority | 0.20 | 0.20 | 0.40 | 0.20 |
| Fairness priority | 0.20 | 0.20 | 0.20 | 0.40 |

## Summary Results

| dataset       | scenario          |   mean_safe |   std_safe |   min_safe |   max_safe |   delta_vs_equal |   approved_folds |   rejected_folds |
|:--------------|:------------------|------------:|-----------:|-----------:|-----------:|-----------------:|-----------------:|-----------------:|
| German Credit | Equal weights     |    0.810817 | 0.0353824  |   0.752899 |   0.844283 |        0         |                5 |                0 |
| German Credit | RGA priority      |    0.791647 | 0.0336953  |   0.735923 |   0.820858 |       -0.0191694 |                4 |                1 |
| German Credit | RGR priority      |    0.837895 | 0.0283151  |   0.79205  |   0.864598 |        0.0270784 |                5 |                0 |
| German Credit | RGE priority      |    0.842313 | 0.0279407  |   0.796579 |   0.869227 |        0.0314957 |                5 |                0 |
| German Credit | Fairness priority |    0.771412 | 0.0520933  |   0.687043 |   0.822449 |       -0.0394047 |                4 |                1 |
| Taiwan Credit | Equal weights     |    0.864588 | 0.00661267 |   0.857449 |   0.87306  |        0         |                5 |                0 |
| Taiwan Credit | RGA priority      |    0.816107 | 0.00869558 |   0.807481 |   0.826888 |       -0.0484807 |                5 |                0 |
| Taiwan Credit | RGR priority      |    0.879104 | 0.00534222 |   0.873537 |   0.886087 |        0.0145167 |                5 |                0 |
| Taiwan Credit | RGE priority      |    0.881689 | 0.00566988 |   0.875114 |   0.88881  |        0.0171015 |                5 |                0 |
| Taiwan Credit | Fairness priority |    0.88145  | 0.00684515 |   0.873662 |   0.890454 |        0.0168625 |                5 |                0 |

## Stability Across Weighting Scenarios

| dataset       | highest_scenario   |   highest_mean_safe | lowest_scenario   |   lowest_mean_safe |   range_across_weights |
|:--------------|:-------------------|--------------------:|:------------------|-------------------:|-----------------------:|
| German Credit | RGE priority       |            0.842313 | Fairness priority |           0.771412 |              0.0709005 |
| Taiwan Credit | RGE priority       |            0.881689 | RGA priority      |           0.816107 |              0.0655823 |

The `delta_vs_equal` column reports the change in mean SAFE score
relative to the original equal-weight specification.

The range between the highest and lowest mean SAFE scores provides a
simple measure of sensitivity to alternative governance priorities.

Approval decisions were evaluated using the same SAFE approval
threshold used in the main framework.

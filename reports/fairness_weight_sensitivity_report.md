
# Fairness Aggregate Weight Sensitivity

The Fairness Aggregate was recalculated under five alternative
weighting policies for SPD, EOD, AOD, and DIR.

No predictive model was retrained. The analysis isolates the effect
of the fairness aggregation policy.

## Summary

| dataset       | scenario      |   mean_fairness |   std_fairness |   min_fairness |   max_fairness |   delta_vs_equal |
|:--------------|:--------------|----------------:|---------------:|---------------:|---------------:|-----------------:|
| German Credit | AOD priority  |        0.691638 |     0.0949747  |       0.548018 |       0.78374  |       0.0105364  |
| German Credit | DIR priority  |        0.643474 |     0.0949327  |       0.510571 |       0.741511 |      -0.0376277  |
| German Credit | EOD priority  |        0.676024 |     0.101941   |       0.521431 |       0.783537 |      -0.00507743 |
| German Credit | Equal weights |        0.681101 |     0.0926688  |       0.544646 |       0.771087 |       0          |
| German Credit | SPD priority  |        0.71327  |     0.0801319  |       0.598564 |       0.792234 |       0.0321687  |
| Taiwan Credit | AOD priority  |        0.956535 |     0.00669032 |       0.948005 |       0.964725 |       0.00763506 |
| Taiwan Credit | DIR priority  |        0.927831 |     0.0135434  |       0.911918 |       0.946786 |      -0.0210688  |
| Taiwan Credit | EOD priority  |        0.955851 |     0.00640539 |       0.948082 |       0.963119 |       0.00695044 |
| Taiwan Credit | Equal weights |        0.9489   |     0.00849355 |       0.938516 |       0.960032 |       0          |
| Taiwan Credit | SPD priority  |        0.955384 |     0.00764446 |       0.94606  |       0.965499 |       0.00648335 |

## Stability

| dataset       | highest_scenario   |   highest_mean_fairness | lowest_scenario   |   lowest_mean_fairness |   range_across_weights |
|:--------------|:-------------------|------------------------:|:------------------|-----------------------:|-----------------------:|
| German Credit | SPD priority       |                0.71327  | DIR priority      |               0.643474 |              0.0697964 |
| Taiwan Credit | AOD priority       |                0.956535 | DIR priority      |               0.927831 |              0.0287039 |

The `delta_vs_equal` column measures the change relative to the
equal-weight fairness specification.

The range across weighting scenarios quantifies the sensitivity of
the Fairness Aggregate to alternative fairness priorities.

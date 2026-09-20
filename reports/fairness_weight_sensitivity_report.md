
# Fairness Aggregate Weight Sensitivity

The Fairness Aggregate was recalculated under five alternative
weighting policies for SPD, EOD, AOD, and DIR.

No predictive model was retrained. The analysis isolates the effect
of the fairness aggregation policy.

## Summary

| dataset       | scenario      |   mean_fairness |   std_fairness |   min_fairness |   max_fairness |   delta_vs_equal |
|:--------------|:--------------|----------------:|---------------:|---------------:|---------------:|-----------------:|
| German Credit | AOD priority  |        0.63429  |     0.109912   |       0.460291 |       0.746179 |      0.0204971   |
| German Credit | DIR priority  |        0.557302 |     0.140055   |       0.338895 |       0.700589 |     -0.0564913   |
| German Credit | EOD priority  |        0.614565 |     0.121873   |       0.418895 |       0.734756 |      0.000771807 |
| German Credit | Equal weights |        0.613793 |     0.120037   |       0.423619 |       0.735112 |      0           |
| German Credit | SPD priority  |        0.649016 |     0.109095   |       0.476395 |       0.758923 |      0.0352224   |
| Taiwan Credit | AOD priority  |        0.956535 |     0.00669032 |       0.948005 |       0.964725 |      0.00763506  |
| Taiwan Credit | DIR priority  |        0.927831 |     0.0135434  |       0.911918 |       0.946786 |     -0.0210688   |
| Taiwan Credit | EOD priority  |        0.955851 |     0.00640539 |       0.948082 |       0.963119 |      0.00695044  |
| Taiwan Credit | Equal weights |        0.9489   |     0.00849355 |       0.938516 |       0.960032 |      0           |
| Taiwan Credit | SPD priority  |        0.955384 |     0.00764446 |       0.94606  |       0.965499 |      0.00648335  |

## Stability

| dataset       | highest_scenario   |   highest_mean_fairness | lowest_scenario   |   lowest_mean_fairness |   range_across_weights |
|:--------------|:-------------------|------------------------:|:------------------|-----------------------:|-----------------------:|
| German Credit | SPD priority       |                0.649016 | DIR priority      |               0.557302 |              0.0917137 |
| Taiwan Credit | AOD priority       |                0.956535 | DIR priority      |               0.927831 |              0.0287039 |

The `delta_vs_equal` column measures the change relative to the
equal-weight fairness specification.

The range across weighting scenarios quantifies the sensitivity of
the Fairness Aggregate to alternative fairness priorities.

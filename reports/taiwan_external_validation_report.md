
# Taiwan External Dataset Validation

The SAFE governance framework was independently replicated on the
Taiwan Default of Credit Card Clients dataset.

Dataset size: 30000

Sensitive attribute: SEX

Target: CreditRisk

Validation design:
- Stratified 5-fold cross-validation
- Preprocessing fitted independently within each training fold
- Sensitive attribute excluded from predictive model inputs
- XGBoost used as the governance model family
- SAFE score calculated from RGA, RGR, RGE, and Fairness

## Fold-Level Results

|   fold |   n_train |   n_validation |      auc |   calibration_intercept |   calibration_slope |   brier_score |   cvm_statistic |   cvm_p_value |    aurga |   aurgr_gaussian |   aurgr_swapping |   reference_rgr |   rgr_aggregate |    aurge |   fairness_score_spd |   fairness_score_eod |   fairness_score_aod |   fairness_score_dir |   fairness_aggregate |   safe_score |
|-------:|----------:|---------------:|---------:|------------------------:|--------------------:|--------------:|----------------:|--------------:|---------:|-----------------:|-----------------:|----------------:|----------------:|---------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|-------------:|
|      1 |     24000 |           6000 | 0.793531 |              0.00277113 |            1.00792  |      0.132519 |        108.046  |   2.65666e-08 | 0.637492 |         0.858055 |         0.474483 |        0.936751 |        0.936751 | 0.95155  |             0.981963 |             0.993916 |             0.991067 |             0.846618 |             0.953391 |     0.869796 |
|      2 |     24000 |           6000 | 0.778227 |             -0.0671207  |            0.938865 |      0.135714 |         96.8023 |   4.97982e-08 | 0.620349 |         0.840118 |         0.480921 |        0.936843 |        0.936843 | 0.950585 |             0.978061 |             0.98627  |             0.988864 |             0.818383 |             0.942895 |     0.862668 |
|      3 |     24000 |           6000 | 0.77875  |             -0.0608009  |            0.947315 |      0.135681 |         97.0421 |   5.15088e-08 | 0.607611 |         0.873486 |         0.47304  |        0.937891 |        0.937891 | 0.945777 |             0.976235 |             0.986344 |             0.985961 |             0.805526 |             0.938516 |     0.857449 |
|      4 |     24000 |           6000 | 0.778806 |             -0.00197074 |            0.985733 |      0.132288 |         97.9097 |   1.91918e-08 | 0.603269 |         0.867027 |         0.48071  |        0.936176 |        0.936176 | 0.950753 |             0.98296  |             0.976267 |             0.985989 |             0.853453 |             0.949667 |     0.859966 |
|      5 |     24000 |           6000 | 0.774904 |             -0.0597865  |            0.941926 |      0.135924 |         92.8289 |   2.77605e-08 | 0.642201 |         0.84984  |         0.502007 |        0.938194 |        0.938194 | 0.951812 |             0.987367 |             0.975465 |             0.983497 |             0.8938   |             0.960032 |     0.87306  |

## Five-Fold Summary

## Cramer-von Mises Distributional Test

The two-sample Cramer-von Mises test was used within each validation
fold to compare the distribution of predicted default probabilities
between observations with CreditRisk = 0 and CreditRisk = 1.

Null hypothesis:
the two predicted-probability distributions are identical.

Significant folds at alpha = 0.05:
5 out of 5.

Minimum fold p-value:
1.91918e-08

Maximum fold p-value:
5.15088e-08

Fold-level p-values are reported individually and are not averaged.

| metric                |       mean |         std |   ci_lower |     ci_upper |   n_folds | mean_sd          | mean_95ci                   |
|:----------------------|-----------:|------------:|-----------:|-------------:|----------:|:-----------------|:----------------------------|
| auc                   |  0.780844  | 0.00727386  |  0.771812  |   0.789875   |         5 | 0.7808 (0.0073)  | 0.7808 [0.7718, 0.7899]     |
| calibration_intercept | -0.0373816 | 0.0346447   | -0.0803987 |   0.00563557 |         5 | -0.0374 (0.0346) | -0.0374 [-0.0804, 0.0056]   |
| calibration_slope     |  0.964352  | 0.0308148   |  0.92609   |   1.00261    |         5 | 0.9644 (0.0308)  | 0.9644 [0.9261, 1.0026]     |
| brier_score           |  0.134425  | 0.00184997  |  0.132128  |   0.136722   |         5 | 0.1344 (0.0018)  | 0.1344 [0.1321, 0.1367]     |
| cvm_statistic         | 98.5258    | 5.67114     | 91.4842    | 105.567      |         5 | 98.5258 (5.6711) | 98.5258 [91.4842, 105.5675] |
| aurga                 |  0.622184  | 0.0173818   |  0.600602  |   0.643767   |         5 | 0.6222 (0.0174)  | 0.6222 [0.6006, 0.6438]     |
| rgr_aggregate         |  0.937171  | 0.000842463 |  0.936125  |   0.938217   |         5 | 0.9372 (0.0008)  | 0.9372 [0.9361, 0.9382]     |
| reference_rgr         |  0.937171  | 0.000842463 |  0.936125  |   0.938217   |         5 | 0.9372 (0.0008)  | 0.9372 [0.9361, 0.9382]     |
| aurge                 |  0.950095  | 0.00246926  |  0.947029  |   0.953161   |         5 | 0.9501 (0.0025)  | 0.9501 [0.9470, 0.9532]     |
| fairness_aggregate    |  0.9489    | 0.00849355  |  0.938354  |   0.959446   |         5 | 0.9489 (0.0085)  | 0.9489 [0.9384, 0.9594]     |
| safe_score            |  0.864588  | 0.00661267  |  0.856377  |   0.872798   |         5 | 0.8646 (0.0066)  | 0.8646 [0.8564, 0.8728]     |

Values in `mean_sd` are reported as mean (standard deviation).

The `mean_95ci` column reports fold-level 95% confidence intervals.

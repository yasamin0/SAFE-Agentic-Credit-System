# SAFE Model Selection Report

This report compares the top candidate models using core SAFE governance metrics. The top candidates are first selected by cross-validation AUC, then compared using AURGA, RGR Aggregate, AURGE, Fairness Aggregate, and paper-based SAFE score.

## Selection Rule

The selected operational governance model is the candidate with the highest paper-based SAFE score among the top CV-AUC candidates.

## Selected Model

- Selected model: Voting Ensemble
- Selected baseline SAFE score: 0.7872

## SAFE Model Selection Table

| model             |   cv_auc |   test_auc |    aurga |   rgr_aggregate |    aurge |   fairness_aggregate |   paper_safe_score |   baseline_safe_score | decision   |
|:------------------|---------:|-----------:|---------:|----------------:|---------:|---------------------:|-------------------:|----------------------:|:-----------|
| Voting Ensemble   | 0.782489 |   0.804762 | 0.709529 |        0.928066 | 0.971187 |             0.539931 |           0.787178 |              0.787178 | APPROVED   |
| Stacking Ensemble | 0.779621 |   0.808571 | 0.701846 |        0.931696 | 0.972548 |             0.539931 |           0.786505 |              0.786505 | APPROVED   |
| XGBoost           | 0.78137  |   0.776667 | 0.690258 |        0.914455 | 0.981596 |             0.536254 |           0.780641 |              0.780641 | APPROVED   |
| Random Forest     | 0.788969 |   0.805833 | 0.7211   |        0.959478 | 0.982013 |             0.388889 |           0.76287  |              0.76287  | APPROVED   |

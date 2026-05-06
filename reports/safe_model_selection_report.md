# SAFE Model Selection Report

This report compares the top candidate models using core SAFE governance metrics. The top candidates are first selected by cross-validation AUC, then compared using test AUC, fairness aggregate, robustness aggregate, and baseline SAFE score.

## Selection Rule

The selected operational governance model is the candidate with the highest baseline SAFE score among the top CV-AUC candidates.

## Selected Model

- Selected model: Stacking Ensemble
- Selected baseline SAFE score: 0.7056

## SAFE Model Selection Table

| model             |   cv_auc |   test_auc |   fairness_aggregate |   robustness_aggregate |   baseline_safe_score | decision   |
|:------------------|---------:|-----------:|---------------------:|-----------------------:|----------------------:|:-----------|
| Stacking Ensemble | 0.779621 |   0.808571 |             0.539931 |               0.965548 |              0.705646 | REJECTED   |
| Voting Ensemble   | 0.782489 |   0.804762 |             0.539931 |               0.965483 |              0.70449  | REJECTED   |
| XGBoost           | 0.78137  |   0.776667 |             0.536254 |               0.956571 |              0.692441 | REJECTED   |
| Random Forest     | 0.788969 |   0.805833 |             0.388889 |               0.958438 |              0.627882 | REJECTED   |

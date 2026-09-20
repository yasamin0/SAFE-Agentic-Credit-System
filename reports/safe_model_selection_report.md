# SAFE Model Selection Report

This report compares the top candidate models using core SAFE governance metrics. The top candidates are first selected by cross-validation AUC, then compared using AURGA, RGR Aggregate, AURGE, Fairness Aggregate, and paper-based SAFE score.

## Selection Rule

The selected operational governance model is the candidate with the highest paper-based SAFE score among the top CV-AUC candidates.

## Selected Model

- Selected model: Voting Ensemble
- Selected baseline SAFE score: 0.7951

## SAFE Model Selection Table

| model             |   cv_auc |   test_auc |    aurga |   rgr_aggregate |    aurge |   fairness_aggregate |   paper_safe_score |   baseline_safe_score | decision   |
|:------------------|---------:|-----------:|---------:|----------------:|---------:|---------------------:|-------------------:|----------------------:|:-----------|
| Voting Ensemble   | 0.796763 |   0.804762 | 0.709529 |        0.959555 | 0.971187 |             0.539931 |           0.79505  |              0.79505  | APPROVED   |
| XGBoost           | 0.802493 |   0.7825   | 0.704854 |        0.962115 | 0.971855 |             0.539931 |           0.794689 |              0.794689 | APPROVED   |
| Stacking Ensemble | 0.791034 |   0.808571 | 0.701846 |        0.961944 | 0.972548 |             0.539931 |           0.794067 |              0.794067 | APPROVED   |
| Random Forest     | 0.80253  |   0.79119  | 0.68297  |        0.962168 | 0.970118 |             0.435764 |           0.762755 |              0.762755 | APPROVED   |

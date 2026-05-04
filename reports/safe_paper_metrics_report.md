# SAFE AI Paper Metrics Report

This report summarizes the implemented SAFE AI paper metrics across multiple models.

## Metrics Implemented
- AURGA for rank-based accuracy
- AURGR for rank-based robustness
- AURGE for rank-based explainability
- Compliance Score using Arithmetic Mean, Geometric Mean, RMS, and TOPSIS

## Current Governance Model Metrics
- AURGA: 0.7012
- AURGR Gaussian: 0.9814
- AURGR Swapping: 0.8475
- AURGE: 0.9816

## SHAP vs RGE
- SHAP comparison status: completed
- Spearman correlation: 0.983841540319698

## Model Metrics Comparison

| Model               |    AURGA |   AURGR_Gaussian |   AURGR_Swapping |    AURGR |    AURGE |
|:--------------------|---------:|-----------------:|-----------------:|---------:|---------:|
| Logistic Regression | 0.656765 |         0.996301 |         0.846401 | 0.921351 | 0.974049 |
| Random Forest       | 0.730336 |         0.99553  |         0.923426 | 0.959478 | 0.982013 |
| XGBoost             | 0.701229 |         0.981441 |         0.847468 | 0.914455 | 0.981596 |
| Voting Ensemble     | 0.720947 |         0.99164  |         0.864491 | 0.928066 | 0.971187 |
| Stacking Ensemble   | 0.713829 |         0.994305 |         0.869086 | 0.931696 | 0.972548 |
| Random Baseline     | 0.5      |         1        |         1        | 1        | 1        |

## Compliance Score Comparison

| Model               |    AURGA |   AURGR_Gaussian |   AURGR_Swapping |    AURGR |    AURGE |   Compliance_Arithmetic |   Compliance_Geometric |   Compliance_RMS |   Compliance_TOPSIS |
|:--------------------|---------:|-----------------:|-----------------:|---------:|---------:|------------------------:|-----------------------:|-----------------:|--------------------:|
| Random Forest       | 0.730336 |         0.99553  |         0.923426 | 0.959478 | 0.982013 |                0.890609 |               0.88286  |         0.897838 |            0.880599 |
| Voting Ensemble     | 0.720947 |         0.99164  |         0.864491 | 0.928066 | 0.971187 |                0.8734   |               0.866154 |         0.880203 |            0.797964 |
| Stacking Ensemble   | 0.713829 |         0.994305 |         0.869086 | 0.931696 | 0.972548 |                0.872691 |               0.864821 |         0.880049 |            0.795747 |
| XGBoost             | 0.701229 |         0.981441 |         0.847468 | 0.914455 | 0.981596 |                0.86576  |               0.857008 |         0.873972 |            0.744791 |
| Logistic Regression | 0.656765 |         0.996301 |         0.846401 | 0.921351 | 0.974049 |                0.850722 |               0.83844  |         0.861974 |            0.624381 |
| Random Baseline     | 0.5      |         1        |         1        | 1        | 1        |                0.833333 |               0.793701 |         0.866025 |            0.218483 |

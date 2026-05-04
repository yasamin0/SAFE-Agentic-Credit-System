# SAFE AI Paper Metrics Report

This report summarizes the implemented SAFE AI paper metrics across multiple models.

## Metrics Implemented
- AURGA for rank-based accuracy
- AURGR for rank-based robustness
- AURGE for rank-based explainability
- Compliance Score using Arithmetic Mean, Geometric Mean, RMS, and TOPSIS

## Current Governance Model Metrics
- AURGA: 0.7252
- AURGR Gaussian: 0.9664
- AURGR Swapping: 0.8162
- AURGE: 0.9658

## SHAP vs RGE
- SHAP comparison status: completed
- Spearman correlation: 0.9359205941635693

## Model Metrics Comparison

| Model               |    AURGA |   AURGR_Gaussian |   AURGR_Swapping |    AURGR |    AURGE |
|:--------------------|---------:|-----------------:|-----------------:|---------:|---------:|
| Logistic Regression | 0.64564  |         0.996214 |         0.86436  | 0.930287 | 0.96933  |
| Random Forest       | 0.727355 |         0.990898 |         0.879695 | 0.935296 | 0.96643  |
| XGBoost             | 0.725155 |         0.966399 |         0.816184 | 0.891291 | 0.965834 |
| Voting Ensemble     | 0.720947 |         0.99164  |         0.864491 | 0.928066 | 0.971187 |
| Stacking Ensemble   | 0.713829 |         0.994305 |         0.869086 | 0.931696 | 0.972548 |
| Random Baseline     | 0.5      |         1        |         1        | 1        | 1        |

## Compliance Score Comparison

| Model               |    AURGA |   AURGR_Gaussian |   AURGR_Swapping |    AURGR |    AURGE |   Compliance_Arithmetic |   Compliance_Geometric |   Compliance_RMS |   Compliance_TOPSIS |
|:--------------------|---------:|-----------------:|-----------------:|---------:|---------:|------------------------:|-----------------------:|-----------------:|--------------------:|
| Random Forest       | 0.727355 |         0.990898 |         0.879695 | 0.935296 | 0.96643  |                0.876361 |               0.869538 |         0.882763 |            0.814536 |
| Voting Ensemble     | 0.720947 |         0.99164  |         0.864491 | 0.928066 | 0.971187 |                0.8734   |               0.866154 |         0.880203 |            0.798708 |
| Stacking Ensemble   | 0.713829 |         0.994305 |         0.869086 | 0.931696 | 0.972548 |                0.872691 |               0.864821 |         0.880049 |            0.798014 |
| XGBoost             | 0.725155 |         0.966399 |         0.816184 | 0.891291 | 0.965834 |                0.86076  |               0.854642 |         0.866619 |            0.732732 |
| Logistic Regression | 0.64564  |         0.996214 |         0.86436  | 0.930287 | 0.96933  |                0.848419 |               0.835013 |         0.860598 |            0.601374 |
| Random Baseline     | 0.5      |         1        |         1        | 1        | 1        |                0.833333 |               0.793701 |         0.866025 |            0.265298 |

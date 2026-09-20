# SAFE AI Paper Metrics Report

This report summarizes the implemented SAFE AI paper metrics across multiple models.

## Metrics Implemented
- AURGA for rank-based accuracy
- AURGR for rank-based robustness
- AURGE for rank-based explainability
- Compliance Score using Arithmetic Mean, Geometric Mean, RMS, and TOPSIS

## Current Governance Model Metrics
- AURGA: 0.7095
- AURGR Gaussian: 0.9916
- AURGR Swapping: 0.8645
- AURGE: 0.9712

## SHAP vs RGE
- SHAP comparison status: completed
- Spearman correlation: 0.9025383395029084

## Model Metrics Comparison

| Model               |    AURGA |   AURGR_Gaussian |   AURGR_Swapping |   RGR_Reference |    AURGR |    AURGE |
|:--------------------|---------:|-----------------:|-----------------:|----------------:|---------:|---------:|
| Logistic Regression | 0.640236 |         0.996301 |         0.846401 |        0.961271 | 0.961271 | 0.974049 |
| Random Forest       | 0.68297  |         0.995074 |         0.907102 |        0.962168 | 0.962168 | 0.970118 |
| XGBoost             | 0.704854 |         0.978364 |         0.835958 |        0.962115 | 0.962115 | 0.971855 |
| Voting Ensemble     | 0.709529 |         0.99164  |         0.864491 |        0.959555 | 0.959555 | 0.971187 |
| Stacking Ensemble   | 0.701846 |         0.994305 |         0.869086 |        0.961944 | 0.961944 | 0.972548 |
| Random Baseline     | 0.5      |         1        |         1        |        1        | 1        | 1        |

## Compliance Score Comparison

| Model               |    AURGA |   AURGR_Gaussian |   AURGR_Swapping |   RGR_Reference |    AURGR |    AURGE |   AURGR_for_compliance |   AURGE_for_compliance |   Compliance_Arithmetic |   Compliance_Geometric |   Compliance_RMS |   Compliance_TOPSIS |
|:--------------------|---------:|-----------------:|-----------------:|----------------:|---------:|---------:|-----------------------:|-----------------------:|------------------------:|-----------------------:|-----------------:|--------------------:|
| Voting Ensemble     | 0.709529 |         0.99164  |         0.864491 |        0.959555 | 0.959555 | 0.971187 |               0.959555 |               0.971187 |                0.88009  |               0.871193 |         0.888328 |            0.994633 |
| XGBoost             | 0.704854 |         0.978364 |         0.835958 |        0.962115 | 0.962115 | 0.971855 |               0.962115 |               0.971855 |                0.879608 |               0.870247 |         0.888254 |            0.990654 |
| Stacking Ensemble   | 0.701846 |         0.994305 |         0.869086 |        0.961944 | 0.961944 | 0.972548 |               0.961944 |               0.972548 |                0.878779 |               0.869162 |         0.887651 |            0.985364 |
| Random Forest       | 0.68297  |         0.995074 |         0.907102 |        0.962168 | 0.962168 | 0.970118 |               0.962168 |               0.970118 |                0.871752 |               0.860649 |         0.88192  |            0.950645 |
| Logistic Regression | 0.640236 |         0.996301 |         0.846401 |        0.961271 | 0.961271 | 0.974049 |               0.961271 |               0.974049 |                0.858519 |               0.843184 |         0.872299 |            0.878789 |
| Random Baseline     | 0.5      |         1        |         1        |        1        | 1        | 1        |               0.5      |               0.5      |                0.5      |               0.5      |         0.5      |            0        |

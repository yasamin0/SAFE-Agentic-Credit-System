# SAFE AI Paper Metrics Report

This report summarizes the implemented SAFE AI paper metrics across multiple models.

## Metrics Implemented
- AURGA for rank-based accuracy
- AURGR for rank-based robustness
- AURGE for rank-based explainability
- Compliance Score using Arithmetic Mean, Geometric Mean, RMS, and TOPSIS

## Current Governance Model Metrics
- AURGA: 0.7018
- AURGR Gaussian: 0.9943
- AURGR Swapping: 0.8691
- AURGE: 0.9725

## SHAP vs RGE
- SHAP comparison status: completed
- Spearman correlation: 0.8902696985721839

## Model Metrics Comparison

| Model               |    AURGA |   AURGR_Gaussian |   AURGR_Swapping |    AURGR |    AURGE |
|:--------------------|---------:|-----------------:|-----------------:|---------:|---------:|
| Logistic Regression | 0.640236 |         0.996301 |         0.846401 | 0.921351 | 0.974049 |
| Random Forest       | 0.7211   |         0.99553  |         0.923426 | 0.959478 | 0.982013 |
| XGBoost             | 0.690258 |         0.981441 |         0.847468 | 0.914455 | 0.981596 |
| Voting Ensemble     | 0.709529 |         0.99164  |         0.864491 | 0.928066 | 0.971187 |
| Stacking Ensemble   | 0.701846 |         0.994305 |         0.869086 | 0.931696 | 0.972548 |
| Random Baseline     | 0.5      |         1        |         1        | 1        | 1        |

## Compliance Score Comparison

| Model               |    AURGA |   AURGR_Gaussian |   AURGR_Swapping |    AURGR |    AURGE |   AURGR_for_compliance |   AURGE_for_compliance |   Compliance_Arithmetic |   Compliance_Geometric |   Compliance_RMS |   Compliance_TOPSIS |
|:--------------------|---------:|-----------------:|-----------------:|---------:|---------:|-----------------------:|-----------------------:|------------------------:|-----------------------:|-----------------:|--------------------:|
| Random Forest       | 0.7211   |         0.99553  |         0.923426 | 0.959478 | 0.982013 |               0.959478 |               0.982013 |                0.88753  |               0.879122 |         0.895346 |            1        |
| Voting Ensemble     | 0.709529 |         0.99164  |         0.864491 | 0.928066 | 0.971187 |               0.928066 |               0.971187 |                0.869594 |               0.861557 |         0.877105 |            0.949168 |
| Stacking Ensemble   | 0.701846 |         0.994305 |         0.869086 | 0.931696 | 0.972548 |               0.931696 |               0.972548 |                0.868696 |               0.859955 |         0.87683  |            0.946124 |
| XGBoost             | 0.690258 |         0.981441 |         0.847468 | 0.914455 | 0.981596 |               0.914455 |               0.981596 |                0.862103 |               0.852515 |         0.871056 |            0.916847 |
| Logistic Regression | 0.640236 |         0.996301 |         0.846401 | 0.921351 | 0.974049 |               0.921351 |               0.974049 |                0.845212 |               0.831346 |         0.857819 |            0.850837 |
| Random Baseline     | 0.5      |         1        |         1        | 1        | 1        |               0.5      |               0.5      |                0.5      |               0.5      |         0.5      |            0        |

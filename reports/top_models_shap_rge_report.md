# Top Models SHAP-RGE Comparison Report

This report compares RGE feature ranking with SHAP feature ranking for the top selected candidate models.

## Summary

| model             | status    | shap_method      |   sample_size |   rge_shap_spearman | error   |
|:------------------|:----------|:-----------------|--------------:|--------------------:|:--------|
| Stacking Ensemble | completed | kernel_explainer |           100 |            0.899577 |         |
| Voting Ensemble   | completed | kernel_explainer |           100 |            0.897462 |         |
| XGBoost           | completed | xgboost_treeshap |           100 |            0.982526 |         |
| Random Forest     | completed | tree_explainer   |           100 |            0.914754 |         |

## Interpretation

A higher Spearman correlation means that the RGE-based ranking and the SHAP-based ranking agree more strongly. XGBoost uses built-in TreeSHAP, Random Forest uses SHAP TreeExplainer, and meta-models such as Voting or Stacking use model-agnostic KernelExplainer.

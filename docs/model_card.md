# Model Card

## Training Summary

Multiple model candidates were trained and compared using 5-fold stratified cross-validation.

| model               |   best_cv_auc | best_params                                                                                                                    | artifact                      |
|:--------------------|--------------:|:-------------------------------------------------------------------------------------------------------------------------------|:------------------------------|
| Random Forest       |      0.80253  | {'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 100}               | model_random_forest.pkl       |
| XGBoost             |      0.802493 | {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 0.8} | model_xgboost.pkl             |
| Voting Ensemble     |      0.796763 | default                                                                                                                        | model_voting_ensemble.pkl     |
| Stacking Ensemble   |      0.791034 | default                                                                                                                        | model_stacking_ensemble.pkl   |
| Logistic Regression |      0.781399 | {'C': 0.1, 'solver': 'lbfgs'}                                                                                                  | model_logistic_regression.pkl |
| Random Baseline     |      0.5      | default                                                                                                                        | model_random_baseline.pkl     |

## Operational Governance Model

The operational model saved as `best_model.pkl` is XGBoost for compatibility with the existing SAFE pipeline.

Detailed CV results are saved to `cv_results.csv`.

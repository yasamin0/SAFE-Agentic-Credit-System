# Model Comparison Report

| model               |   best_cv_auc | best_params                                                 | artifact                      |
|:--------------------|--------------:|:------------------------------------------------------------|:------------------------------|
| Random Forest       |      0.788969 | {'max_depth': 5, 'n_estimators': 200}                       | model_random_forest.pkl       |
| Voting Ensemble     |      0.782489 | default                                                     | model_voting_ensemble.pkl     |
| XGBoost             |      0.78137  | {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 80} | model_xgboost.pkl             |
| Stacking Ensemble   |      0.779621 | default                                                     | model_stacking_ensemble.pkl   |
| Logistic Regression |      0.77489  | {'C': 0.1}                                                  | model_logistic_regression.pkl |
| Random Baseline     |      0.5      | default                                                     | model_random_baseline.pkl     |

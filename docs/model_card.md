## Multi-Model Card

The following models were trained for SAFE AI paper-style comparison:

| model               | artifact                      |   features | status   |
|:--------------------|:------------------------------|-----------:|:---------|
| Logistic Regression | model_logistic_regression.pkl |         61 | trained  |
| Random Forest       | model_random_forest.pkl       |         61 | trained  |
| XGBoost             | model_xgboost.pkl             |         61 | trained  |
| Voting Ensemble     | model_voting_ensemble.pkl     |         61 | trained  |
| Stacking Ensemble   | model_stacking_ensemble.pkl   |         61 | trained  |
| Random Baseline     | model_random_baseline.pkl     |         61 | trained  |

The main governance model saved as best_model.pkl is XGBoost.

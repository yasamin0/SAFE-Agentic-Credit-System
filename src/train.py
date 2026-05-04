# src/train.py

import joblib
import numpy as np
import pandas as pd

from crewai.tools import tool

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score

from src.model import build_model_candidates

from src.paths import (
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
    MODEL_PATH,
    MODEL_CARD_PATH,
    ALL_MODEL_PATHS,
    CV_RESULTS_PATH,
    MODEL_COMPARISON_REPORT_PATH,
)


def _small_param_grid(model_name):
    """
    Return a small hyperparameter grid for fast model selection.
    Large grids are avoided so the pipeline stays practical.
    """
    if model_name == "Logistic Regression":
        return {
            "C": [0.1, 1.0, 10.0],
        }

    if model_name == "Random Forest":
        return {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
        }

    if model_name == "XGBoost":
        return {
            "n_estimators": [80, 120],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        }

    return None


@tool
def model_training_tool(description: str):
    """
    Train multiple models with cross-validation and small hyperparameter search.

    The main governance model remains XGBoost and is saved as best_model.pkl.
    """
    try:
        X_train = pd.read_csv(TRAIN_FEATURES_PATH)
        y_train = pd.read_csv(TRAIN_TARGET_PATH).values.ravel()

        models = build_model_candidates()

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        summary_rows = []
        cv_rows = {}
        trained_models = {}

        for model_name, model in models.items():
            param_grid = _small_param_grid(model_name)

            if param_grid:
                search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring="roc_auc",
                    cv=cv,
                    n_jobs=-1,
                    refit=True
                )
                search.fit(X_train, y_train)

                best_model = search.best_estimator_
                best_cv_auc = float(search.best_score_)

                temp_df = pd.DataFrame(search.cv_results_)
                temp_df["model"] = model_name
                cv_rows[model_name] = temp_df

                best_params = search.best_params_

            else:
                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    scoring="roc_auc",
                    cv=cv,
                    n_jobs=-1
                )

                model.fit(X_train, y_train)
                best_model = model
                best_cv_auc = float(np.mean(scores))
                best_params = "default"

                cv_rows[model_name] = pd.DataFrame({
                    "model": [model_name],
                    "mean_test_score": [best_cv_auc],
                    "params": [best_params],
                })

            trained_models[model_name] = best_model

            model_path = ALL_MODEL_PATHS[model_name]
            joblib.dump(best_model, model_path)

            summary_rows.append({
                "model": model_name,
                "best_cv_auc": best_cv_auc,
                "best_params": str(best_params),
                "artifact": model_path.name,
            })

        # Keep XGBoost as the operational governance model.
        joblib.dump(trained_models["XGBoost"], MODEL_PATH)

        summary_df = pd.DataFrame(summary_rows).sort_values("best_cv_auc", ascending=False)
        cv_results_df = pd.concat(cv_rows.values(), ignore_index=True, sort=False)

        cv_results_df.to_csv(CV_RESULTS_PATH, index=False)

        with open(MODEL_CARD_PATH, "w", encoding="utf-8") as f:
            f.write("# Model Card\n\n")
            f.write("## Training Summary\n\n")
            f.write("Multiple model candidates were trained and compared using 3-fold stratified cross-validation.\n\n")
            f.write(summary_df.to_markdown(index=False))
            f.write("\n\n")
            f.write("## Operational Governance Model\n\n")
            f.write("The operational model saved as `best_model.pkl` is XGBoost for compatibility with the existing SAFE pipeline.\n\n")
            f.write(f"Detailed CV results are saved to `{CV_RESULTS_PATH.name}`.\n")

        with open(MODEL_COMPARISON_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("# Model Comparison Report\n\n")
            f.write(summary_df.to_markdown(index=False))
            f.write("\n")

        return (
            "SUCCESS: Multiple models trained with cross-validation and hyperparameter search. "
            "CV results and model card generated."
        )

    except Exception as e:
        return f"MODEL TRAINING FAILED: {e}"
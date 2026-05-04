# src/train.py

import joblib
import numpy as np
import pandas as pd

from crewai.tools import tool

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from src.model import build_model_candidates
from src.paths import (
    ALL_MODEL_PATHS,
    CV_RESULTS_PATH,
    MODEL_CARD_PATH,
    MODEL_COMPARISON_REPORT_PATH,
    MODEL_PATH,
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
)


def _small_param_grid(model_name):
    """Return a small grid so model selection stays fast."""
    grids = {
        "Logistic Regression": {
            "C": [0.1, 1.0, 10.0],
        },
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
        },
        "XGBoost": {
            "n_estimators": [80, 120],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        },
    }

    return grids.get(model_name)


def _load_training_data():
    """Load processed training features and target."""
    X_train = pd.read_csv(TRAIN_FEATURES_PATH)
    y_train = pd.read_csv(TRAIN_TARGET_PATH).values.ravel()

    return X_train, y_train


def _fit_with_grid_search(model_name, model, X_train, y_train, cv, param_grid):
    """Fit a model using GridSearchCV."""
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results["model"] = model_name

    return search.best_estimator_, float(search.best_score_), search.best_params_, cv_results


def _fit_with_default_params(model_name, model, X_train, y_train, cv):
    """Fit a model using default parameters and cross-validation."""
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    cv_results = pd.DataFrame({
        "model": [model_name],
        "mean_test_score": [float(np.mean(scores))],
        "params": ["default"],
    })

    return model, float(np.mean(scores)), "default", cv_results


def _train_one_model(model_name, model, X_train, y_train, cv):
    """Train one candidate model and return its summary."""
    param_grid = _small_param_grid(model_name)

    if param_grid:
        best_model, best_cv_auc, best_params, cv_results = _fit_with_grid_search(
            model_name=model_name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
            param_grid=param_grid,
        )
    else:
        best_model, best_cv_auc, best_params, cv_results = _fit_with_default_params(
            model_name=model_name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
        )

    model_path = ALL_MODEL_PATHS[model_name]
    joblib.dump(best_model, model_path)

    summary_row = {
        "model": model_name,
        "best_cv_auc": best_cv_auc,
        "best_params": str(best_params),
        "artifact": model_path.name,
    }

    return best_model, summary_row, cv_results


def _write_model_card(summary_df):
    """Write the model card from real training artifacts."""
    with open(MODEL_CARD_PATH, "w", encoding="utf-8") as f:
        f.write("# Model Card\n\n")
        f.write("## Training Summary\n\n")
        f.write(
            "Multiple model candidates were trained and compared using "
            "3-fold stratified cross-validation.\n\n"
        )
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Operational Governance Model\n\n")
        f.write(
            "The operational model saved as `best_model.pkl` is XGBoost "
            "for compatibility with the existing SAFE pipeline.\n\n"
        )
        f.write(f"Detailed CV results are saved to `{CV_RESULTS_PATH.name}`.\n")


def _write_model_comparison_report(summary_df):
    """Write a compact model-comparison report."""
    with open(MODEL_COMPARISON_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Model Comparison Report\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n")


@tool
def model_training_tool(description: str):
    """
    Train model candidates and save training artifacts.

    XGBoost remains the operational governance model saved as best_model.pkl.
    """
    try:
        X_train, y_train = _load_training_data()

        models = build_model_candidates()
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        trained_models = {}
        summary_rows = []
        cv_results_list = []

        for model_name, model in models.items():
            best_model, summary_row, cv_results = _train_one_model(
                model_name=model_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                cv=cv,
            )

            trained_models[model_name] = best_model
            summary_rows.append(summary_row)
            cv_results_list.append(cv_results)

        # Keep XGBoost as the current governance model.
        joblib.dump(trained_models["XGBoost"], MODEL_PATH)

        summary_df = pd.DataFrame(summary_rows).sort_values(
            "best_cv_auc",
            ascending=False,
        )
        cv_results_df = pd.concat(cv_results_list, ignore_index=True, sort=False)

        cv_results_df.to_csv(CV_RESULTS_PATH, index=False)
        _write_model_card(summary_df)
        _write_model_comparison_report(summary_df)

        return (
            "SUCCESS: Multiple model candidates trained with cross-validation and hyperparameter search. "
            f"Saved governance model: {MODEL_PATH.name}. "
            f"Generated model card: docs/{MODEL_CARD_PATH.name}. "
            f"Generated CV results: reports/{CV_RESULTS_PATH.name}. "
            f"Generated model comparison report: reports/{MODEL_COMPARISON_REPORT_PATH.name}."
        )

    except Exception as e:
        return f"MODEL TRAINING FAILED: {e}"
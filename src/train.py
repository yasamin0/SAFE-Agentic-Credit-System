# src/train.py

import joblib
import pandas as pd

from crewai.tools import tool

from src.model import build_model_candidates

from src.paths import (
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
    MODEL_PATH,
    MODEL_CARD_PATH,
    ALL_MODEL_PATHS,
)


@tool
def model_training_tool(description: str):
    """
    Train all model candidates needed for the SAFE AI paper comparison.

    The existing governance pipeline still uses best_model.pkl.
    To keep compatibility, best_model.pkl is saved as the XGBoost model.
    """
    try:
        X_train = pd.read_csv(TRAIN_FEATURES_PATH)
        y_train = pd.read_csv(TRAIN_TARGET_PATH).values.ravel()

        models = build_model_candidates()
        trained_rows = []

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            model_path = ALL_MODEL_PATHS[model_name]
            joblib.dump(model, model_path)

            trained_rows.append({
                "model": model_name,
                "artifact": model_path.name,
                "features": X_train.shape[1],
                "status": "trained"
            })

        # Keep the original pipeline compatible: best_model.pkl remains XGBoost.
        joblib.dump(models["XGBoost"], MODEL_PATH)

        trained_df = pd.DataFrame(trained_rows)

        with open(MODEL_CARD_PATH, "w", encoding="utf-8") as f:
            f.write("## Multi-Model Card\n\n")
            f.write("The following models were trained for SAFE AI paper-style comparison:\n\n")
            f.write(trained_df.to_markdown(index=False))
            f.write("\n\n")
            f.write("The main governance model saved as best_model.pkl is XGBoost.\n")

        return (
            "SUCCESS: All model candidates trained and saved. "
            "best_model.pkl remains the XGBoost governance model."
        )

    except Exception as e:
        return f"MODEL TRAINING FAILED: {e}"
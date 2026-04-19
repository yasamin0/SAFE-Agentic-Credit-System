import joblib
import pandas as pd
from crewai.tools import tool

from src.model import build_model
from src.paths import TRAIN_FEATURES_PATH, TRAIN_TARGET_PATH, MODEL_PATH, MODEL_CARD_PATH


@tool
def model_training_tool(description: str):
    """Trains an XGBoost model on the cleaned training data. Saves the model as 'best_model.pkl' and creates a 'model_card.md'."""
    try:
        X_train = pd.read_csv(TRAIN_FEATURES_PATH)
        y_train = pd.read_csv(TRAIN_TARGET_PATH).values.ravel()

        model = build_model()
        model.fit(X_train, y_train)

        joblib.dump(model, MODEL_PATH)

        with open(MODEL_CARD_PATH, "w", encoding="utf-8") as f:
            f.write(
                "## XGBoost Model Card\n\n"
                "- Model Type: XGBoost Classifier\n"
                "- Status: Trained\n"
                f"- Features: {X_train.shape[1]}"
            )

        return f"SUCCESS: Model trained and saved as '{MODEL_PATH.name}'. Model Card generated."
    except Exception as e:
        return f"MODEL TRAINING FAILED: {e}"
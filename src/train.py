# src/train.py

# joblib is used to save the trained model artifact to disk
import joblib

# pandas is used to read the processed training datasets
import pandas as pd

# CrewAI tool decorator so this training step can be called by the Modeling Agent
from crewai.tools import tool

# Model builder function that returns the configured XGBoost classifier
from src.model import build_model

# Centralized paths for processed training data, model artifact, and model card
from src.paths import TRAIN_FEATURES_PATH, TRAIN_TARGET_PATH, MODEL_PATH, MODEL_CARD_PATH


@tool
def model_training_tool(description: str):
    """
    Train the main XGBoost model on the processed training data.

    Workflow:
    - load the cleaned training features and target
    - build the configured model
    - fit the model on the training data
    - save the trained model artifact
    - generate a simple model card
    """
    try:
        # ------------------------------------------------------------
        # LOAD PROCESSED TRAINING DATA
        # ------------------------------------------------------------
        # X_train contains the processed feature matrix
        # y_train contains the target labels flattened into a 1D array
        X_train = pd.read_csv(TRAIN_FEATURES_PATH)
        y_train = pd.read_csv(TRAIN_TARGET_PATH).values.ravel()

        # ------------------------------------------------------------
        # BUILD AND TRAIN MODEL
        # ------------------------------------------------------------
        # build_model() returns the configured XGBoost classifier
        model = build_model()
        model.fit(X_train, y_train)

        # ------------------------------------------------------------
        # SAVE TRAINED MODEL ARTIFACT
        # ------------------------------------------------------------
        # Persist the trained model so it can be reused in evaluation
        joblib.dump(model, MODEL_PATH)

        # ------------------------------------------------------------
        # GENERATE MODEL CARD
        # ------------------------------------------------------------
        # Create a lightweight summary artifact describing the trained model
        with open(MODEL_CARD_PATH, "w", encoding="utf-8") as f:
            f.write(
                "## XGBoost Model Card\n\n"
                "- Model Type: XGBoost Classifier\n"
                "- Status: Trained\n"
                f"- Features: {X_train.shape[1]}"
            )

        # Return a short status message for the Modeling Agent
        return f"SUCCESS: Model trained and saved as '{MODEL_PATH.name}'. Model Card generated."

    except Exception as e:
        return f"MODEL TRAINING FAILED: {e}"
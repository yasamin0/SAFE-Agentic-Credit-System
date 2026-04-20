# src/model.py

# XGBoost classifier used as the predictive model for credit-risk classification
from xgboost import XGBClassifier

# Shared random seed from project configuration for reproducibility
from src.config import RANDOM_STATE


def build_model():
    """
    Build and return the main classification model used in the SAFE pipeline.

    In this project, the predictive core is an XGBoost classifier.
    The model is configured with a fixed set of baseline hyperparameters
    suitable for structured/tabular credit-risk data.
    """
    return XGBClassifier(
        # Number of boosting trees
        n_estimators=100,

        # Step size used when updating the model during boosting
        learning_rate=0.1,

        # Maximum depth of each tree
        max_depth=5,

        # Fixed seed for reproducibility
        random_state=RANDOM_STATE,

        # Evaluation metric used internally by XGBoost during training
        eval_metric="logloss"
    )
# src/config.py

# Standard library import for reading environment variables
import os

# Loads values from the local .env file into the environment
from dotenv import load_dotenv

# Shared LLM interface used by CrewAI agents and the grounded chatbot
from crewai import LLM

# Load .env variables before reading configuration values
load_dotenv()


# ------------------------------------------------------------
# DATA SOURCE CONFIGURATION
# ------------------------------------------------------------

# Optional local CSV path.
# If this is provided, the pipeline will use the local dataset.
# If it is empty, the pipeline will fall back to OpenML.
DATA_PATH = os.getenv("DATA_PATH", "").strip() or None

# Default OpenML dataset ID used when DATA_PATH is not provided.
# In this project, OpenML ID 31 corresponds to the German Credit dataset.
OPENML_ID = int(os.getenv("OPENML_ID", "31"))


# ------------------------------------------------------------
# DECISION / GOVERNANCE THRESHOLDS
# ------------------------------------------------------------

# Prediction threshold used to convert model probabilities into binary predictions.
# Example:
# - if predicted probability >= PRED_THRESHOLD -> class 1
# - otherwise -> class 0
PRED_THRESHOLD = float(os.getenv("PRED_THRESHOLD", "0.50"))

# Final approval threshold used for the governance decision.
# If SAFE_SCORE >= APPROVAL_THRESHOLD -> APPROVED
# Otherwise -> REJECTED
APPROVAL_THRESHOLD = float(os.getenv("APPROVAL_THRESHOLD", "0.75"))


# ------------------------------------------------------------
# SAFE SCORE WEIGHTS
# ------------------------------------------------------------

# Weight of predictive performance (AUC) in the final SAFE score
W_AUC = float(os.getenv("W_AUC", "0.4"))

# Weight of fairness aggregate in the final SAFE score
W_FAIR = float(os.getenv("W_FAIR", "0.4"))

# Weight of robustness aggregate in the final SAFE score
W_ROB = float(os.getenv("W_ROB", "0.2"))

# Normalize weights so they always sum to 1.
# This prevents invalid policy settings such as all-zero weights
# and ensures the SAFE score remains a proper weighted combination.
w_sum = W_AUC + W_FAIR + W_ROB
if w_sum <= 0:
    raise ValueError("Weights must sum to > 0")

W_AUC, W_FAIR, W_ROB = W_AUC / w_sum, W_FAIR / w_sum, W_ROB / w_sum


# ------------------------------------------------------------
# FAIRNESS / SENSITIVE FEATURE CONFIGURATION
# ------------------------------------------------------------

# Main sensitive feature used for fairness analysis
# Example values may include personal_status, age, or foreign_worker
SENSITIVE_FEATURE = os.getenv("SENSITIVE_FEATURE", "personal_status")

# If set to "1", the sensitive feature will be removed from the model input.
# If set to "0", it remains in the model input but is still tracked separately.
DROP_SENSITIVE_FROM_MODEL = os.getenv("DROP_SENSITIVE_FROM_MODEL", "0") == "1"

# Alternative sensitive features used during sensitivity analysis.
# This helps test how the governance result changes if fairness is defined
# using a different sensitive attribute.
ALT_SENSITIVE_FEATURES = [
    x.strip() for x in os.getenv("ALT_SENSITIVE_FEATURES", "foreign_worker,sex,age").split(",") if x.strip()
]


# ------------------------------------------------------------
# REPRODUCIBILITY / ROBUSTNESS SETTINGS
# ------------------------------------------------------------

# Random seed used for reproducibility in train/test split and other randomized steps
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# Standard deviation of Gaussian noise used in robustness testing
ROBUST_NOISE_STD = float(os.getenv("ROBUST_NOISE_STD", "0.10"))

# Fraction of columns dropped/zeroed during feature-dropout robustness testing
ROBUST_DROPOUT_RATE = float(os.getenv("ROBUST_DROPOUT_RATE", "0.10"))

# Fraction of rows affected during missingness robustness testing
ROBUST_MISSING_RATE = float(os.getenv("ROBUST_MISSING_RATE", "0.10"))


# ------------------------------------------------------------
# SHARED LLM CONFIGURATION
# ------------------------------------------------------------

# Shared LLM instance used across agents and the grounded chatbot.
# The model name is read from .env and defaults to gpt-4o.
crew_llm = LLM(model=os.getenv("CREW_LLM_MODEL", "gpt-4o"))


def current_config():
    """
    Return the current pipeline configuration as a structured dictionary.

    This is used in generated artifacts such as the Data Card so that
    every run records:
    - the data source
    - decision thresholds
    - SAFE policy weights
    - fairness-sensitive settings
    - robustness settings
    - reproducibility settings
    """
    return {
        # Shows whether the current run used a local CSV or OpenML dataset
        "data_source": f"CSV ({DATA_PATH})" if DATA_PATH else f"OpenML ({OPENML_ID})",

        # Threshold for turning probabilities into binary predictions
        "prediction_threshold": PRED_THRESHOLD,

        # Threshold for turning the SAFE score into APPROVED / REJECTED
        "approval_threshold": APPROVAL_THRESHOLD,

        # Policy weights for the SAFE score formula
        "weights": {
            "auc": W_AUC,
            "fairness": W_FAIR,
            "robustness": W_ROB
        },

        # Fairness-related configuration
        "sensitive_feature": SENSITIVE_FEATURE,
        "alternative_sensitive_features": ALT_SENSITIVE_FEATURES,
        "drop_sensitive_from_model": DROP_SENSITIVE_FROM_MODEL,

        # Reproducibility setting
        "random_state": RANDOM_STATE,

        # Robustness test configuration
        "robustness_settings": {
            "noise_std": ROBUST_NOISE_STD,
            "dropout_rate": ROBUST_DROPOUT_RATE,
            "missing_rate": ROBUST_MISSING_RATE,
        },

        # Human-readable explanation of the governance decision rule
        "decision_rule": (
            "APPROVED if SAFE_SCORE >= APPROVAL_THRESHOLD else REJECTED, "
            "where SAFE_SCORE = W_AUC*AUC + W_FAIR*FAIRNESS_AGG + W_ROB*ROBUSTNESS_AGG"
        ),
    }
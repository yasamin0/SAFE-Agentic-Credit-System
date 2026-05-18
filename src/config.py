# src/config.py

import os

from dotenv import load_dotenv
from crewai import LLM


load_dotenv()


# -----------------------------
# Data source
# -----------------------------

# Use local CSV if DATA_PATH is set; otherwise use OpenML.
DATA_PATH = os.getenv("DATA_PATH", "").strip() or None
OPENML_ID = int(os.getenv("OPENML_ID", "31"))


# -----------------------------
# Governance thresholds
# -----------------------------

# Probability threshold for binary predictions.
PRED_THRESHOLD = float(os.getenv("PRED_THRESHOLD", "0.50"))

# Minimum SAFE score required for approval.
APPROVAL_THRESHOLD = float(os.getenv("APPROVAL_THRESHOLD", "0.75"))


# -----------------------------
# SAFE score weights
# -----------------------------

# Equal weights requested for the final SAFE score:
# SAFE = 0.25*RGA + 0.25*RGR + 0.25*RGE + 0.25*Fairness
W_RGA = float(os.getenv("W_RGA", "0.25"))
W_RGR = float(os.getenv("W_RGR", "0.25"))
W_RGE = float(os.getenv("W_RGE", "0.25"))
W_FAIR = float(os.getenv("W_FAIR", "0.25"))

# Normalize weights to keep the SAFE score well-defined.
w_sum = W_RGA + W_RGR + W_RGE + W_FAIR
if w_sum <= 0:
    raise ValueError("Weights must sum to > 0")

W_RGA = W_RGA / w_sum
W_RGR = W_RGR / w_sum
W_RGE = W_RGE / w_sum
W_FAIR = W_FAIR / w_sum


# -----------------------------
# Fairness settings
# -----------------------------

SENSITIVE_FEATURE = os.getenv("SENSITIVE_FEATURE", "personal_status")
DROP_SENSITIVE_FROM_MODEL = os.getenv("DROP_SENSITIVE_FROM_MODEL", "0") == "1"

ALT_SENSITIVE_FEATURES = [
    x.strip()
    for x in os.getenv("ALT_SENSITIVE_FEATURES", "foreign_worker,sex,age").split(",")
    if x.strip()
]


# -----------------------------
# Reproducibility and robustness
# -----------------------------

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

ROBUST_NOISE_STD = float(os.getenv("ROBUST_NOISE_STD", "0.10"))
ROBUST_DROPOUT_RATE = float(os.getenv("ROBUST_DROPOUT_RATE", "0.10"))
ROBUST_MISSING_RATE = float(os.getenv("ROBUST_MISSING_RATE", "0.10"))


# -----------------------------
# Shared LLM
# -----------------------------

crew_llm = LLM(model=os.getenv("CREW_LLM_MODEL", "gpt-4o"))


def current_config():
    """Return the configuration recorded in generated artifacts."""
    return {
        "data_source": f"CSV ({DATA_PATH})" if DATA_PATH else f"OpenML ({OPENML_ID})",
        "prediction_threshold": PRED_THRESHOLD,
        "approval_threshold": APPROVAL_THRESHOLD,
        "weights": {
            "rga": W_RGA,
            "rgr": W_RGR,
            "rge": W_RGE,
            "fairness": W_FAIR,
        },
        "sensitive_feature": SENSITIVE_FEATURE,
        "alternative_sensitive_features": ALT_SENSITIVE_FEATURES,
        "drop_sensitive_from_model": DROP_SENSITIVE_FROM_MODEL,
        "random_state": RANDOM_STATE,
        "robustness_settings": {
            "noise_std": ROBUST_NOISE_STD,
            "dropout_rate": ROBUST_DROPOUT_RATE,
            "missing_rate": ROBUST_MISSING_RATE,
        },
        "decision_rule": (
            "APPROVED if SAFE_SCORE >= APPROVAL_THRESHOLD else REJECTED, "
            "where SAFE_SCORE = W_RGA*AURGA + W_RGR*RGR_AGG + W_RGE*AURGE + W_FAIR*FAIRNESS_AGG"
        ),
    }


import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "").strip() or None
OPENML_ID = int(os.getenv("OPENML_ID", "31"))

PRED_THRESHOLD = float(os.getenv("PRED_THRESHOLD", "0.50"))
APPROVAL_THRESHOLD = float(os.getenv("APPROVAL_THRESHOLD", "0.75"))

W_AUC = float(os.getenv("W_AUC", "0.4"))
W_FAIR = float(os.getenv("W_FAIR", "0.4"))
W_ROB = float(os.getenv("W_ROB", "0.2"))

w_sum = W_AUC + W_FAIR + W_ROB
if w_sum <= 0:
    raise ValueError("Weights must sum to > 0")
W_AUC, W_FAIR, W_ROB = W_AUC / w_sum, W_FAIR / w_sum, W_ROB / w_sum

SENSITIVE_FEATURE = os.getenv("SENSITIVE_FEATURE", "personal_status")
DROP_SENSITIVE_FROM_MODEL = os.getenv("DROP_SENSITIVE_FROM_MODEL", "0") == "1"
ALT_SENSITIVE_FEATURES = [
    x.strip() for x in os.getenv("ALT_SENSITIVE_FEATURES", "foreign_worker,sex,age").split(",") if x.strip()
]

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
ROBUST_NOISE_STD = float(os.getenv("ROBUST_NOISE_STD", "0.10"))
ROBUST_DROPOUT_RATE = float(os.getenv("ROBUST_DROPOUT_RATE", "0.10"))
ROBUST_MISSING_RATE = float(os.getenv("ROBUST_MISSING_RATE", "0.10"))

crew_llm = LLM(model=os.getenv("CREW_LLM_MODEL", "gpt-4o"))


def current_config():
    return {
        "data_source": f"CSV ({DATA_PATH})" if DATA_PATH else f"OpenML ({OPENML_ID})",
        "prediction_threshold": PRED_THRESHOLD,
        "approval_threshold": APPROVAL_THRESHOLD,
        "weights": {"auc": W_AUC, "fairness": W_FAIR, "robustness": W_ROB},
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
            "where SAFE_SCORE = W_AUC*AUC + W_FAIR*FAIRNESS_AGG + W_ROB*ROBUSTNESS_AGG"
        ),
    }
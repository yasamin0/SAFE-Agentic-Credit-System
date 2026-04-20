# src/paths.py

# pathlib is used for clean, platform-independent path management
from pathlib import Path

# Base project directory:
# this points to the root of the repository
BASE_DIR = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# MAIN PROJECT FOLDERS
# ------------------------------------------------------------

# Root folder for all dataset-related artifacts
DATA_DIR = BASE_DIR / "data"

# Stores the original/raw dataset
RAW_DIR = DATA_DIR / "raw"

# Stores processed machine-learning-ready train/test files
PROCESSED_DIR = DATA_DIR / "processed"

# Stores sensitive-feature splits used for fairness analysis
SENSITIVE_DIR = DATA_DIR / "sensitive"

# Stores documentation-style artifacts such as cards and summaries
DOCS_DIR = BASE_DIR / "docs"

# Stores trained model artifacts
MODELS_DIR = BASE_DIR / "models"

# Stores generated reports from the evaluation stage
REPORTS_DIR = BASE_DIR / "reports"


# ------------------------------------------------------------
# RAW DATA PATH
# ------------------------------------------------------------

# Saved raw credit dataset used as the input to preprocessing
RAW_DATA_PATH = RAW_DIR / "raw_credit_data.csv"


# ------------------------------------------------------------
# PROCESSED DATA PATHS
# ------------------------------------------------------------

# Clean training features after preprocessing
TRAIN_FEATURES_PATH = PROCESSED_DIR / "clean_train_features.csv"

# Clean training target labels
TRAIN_TARGET_PATH = PROCESSED_DIR / "clean_train_target.csv"

# Clean test features after preprocessing
TEST_FEATURES_PATH = PROCESSED_DIR / "clean_test_features.csv"

# Clean test target labels
TEST_TARGET_PATH = PROCESSED_DIR / "clean_test_target.csv"


# ------------------------------------------------------------
# FAIRNESS / SENSITIVE FEATURE PATHS
# ------------------------------------------------------------

# Sensitive feature values for the training split
SENSITIVE_TRAIN_PATH = SENSITIVE_DIR / "sensitive_train.csv"

# Sensitive feature values for the test split
SENSITIVE_TEST_PATH = SENSITIVE_DIR / "sensitive_test.csv"


# ------------------------------------------------------------
# MODEL ARTIFACT PATH
# ------------------------------------------------------------

# Saved trained classifier artifact
MODEL_PATH = MODELS_DIR / "best_model.pkl"


# ------------------------------------------------------------
# DOCUMENTATION ARTIFACT PATHS
# ------------------------------------------------------------

# Data card describing preprocessing outputs and configuration
DATACARD_PATH = DOCS_DIR / "datacard.json"

# Model card describing the trained model
MODEL_CARD_PATH = DOCS_DIR / "model_card.md"

# System card describing the final SAFE decision
SYSTEM_CARD_PATH = DOCS_DIR / "system_card.md"


# ------------------------------------------------------------
# REPORT ARTIFACT PATHS
# ------------------------------------------------------------

# Main evaluation summary report
EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.md"

# Detailed final report with fairness, robustness, sensitivity, and explainability
FINAL_REPORT_PATH = REPORTS_DIR / "final_report.md"

# Dedicated report for sensitivity and interaction analysis
SENSITIVITY_REPORT_PATH = REPORTS_DIR / "sensitivity_report.md"

# Markdown log of chatbot conversations
CHATBOT_LOG_PATH = REPORTS_DIR / "chatbot_log.md"


def ensure_directories():
    """
    Ensure that all required output directories exist before the pipeline runs.

    This is called at startup so the system can safely write:
    - raw data
    - processed data
    - sensitive-feature files
    - model artifacts
    - documentation artifacts
    - evaluation reports

    If a directory already exists, it is left unchanged.
    """
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        SENSITIVE_DIR,
        DOCS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
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

# ------------------------------------------------------------
# RGR / RANK-BASED ROBUSTNESS ARTIFACT PATHS
# ------------------------------------------------------------

RGR_GAUSSIAN_CSV_PATH = REPORTS_DIR / "rgr_gaussian_curve.csv"
RGR_SWAPPING_CSV_PATH = REPORTS_DIR / "rgr_swapping_curve.csv"

RGR_GAUSSIAN_PLOT_PATH = REPORTS_DIR / "rgr_gaussian_curve.png"
RGR_SWAPPING_PLOT_PATH = REPORTS_DIR / "rgr_swapping_curve.png"

RGR_REPORT_PATH = REPORTS_DIR / "rgr_report.md"

# ------------------------------------------------------------
# RGE / RANK-BASED EXPLAINABILITY ARTIFACT PATHS
# ------------------------------------------------------------
# These paths store the paper-style explainability outputs.
# RGE measures how prediction rankings change when features are removed.

RGE_IMPORTANCE_CSV_PATH = REPORTS_DIR / "rge_feature_importance.csv"
RGE_CURVE_CSV_PATH = REPORTS_DIR / "rge_curve.csv"

RGE_PLOT_PATH = REPORTS_DIR / "rge_curve.png"
RGE_IMPORTANCE_PLOT_PATH = REPORTS_DIR / "rge_feature_importance.png"

RGE_REPORT_PATH = REPORTS_DIR / "rge_report.md"

# ------------------------------------------------------------
# MULTI-MODEL ARTIFACT PATHS
# ------------------------------------------------------------
# These paths store the additional models needed for the SAFE AI paper comparison.

MODEL_LR_PATH = MODELS_DIR / "model_logistic_regression.pkl"
MODEL_RF_PATH = MODELS_DIR / "model_random_forest.pkl"
MODEL_XGB_PATH = MODELS_DIR / "model_xgboost.pkl"
MODEL_VOTING_PATH = MODELS_DIR / "model_voting_ensemble.pkl"
MODEL_STACKING_PATH = MODELS_DIR / "model_stacking_ensemble.pkl"
MODEL_RANDOM_BASELINE_PATH = MODELS_DIR / "model_random_baseline.pkl"

ALL_MODEL_PATHS = {
    "Logistic Regression": MODEL_LR_PATH,
    "Random Forest": MODEL_RF_PATH,
    "XGBoost": MODEL_XGB_PATH,
    "Voting Ensemble": MODEL_VOTING_PATH,
    "Stacking Ensemble": MODEL_STACKING_PATH,
    "Random Baseline": MODEL_RANDOM_BASELINE_PATH,
}


# ------------------------------------------------------------
# RGA / RANK-BASED ACCURACY ARTIFACT PATHS
# ------------------------------------------------------------
# RGA measures predictive accuracy using a rank-based curve.

RGA_CURVE_CSV_PATH = REPORTS_DIR / "rga_curve.csv"
RGA_PLOT_PATH = REPORTS_DIR / "rga_curve.png"
RGA_REPORT_PATH = REPORTS_DIR / "rga_report.md"


# ------------------------------------------------------------
# SHAP VS RGE COMPARISON ARTIFACT PATHS
# ------------------------------------------------------------
# These files compare RGE feature ranking with SHAP feature ranking.

SHAP_RGE_COMPARISON_CSV_PATH = REPORTS_DIR / "shap_rge_comparison.csv"
SHAP_RGE_REPORT_PATH = REPORTS_DIR / "shap_rge_report.md"


# ------------------------------------------------------------
# SAFE AI PAPER COMPLIANCE SCORE ARTIFACT PATHS
# ------------------------------------------------------------
# These files store the final comparison tables and plots across models.

MODEL_METRICS_COMPARISON_CSV_PATH = REPORTS_DIR / "model_metrics_comparison.csv"
COMPLIANCE_SCORE_CSV_PATH = REPORTS_DIR / "compliance_score_comparison.csv"
COMPLIANCE_SCORE_PLOT_PATH = REPORTS_DIR / "compliance_score_comparison.png"
SAFE_PAPER_METRICS_REPORT_PATH = REPORTS_DIR / "safe_paper_metrics_report.md"

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
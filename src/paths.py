# src/paths.py

from pathlib import Path


# Root directory
BASE_DIR = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# Main folders
# ------------------------------------------------------------

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SENSITIVE_DIR = DATA_DIR / "sensitive"

DOCS_DIR = BASE_DIR / "docs"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# ------------------------------------------------------------
# Data artifacts
# ------------------------------------------------------------

RAW_DATA_PATH = RAW_DIR / "raw_credit_data.csv"

TRAIN_FEATURES_PATH = PROCESSED_DIR / "clean_train_features.csv"
TRAIN_TARGET_PATH = PROCESSED_DIR / "clean_train_target.csv"
TEST_FEATURES_PATH = PROCESSED_DIR / "clean_test_features.csv"
TEST_TARGET_PATH = PROCESSED_DIR / "clean_test_target.csv"

CLEAN_TRAIN_FULL_PATH = PROCESSED_DIR / "clean_train.csv"
CLEAN_TEST_FULL_PATH = PROCESSED_DIR / "clean_test.csv"

SENSITIVE_TRAIN_PATH = SENSITIVE_DIR / "sensitive_train.csv"
SENSITIVE_TEST_PATH = SENSITIVE_DIR / "sensitive_test.csv"


# ------------------------------------------------------------
# Model artifacts
# ------------------------------------------------------------

MODEL_PATH = MODELS_DIR / "best_model.pkl"

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
# Documentation artifacts
# ------------------------------------------------------------

DATACARD_PATH = DOCS_DIR / "datacard.json"
MODEL_CARD_PATH = DOCS_DIR / "model_card.md"
SYSTEM_CARD_PATH = DOCS_DIR / "system_card.md"


# ------------------------------------------------------------
# Main report artifacts
# ------------------------------------------------------------

EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.md"
FINAL_REPORT_PATH = REPORTS_DIR / "final_report.md"
SENSITIVITY_REPORT_PATH = REPORTS_DIR / "sensitivity_report.md"
CHATBOT_LOG_PATH = REPORTS_DIR / "chatbot_log.md"

OUTLIER_REPORT_PATH = REPORTS_DIR / "outlier_analysis_report.md"
MODEL_COMPARISON_REPORT_PATH = REPORTS_DIR / "model_comparison_report.md"
CV_RESULTS_PATH = REPORTS_DIR / "cv_results.csv"


# ------------------------------------------------------------
# EDA and classification artifacts
# ------------------------------------------------------------

EDA_TARGET_DISTRIBUTION_PLOT_PATH = FIGURES_DIR / "eda_target_distribution.png"
EDA_NUMERIC_DISTRIBUTION_PLOT_PATH = FIGURES_DIR / "eda_numeric_distributions.png"
EDA_MISSING_VALUES_PLOT_PATH = FIGURES_DIR / "eda_missing_values.png"

CLASSIFICATION_METRICS_CSV_PATH = REPORTS_DIR / "classification_metrics.csv"
CONFUSION_MATRIX_CSV_PATH = REPORTS_DIR / "confusion_matrix.csv"
CONFUSION_MATRIX_PLOT_PATH = FIGURES_DIR / "confusion_matrix.png"
CALIBRATION_CURVE_CSV_PATH = REPORTS_DIR / "calibration_curve.csv"
CALIBRATION_CURVE_PLOT_PATH = FIGURES_DIR / "calibration_curve.png"


# ------------------------------------------------------------
# RGA / RGR / RGE artifacts
# ------------------------------------------------------------

RGA_CURVE_CSV_PATH = REPORTS_DIR / "rga_curve.csv"
RGA_PLOT_PATH = REPORTS_DIR / "rga_curve.png"
RGA_REPORT_PATH = REPORTS_DIR / "rga_report.md"

RGR_GAUSSIAN_CSV_PATH = REPORTS_DIR / "rgr_gaussian_curve.csv"
RGR_SWAPPING_CSV_PATH = REPORTS_DIR / "rgr_swapping_curve.csv"
RGR_GAUSSIAN_PLOT_PATH = REPORTS_DIR / "rgr_gaussian_curve.png"
RGR_SWAPPING_PLOT_PATH = REPORTS_DIR / "rgr_swapping_curve.png"
RGR_REPORT_PATH = REPORTS_DIR / "rgr_report.md"

RGE_IMPORTANCE_CSV_PATH = REPORTS_DIR / "rge_feature_importance.csv"
RGE_CURVE_CSV_PATH = REPORTS_DIR / "rge_curve.csv"
RGE_PLOT_PATH = REPORTS_DIR / "rge_curve.png"
RGE_IMPORTANCE_PLOT_PATH = REPORTS_DIR / "rge_feature_importance.png"
RGE_REPORT_PATH = REPORTS_DIR / "rge_report.md"


# ------------------------------------------------------------
# SHAP and compliance artifacts
# ------------------------------------------------------------

SHAP_RGE_COMPARISON_CSV_PATH = REPORTS_DIR / "shap_rge_comparison.csv"
SHAP_RGE_REPORT_PATH = REPORTS_DIR / "shap_rge_report.md"

MODEL_METRICS_COMPARISON_CSV_PATH = REPORTS_DIR / "model_metrics_comparison.csv"
COMPLIANCE_SCORE_CSV_PATH = REPORTS_DIR / "compliance_score_comparison.csv"
COMPLIANCE_SCORE_PLOT_PATH = REPORTS_DIR / "compliance_score_comparison.png"
SAFE_PAPER_METRICS_REPORT_PATH = REPORTS_DIR / "safe_paper_metrics_report.md"


def ensure_directories():
    """Create all folders needed by the pipeline."""
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        SENSITIVE_DIR,
        DOCS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
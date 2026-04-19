from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SENSITIVE_DIR = DATA_DIR / "sensitive"

DOCS_DIR = BASE_DIR / "docs"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RAW_DATA_PATH = RAW_DIR / "raw_credit_data.csv"

TRAIN_FEATURES_PATH = PROCESSED_DIR / "clean_train_features.csv"
TRAIN_TARGET_PATH = PROCESSED_DIR / "clean_train_target.csv"
TEST_FEATURES_PATH = PROCESSED_DIR / "clean_test_features.csv"
TEST_TARGET_PATH = PROCESSED_DIR / "clean_test_target.csv"

SENSITIVE_TRAIN_PATH = SENSITIVE_DIR / "sensitive_train.csv"
SENSITIVE_TEST_PATH = SENSITIVE_DIR / "sensitive_test.csv"

MODEL_PATH = MODELS_DIR / "best_model.pkl"

DATACARD_PATH = DOCS_DIR / "datacard.json"
MODEL_CARD_PATH = DOCS_DIR / "model_card.md"
SYSTEM_CARD_PATH = DOCS_DIR / "system_card.md"

EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.md"
FINAL_REPORT_PATH = REPORTS_DIR / "final_report.md"
SENSITIVITY_REPORT_PATH = REPORTS_DIR / "sensitivity_report.md"
CHATBOT_LOG_PATH = REPORTS_DIR / "chatbot_log.md"


def ensure_directories():
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        SENSITIVE_DIR,
        DOCS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
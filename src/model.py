from xgboost import XGBClassifier
from src.config import RANDOM_STATE


def build_model():
    return XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )
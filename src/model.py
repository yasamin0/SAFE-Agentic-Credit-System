# src/model.py

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.dummy import DummyClassifier

from src.config import RANDOM_STATE


def build_model():
    """
    Build the main model used by the current governance pipeline.
    We keep XGBoost as the default model to avoid breaking the existing system.
    """
    return XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )


def build_model_candidates():
    """
    Build all models needed for the SAFE AI paper comparison.

    Models:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - Voting Ensemble
    - Stacking Ensemble
    - Random Baseline
    """

    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )

    voting = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
            ("rf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
            ("xgb", XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE, eval_metric="logloss")),
        ],
        voting="soft"
    )

    stacking = StackingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
            ("rf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
            ("xgb", XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE, eval_metric="logloss")),
        ],
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        n_jobs=-1
    )

    random_baseline = DummyClassifier(
        strategy="prior",
        random_state=RANDOM_STATE
    )

    return {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "XGBoost": xgb,
        "Voting Ensemble": voting,
        "Stacking Ensemble": stacking,
        "Random Baseline": random_baseline,
    }
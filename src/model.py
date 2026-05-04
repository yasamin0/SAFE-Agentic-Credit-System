# src/model.py

from xgboost import XGBClassifier

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE


def build_logistic_regression():
    """Build the logistic regression baseline."""
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def build_random_forest():
    """Build the random forest baseline."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_xgboost():
    """Build the XGBoost governance model."""
    return XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )


def build_voting_ensemble():
    """Build a soft-voting ensemble from LR, RF, and XGBoost."""
    return VotingClassifier(
        estimators=[
            ("lr", build_logistic_regression()),
            ("rf", build_random_forest()),
            ("xgb", build_xgboost()),
        ],
        voting="soft",
    )


def build_stacking_ensemble():
    """Build a stacking ensemble from LR, RF, and XGBoost."""
    return StackingClassifier(
        estimators=[
            ("lr", build_logistic_regression()),
            ("rf", build_random_forest()),
            ("xgb", build_xgboost()),
        ],
        final_estimator=LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        stack_method="predict_proba",
        n_jobs=-1,
    )


def build_random_baseline():
    """Build a prior-based random baseline."""
    return DummyClassifier(
        strategy="prior",
        random_state=RANDOM_STATE,
    )


def build_model():
    """
    Build the main governance model.

    The pipeline still uses XGBoost as best_model.pkl to stay compatible
    with the existing evaluation and governance flow.
    """
    return build_xgboost()


def build_model_candidates():
    """Build all model candidates used for comparison tables."""
    return {
        "Logistic Regression": build_logistic_regression(),
        "Random Forest": build_random_forest(),
        "XGBoost": build_xgboost(),
        "Voting Ensemble": build_voting_ensemble(),
        "Stacking Ensemble": build_stacking_ensemble(),
        "Random Baseline": build_random_baseline(),
    }
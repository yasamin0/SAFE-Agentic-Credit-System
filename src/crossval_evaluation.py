# src/crossval_evaluation.py

import numpy as np
import pandas as pd

from scipy.stats import t
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import (
    RANDOM_STATE,
    SENSITIVE_FEATURE,
    DROP_SENSITIVE_FROM_MODEL,
    PRED_THRESHOLD,
    W_RGA,
    W_RGR,
    W_RGE,
    W_FAIR,
)

from src.data_loader import (
    _build_one_hot_encoder,
    _clean_feature_names,
)

from src.fairness import _compute_fairness_metrics
from src.rga import compute_rga_curve
from src.rgr import (
    compute_rgr_curve,
    compute_reference_rgr,
)
from src.rge import (
    compute_rge_feature_importance,
    compute_rge_curve,
)


def _build_fold_preprocessor(X_train):
    """
    Build preprocessing separately inside each fold.

    This prevents information from the validation fold from leaking
    into preprocessing.
    """
    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    model_categorical_features = [
        c for c in categorical_features
        if not (
            DROP_SENSITIVE_FROM_MODEL
            and c == SENSITIVE_FEATURE
        )
    ]

    model_numerical_features = [
        c for c in numerical_features
        if not (
            DROP_SENSITIVE_FROM_MODEL
            and c == SENSITIVE_FEATURE
        )
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                _build_one_hot_encoder(),
                model_categorical_features,
            ),
            (
                "num",
                StandardScaler(),
                model_numerical_features,
            ),
        ],
        remainder="drop",
    )

    return (
        preprocessor,
        model_categorical_features,
        model_numerical_features,
    )


def _process_fold(
    X_train_raw,
    X_val_raw,
):
    """
    Fit preprocessing only on the training part of a fold,
    then transform both training and validation data.
    """
    (
        preprocessor,
        categorical_features,
        numerical_features,
    ) = _build_fold_preprocessor(X_train_raw)

    X_train_model = X_train_raw.copy()
    X_val_model = X_val_raw.copy()

    if (
        DROP_SENSITIVE_FROM_MODEL
        and SENSITIVE_FEATURE in X_train_model.columns
    ):
        X_train_model = X_train_model.drop(
            columns=[SENSITIVE_FEATURE]
        )
        X_val_model = X_val_model.drop(
            columns=[SENSITIVE_FEATURE]
        )

    X_train_processed = preprocessor.fit_transform(
        X_train_model
    )

    X_val_processed = preprocessor.transform(
        X_val_model
    )

    raw_feature_names = []

    if categorical_features:
        raw_feature_names.extend(
            preprocessor
            .named_transformers_["cat"]
            .get_feature_names_out(categorical_features)
            .tolist()
        )

    raw_feature_names.extend(numerical_features)

    clean_feature_names = _clean_feature_names(
        raw_feature_names
    )

    X_train_df = pd.DataFrame(
        X_train_processed,
        columns=clean_feature_names,
        index=X_train_raw.index,
    )

    X_val_df = pd.DataFrame(
        X_val_processed,
        columns=clean_feature_names,
        index=X_val_raw.index,
    )

    return (
        X_train_df,
        X_val_df,
        numerical_features,
    )


def _compute_safe_score(
    aurga,
    rgr_aggregate,
    aurge,
    fairness_aggregate,
):
    """
    Compute the paper-based SAFE score.
    """
    return float(
        W_RGA * aurga
        + W_RGR * rgr_aggregate
        + W_RGE * aurge
        + W_FAIR * fairness_aggregate
    )


def _mean_sd_ci(values, confidence_level=0.95):
    """
    Compute mean, sample SD, and Student-t confidence interval
    across cross-validation folds.
    """
    values = np.asarray(values, dtype=float)

    values = values[
        np.isfinite(values)
    ]

    n = len(values)

    if n == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_folds": 0,
        }

    mean = float(np.mean(values))

    if n == 1:
        return {
            "mean": mean,
            "std": 0.0,
            "ci_lower": mean,
            "ci_upper": mean,
            "n_folds": 1,
        }

    std = float(
        np.std(
            values,
            ddof=1,
        )
    )

    standard_error = std / np.sqrt(n)

    alpha = 1.0 - confidence_level

    critical_value = float(
        t.ppf(
            1.0 - alpha / 2.0,
            df=n - 1,
        )
    )

    margin = critical_value * standard_error

    return {
        "mean": mean,
        "std": std,
        "ci_lower": float(mean - margin),
        "ci_upper": float(mean + margin),
        "n_folds": int(n),
    }


def run_five_fold_safe_evaluation(
    model,
    raw_df,
):
    """
    Run complete 5-fold SAFE evaluation.

    For each validation fold:
    - fit fold-specific preprocessing
    - clone and train the selected model
    - calculate AUC
    - calculate AURGA
    - calculate RGR
    - calculate AURGE
    - calculate Fairness Aggregate
    - calculate final SAFE score

    Returns:
    1. fold-level results
    2. mean / SD / 95% CI summary
    """
    if "CreditRisk" not in raw_df.columns:
        raise ValueError(
            "raw_df must contain the CreditRisk target column."
        )

    X = raw_df.drop(
        columns=["CreditRisk"]
    )

    y = raw_df["CreditRisk"].astype(int)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows = []

    for fold_number, (train_idx, val_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        X_train_raw = X.iloc[train_idx].copy()
        X_val_raw = X.iloc[val_idx].copy()

        y_train = y.iloc[train_idx].copy()
        y_val = y.iloc[val_idx].copy()

        # --------------------------------------------
        # Sensitive group for validation fold
        # --------------------------------------------
        if SENSITIVE_FEATURE in X_val_raw.columns:
            group_val = (
                X_val_raw[SENSITIVE_FEATURE]
                .astype(str)
                .fillna("NA")
                .reset_index(drop=True)
            )
        else:
            group_val = pd.Series(
                ["UNKNOWN"] * len(X_val_raw)
            )

        # --------------------------------------------
        # Fold-specific preprocessing
        # --------------------------------------------
        (
            X_train_processed,
            X_val_processed,
            numerical_features,
        ) = _process_fold(
            X_train_raw=X_train_raw,
            X_val_raw=X_val_raw,
        )

        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        X_train_processed = X_train_processed.reset_index(
            drop=True
        )

        X_val_processed = X_val_processed.reset_index(
            drop=True
        )

        # --------------------------------------------
        # Clone selected model and refit
        # --------------------------------------------
        fold_model = clone(model)

        fold_model.fit(
            X_train_processed,
            y_train,
        )

        y_probs = fold_model.predict_proba(
            X_val_processed
        )[:, 1]

        # --------------------------------------------
        # AUC
        # --------------------------------------------
        auc_score = float(
            roc_auc_score(
                y_val,
                y_probs,
            )
        )

        # --------------------------------------------
        # FAIRNESS
        # --------------------------------------------
        fairness_metrics, _, _ = (
            _compute_fairness_metrics(
                y_true=y_val,
                y_probs=y_probs,
                group=group_val,
                pred_threshold=PRED_THRESHOLD,
            )
        )

        fairness_aggregate = float(
            fairness_metrics["fairness_aggregate"]
        )

        fairness_score_spd = float(
            fairness_metrics[
                "fairness_score_spd"
            ]
        )

        fairness_score_eod = float(
            fairness_metrics[
                "fairness_score_eod"
            ]
        )

        fairness_score_aod = float(
            fairness_metrics[
                "fairness_score_aod"
            ]
        )

        fairness_score_dir = float(
            fairness_metrics[
                "fairness_score_dir"
            ]
        )

        # --------------------------------------------
        # RGA
        # --------------------------------------------
        _, aurga = compute_rga_curve(
            model=fold_model,
            X_test=X_val_processed,
            y_test=y_val,
        )

        # --------------------------------------------
        # RGR
        # --------------------------------------------
        rgr_columns = [
            c for c in numerical_features
            if c in X_val_processed.columns
        ]

        if not rgr_columns:
            rgr_columns = list(
                X_val_processed.columns
            )

        _, aurgr_gaussian = compute_rgr_curve(
            model=fold_model,
            X_test=X_val_processed,
            perturbation_type="gaussian",
            columns=rgr_columns,
            random_state=RANDOM_STATE + fold_number,
        )

        _, aurgr_swapping = compute_rgr_curve(
            model=fold_model,
            X_test=X_val_processed,
            perturbation_type="swapping",
            columns=rgr_columns,
            random_state=RANDOM_STATE + fold_number,
        )

        reference_rgr = compute_reference_rgr(
            model=fold_model,
            X_test=X_val_processed,
            intensity=0.5,
            random_state=RANDOM_STATE + fold_number,
        )

        rgr_aggregate = float(reference_rgr)

        # --------------------------------------------
        # RGE
        # --------------------------------------------
        rge_importance_df = (
            compute_rge_feature_importance(
                model=fold_model,
                X_test=X_val_processed,
            )
        )

        _, aurge = compute_rge_curve(
            model=fold_model,
            X_test=X_val_processed,
            importance_df=rge_importance_df,
        )

        # --------------------------------------------
        # SAFE SCORE
        # --------------------------------------------
        safe_score = _compute_safe_score(
            aurga=aurga,
            rgr_aggregate=rgr_aggregate,
            aurge=aurge,
            fairness_aggregate=fairness_aggregate,
        )

        fold_rows.append({
            "fold": fold_number,
            "n_train": int(len(train_idx)),
            "n_validation": int(len(val_idx)),
            "auc": auc_score,
            "aurga": float(aurga),
            "aurgr_gaussian": float(
                aurgr_gaussian
            ),
            "aurgr_swapping": float(
                aurgr_swapping
            ),
            "reference_rgr": float(
                reference_rgr
            ),
            "rgr_aggregate": rgr_aggregate,
            "aurge": float(aurge),
            "fairness_score_spd": fairness_score_spd,
            "fairness_score_eod": fairness_score_eod,
            "fairness_score_aod": fairness_score_aod,
            "fairness_score_dir": fairness_score_dir,
            "fairness_aggregate": fairness_aggregate,
            "safe_score": safe_score,
        })

    fold_df = pd.DataFrame(
        fold_rows
    )

    # --------------------------------------------
    # SUMMARY
    # --------------------------------------------
    metrics = [
        "auc",
        "aurga",
        "reference_rgr",
        "rgr_aggregate",
        "aurge",
        "fairness_aggregate",
        "safe_score",
    ]

    summary_rows = []

    for metric in metrics:
        stats = _mean_sd_ci(
            fold_df[metric].values,
            confidence_level=0.95,
        )

        summary_rows.append({
            "metric": metric,
            "mean": stats["mean"],
            "std": stats["std"],
            "ci_lower": stats["ci_lower"],
            "ci_upper": stats["ci_upper"],
            "n_folds": stats["n_folds"],
            "mean_sd": (
                f"{stats['mean']:.4f} "
                f"({stats['std']:.4f})"
            ),
            "mean_95ci": (
                f"{stats['mean']:.4f} "
                f"[{stats['ci_lower']:.4f}, "
                f"{stats['ci_upper']:.4f}]"
            ),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    return fold_df, summary_df


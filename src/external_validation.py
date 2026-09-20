# src/external_validation.py

import numpy as np
import pandas as pd

from scipy.stats import (
    t,
    cramervonmises_2samp,
)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import (
    RANDOM_STATE,
    PRED_THRESHOLD,
    W_RGA,
    W_RGR,
    W_RGE,
    W_FAIR,
)

from src.paths import (
    TAIWAN_DATA_PATH,
    TAIWAN_EXTERNAL_FOLDS_CSV_PATH,
    TAIWAN_EXTERNAL_SUMMARY_CSV_PATH,
    TAIWAN_EXTERNAL_REPORT_PATH,
)

from src.data_loader import (
    _build_one_hot_encoder,
    _clean_feature_names,
)

from src.model import build_xgboost

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


TAIWAN_TARGET_COLUMN = "default payment next month"
TAIWAN_SENSITIVE_COLUMN = "SEX"


# ------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------

def load_taiwan_credit_data():
    """
    Load and standardize the Taiwan Default of Credit Card Clients dataset.
    """

    if not TAIWAN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Taiwan dataset not found at: {TAIWAN_DATA_PATH}"
        )

    df = pd.read_excel(
        TAIWAN_DATA_PATH,
        header=1,
    )

    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    if TAIWAN_TARGET_COLUMN not in df.columns:
        raise ValueError(
            "Taiwan target column was not found.\n"
            f"Expected: {TAIWAN_TARGET_COLUMN}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    if TAIWAN_SENSITIVE_COLUMN not in df.columns:
        raise ValueError(
            "Taiwan sensitive column SEX was not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.rename(
        columns={
            TAIWAN_TARGET_COLUMN: "CreditRisk"
        }
    )

    df["CreditRisk"] = (
        pd.to_numeric(
            df["CreditRisk"],
            errors="coerce",
        )
        .astype("Int64")
    )

    df["SEX"] = (
        pd.to_numeric(
            df["SEX"],
            errors="coerce",
        )
        .map({
            1: "male",
            2: "female",
        })
        .fillna("unknown")
        .astype(str)
    )

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    df = df.dropna(
        subset=["CreditRisk"]
    ).copy()

    df["CreditRisk"] = (
        df["CreditRisk"]
        .astype(int)
    )

    return df


# ------------------------------------------------------------
# Fold preprocessing
# ------------------------------------------------------------

def _build_taiwan_preprocessor(X_train):
    """
    Build preprocessing for one CV fold.

    SEX is excluded from predictive features and used only
    as the sensitive attribute for fairness evaluation.
    """

    numerical_features = X_train.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    model_categorical_features = [
        c for c in categorical_features
        if c != TAIWAN_SENSITIVE_COLUMN
    ]

    model_numerical_features = [
        c for c in numerical_features
        if c != TAIWAN_SENSITIVE_COLUMN
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


def _process_taiwan_fold(
    X_train_raw,
    X_val_raw,
):
    (
        preprocessor,
        categorical_features,
        numerical_features,
    ) = _build_taiwan_preprocessor(
        X_train_raw
    )

    X_train_model = X_train_raw.drop(
        columns=[TAIWAN_SENSITIVE_COLUMN],
        errors="ignore",
    )

    X_val_model = X_val_raw.drop(
        columns=[TAIWAN_SENSITIVE_COLUMN],
        errors="ignore",
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
            .get_feature_names_out(
                categorical_features
            )
            .tolist()
        )

    raw_feature_names.extend(
        numerical_features
    )

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

# ------------------------------------------------------------
# Calibration
# ------------------------------------------------------------

def _compute_calibration_metrics(
    y_true,
    y_probs,
):
    """
    Compute calibration intercept, calibration slope,
    and Brier score.

    Ideal calibration:
        intercept = 0
        slope = 1

    Lower Brier score indicates better probabilistic accuracy.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_probs = np.asarray(
        y_probs,
        dtype=float,
    )

    # Avoid infinite logits for probabilities exactly 0 or 1
    eps = 1e-6

    clipped_probs = np.clip(
        y_probs,
        eps,
        1.0 - eps,
    )

    logits = np.log(
        clipped_probs
        / (1.0 - clipped_probs)
    ).reshape(-1, 1)

    calibration_model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000,
    )

    calibration_model.fit(
        logits,
        y_true,
    )

    calibration_intercept = float(
        calibration_model.intercept_[0]
    )

    calibration_slope = float(
        calibration_model.coef_[0][0]
    )

    brier = float(
        brier_score_loss(
            y_true,
            y_probs,
        )
    )

    return {
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "brier_score": brier,
    }

# ------------------------------------------------------------
# SAFE score
# ------------------------------------------------------------

def _compute_external_safe_score(
    aurga,
    rgr_aggregate,
    aurge,
    fairness_aggregate,
):
    return float(
        W_RGA * aurga
        + W_RGR * rgr_aggregate
        + W_RGE * aurge
        + W_FAIR * fairness_aggregate
    )

# ------------------------------------------------------------
# Cramer-von Mises statistical test
# ------------------------------------------------------------

def _compute_cvm_test(
    y_true,
    y_probs,
):
    """
    Compare the distributions of predicted probabilities
    between the non-default and default classes.

    H0:
        The predicted-probability distributions for CreditRisk=0
        and CreditRisk=1 are the same.

    A small p-value indicates statistically significant
    distributional separation between the two outcome groups.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_probs = np.asarray(
        y_probs,
        dtype=float,
    )

    probs_class_0 = y_probs[
        y_true == 0
    ]

    probs_class_1 = y_probs[
        y_true == 1
    ]

    if (
        len(probs_class_0) == 0
        or len(probs_class_1) == 0
    ):
        return {
            "cvm_statistic": np.nan,
            "cvm_p_value": np.nan,
        }

    result = cramervonmises_2samp(
        probs_class_0,
        probs_class_1,
        method="auto",
    )

    return {
        "cvm_statistic": float(
            result.statistic
        ),
        "cvm_p_value": float(
            result.pvalue
        ),
    }

# ------------------------------------------------------------
# Mean, SD and 95% CI
# ------------------------------------------------------------

def _mean_sd_ci(
    values,
    confidence_level=0.95,
):
    values = np.asarray(
        values,
        dtype=float,
    )

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

    mean = float(
        np.mean(values)
    )

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

    standard_error = (
        std / np.sqrt(n)
    )

    alpha = (
        1.0 - confidence_level
    )

    critical_value = float(
        t.ppf(
            1.0 - alpha / 2.0,
            df=n - 1,
        )
    )

    margin = (
        critical_value
        * standard_error
    )

    return {
        "mean": mean,
        "std": std,
        "ci_lower": float(
            mean - margin
        ),
        "ci_upper": float(
            mean + margin
        ),
        "n_folds": int(n),
    }


# ------------------------------------------------------------
# External validation
# ------------------------------------------------------------

def run_taiwan_external_validation():
    """
    Replicate the SAFE governance framework on the independent
    Taiwan credit dataset using stratified 5-fold cross-validation.
    """

    df = load_taiwan_credit_data()

    X = df.drop(
        columns=["CreditRisk"]
    )

    y = df[
        "CreditRisk"
    ].astype(int)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows = []

    print("\n" + "=" * 70)
    print("TAIWAN EXTERNAL DATASET VALIDATION")
    print("=" * 70)

    for fold_number, (
        train_idx,
        val_idx,
    ) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        print(
            f"\n[TAIWAN] Starting fold "
            f"{fold_number}/5..."
        )

        X_train_raw = (
            X.iloc[train_idx]
            .copy()
        )

        X_val_raw = (
            X.iloc[val_idx]
            .copy()
        )

        y_train = (
            y.iloc[train_idx]
            .copy()
            .reset_index(drop=True)
        )

        y_val = (
            y.iloc[val_idx]
            .copy()
            .reset_index(drop=True)
        )

        group_val = (
            X_val_raw[
                TAIWAN_SENSITIVE_COLUMN
            ]
            .astype(str)
            .fillna("unknown")
            .reset_index(drop=True)
        )

        (
            X_train_processed,
            X_val_processed,
            numerical_features,
        ) = _process_taiwan_fold(
            X_train_raw=X_train_raw,
            X_val_raw=X_val_raw,
        )

        X_train_processed = (
            X_train_processed
            .reset_index(drop=True)
        )

        X_val_processed = (
            X_val_processed
            .reset_index(drop=True)
        )

        # Same governance model family used in the main SAFE pipeline.
        model = build_xgboost()

        model.fit(
            X_train_processed,
            y_train,
        )

        y_probs = model.predict_proba(
            X_val_processed
        )[:, 1]

        # ----------------------------------------------------
        # Predictive performance
        # ----------------------------------------------------

        auc_score = float(
            roc_auc_score(
                y_val,
                y_probs,
            )
        )

        calibration_metrics = (
            _compute_calibration_metrics(
                y_true=y_val,
                y_probs=y_probs,
            )
        )

        calibration_intercept = (
            calibration_metrics[
                "calibration_intercept"
            ]
        )

        calibration_slope = (
            calibration_metrics[
                "calibration_slope"
            ]
        )

        brier_score = (
            calibration_metrics[
                "brier_score"
            ]
        )

        # ----------------------------------------------------
        # Cramer-von Mises statistical test
        # ----------------------------------------------------

        cvm_results = _compute_cvm_test(
            y_true=y_val,
            y_probs=y_probs,
        )

        cvm_statistic = (
            cvm_results[
                "cvm_statistic"
            ]
        )

        cvm_p_value = (
            cvm_results[
                "cvm_p_value"
            ]
        )

        # ----------------------------------------------------
        # Fairness
        # ----------------------------------------------------

        (
            fairness_metrics,
            _,
            _,
        ) = _compute_fairness_metrics(
            y_true=y_val,
            y_probs=y_probs,
            group=group_val,
            pred_threshold=PRED_THRESHOLD,
        )

        fairness_aggregate = float(
            fairness_metrics[
                "fairness_aggregate"
            ]
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

        # ----------------------------------------------------
        # RGA
        # ----------------------------------------------------

        _, aurga = compute_rga_curve(
            model=model,
            X_test=X_val_processed,
            y_test=y_val,
        )

        # ----------------------------------------------------
        # RGR
        # ----------------------------------------------------

        rgr_columns = [
            c
            for c in numerical_features
            if c in X_val_processed.columns
        ]

        if not rgr_columns:
            rgr_columns = list(
                X_val_processed.columns
            )

        (
            _,
            aurgr_gaussian,
        ) = compute_rgr_curve(
            model=model,
            X_test=X_val_processed,
            perturbation_type="gaussian",
            columns=rgr_columns,
            random_state=(
                RANDOM_STATE
                + fold_number
            ),
        )

        (
            _,
            aurgr_swapping,
        ) = compute_rgr_curve(
            model=model,
            X_test=X_val_processed,
            perturbation_type="swapping",
            columns=rgr_columns,
            random_state=(
                RANDOM_STATE
                + fold_number
            ),
        )

        reference_rgr = compute_reference_rgr(
            model=model,
            X_test=X_val_processed,
            intensity=0.5,
            random_state=RANDOM_STATE + fold_number,
        )

        rgr_aggregate = float(reference_rgr)

        # ----------------------------------------------------
        # RGE
        # ----------------------------------------------------

        rge_importance_df = (
            compute_rge_feature_importance(
                model=model,
                X_test=X_val_processed,
            )
        )

        _, aurge = compute_rge_curve(
            model=model,
            X_test=X_val_processed,
            importance_df=rge_importance_df,
        )

        # ----------------------------------------------------
        # SAFE
        # ----------------------------------------------------

        safe_score = (
            _compute_external_safe_score(
                aurga=aurga,
                rgr_aggregate=rgr_aggregate,
                aurge=aurge,
                fairness_aggregate=(
                    fairness_aggregate
                ),
            )
        )

        fold_rows.append({
            "fold": fold_number,
            "n_train": int(
                len(train_idx)
            ),
            "n_validation": int(
                len(val_idx)
            ),
            "auc": auc_score,
            "calibration_intercept": calibration_intercept,
            "calibration_slope": calibration_slope,
            "brier_score": brier_score,
            "cvm_statistic": cvm_statistic,
            "cvm_p_value": cvm_p_value,
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
            "rgr_aggregate": (
                rgr_aggregate
            ),
            "aurge": float(aurge),

            "fairness_score_spd": fairness_score_spd,
            "fairness_score_eod": fairness_score_eod,
            "fairness_score_aod": fairness_score_aod,
            "fairness_score_dir": fairness_score_dir,

            "fairness_aggregate": fairness_aggregate,
            "safe_score": safe_score,
        })

        print(
            f"[TAIWAN] Fold "
            f"{fold_number} completed | "
            f"AUC={auc_score:.4f} | "
            f"Cal.Int={calibration_intercept:.4f} | "
            f"Cal.Slope={calibration_slope:.4f} | "
            f"Brier={brier_score:.4f} | "
            f"CvM={cvm_statistic:.4f} | "
            f"p={cvm_p_value:.6g} | "
            f"SAFE={safe_score:.4f}"
        )

    # --------------------------------------------------------
    # Fold dataframe
    # --------------------------------------------------------

    fold_df = pd.DataFrame(
        fold_rows
    )

    metrics = [
        "auc",
        "calibration_intercept",
        "calibration_slope",
        "brier_score",
        "cvm_statistic",
        "aurga",
        "rgr_aggregate",
        "reference_rgr",
        "aurge",
        "fairness_aggregate",
        "safe_score",
    ]

    summary_rows = []

    for metric in metrics:

        stats = _mean_sd_ci(
            fold_df[
                metric
            ].values,
            confidence_level=0.95,
        )

        summary_rows.append({
            "metric": metric,
            "mean": stats["mean"],
            "std": stats["std"],
            "ci_lower": stats[
                "ci_lower"
            ],
            "ci_upper": stats[
                "ci_upper"
            ],
            "n_folds": stats[
                "n_folds"
            ],
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
    # CvM p-values are reported fold by fold rather than averaged.
    cvm_significant_folds = int(
        (
            fold_df["cvm_p_value"]
            < 0.05
        ).sum()
    )

    cvm_min_p = float(
        fold_df[
            "cvm_p_value"
        ].min()
    )

    cvm_max_p = float(
        fold_df[
            "cvm_p_value"
        ].max()
    )
    summary_df = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # Save CSV artifacts
    # --------------------------------------------------------

    fold_df.to_csv(
        TAIWAN_EXTERNAL_FOLDS_CSV_PATH,
        index=False,
    )

    summary_df.to_csv(
        TAIWAN_EXTERNAL_SUMMARY_CSV_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Markdown report
    # --------------------------------------------------------

    report_content = f"""
# Taiwan External Dataset Validation

The SAFE governance framework was independently replicated on the
Taiwan Default of Credit Card Clients dataset.

Dataset size: {len(df)}

Sensitive attribute: SEX

Target: CreditRisk

Validation design:
- Stratified 5-fold cross-validation
- Preprocessing fitted independently within each training fold
- Sensitive attribute excluded from predictive model inputs
- XGBoost used as the governance model family
- SAFE score calculated from RGA, RGR, RGE, and Fairness

## Fold-Level Results

{fold_df.to_markdown(index=False)}

## Five-Fold Summary

## Cramer-von Mises Distributional Test

The two-sample Cramer-von Mises test was used within each validation
fold to compare the distribution of predicted default probabilities
between observations with CreditRisk = 0 and CreditRisk = 1.

Null hypothesis:
the two predicted-probability distributions are identical.

Significant folds at alpha = 0.05:
{cvm_significant_folds} out of 5.

Minimum fold p-value:
{cvm_min_p:.6g}

Maximum fold p-value:
{cvm_max_p:.6g}

Fold-level p-values are reported individually and are not averaged.

{summary_df.to_markdown(index=False)}

Values in `mean_sd` are reported as mean (standard deviation).

The `mean_95ci` column reports fold-level 95% confidence intervals.
"""

    TAIWAN_EXTERNAL_REPORT_PATH.write_text(
        report_content,
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("TAIWAN VALIDATION COMPLETED")
    print("=" * 70)

    print(
        f"Fold results: "
        f"{TAIWAN_EXTERNAL_FOLDS_CSV_PATH}"
    )

    print(
        f"Summary: "
        f"{TAIWAN_EXTERNAL_SUMMARY_CSV_PATH}"
    )

    print(
        f"Report: "
        f"{TAIWAN_EXTERNAL_REPORT_PATH}"
    )

    print("\nSummary:")
    print(
        summary_df.to_string(
            index=False
        )
    )

    return (
        fold_df,
        summary_df,
    )


# ------------------------------------------------------------
# Dataset diagnostic
# ------------------------------------------------------------

def print_taiwan_dataset_summary():

    df = load_taiwan_credit_data()

    print("\n" + "=" * 60)
    print("TAIWAN CREDIT DATASET")
    print("=" * 60)

    print(
        f"Dataset path: "
        f"{TAIWAN_DATA_PATH}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: "
        f"{len(df.columns)}"
    )

    print("\nTarget distribution:")

    print(
        df["CreditRisk"]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print(
        "\nSensitive attribute "
        "distribution:"
    )

    print(
        df["SEX"]
        .value_counts(
            dropna=False
        )
    )

    return df


if __name__ == "__main__":
    run_taiwan_external_validation()
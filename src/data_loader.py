# src/data_loader.py

import json

import openml
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crewai.tools import tool

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import (
    DATA_PATH,
    OPENML_ID,
    RANDOM_STATE,
    SENSITIVE_FEATURE,
    DROP_SENSITIVE_FROM_MODEL,
    current_config,
)

from src.paths import (
    RAW_DATA_PATH,
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
    TEST_FEATURES_PATH,
    TEST_TARGET_PATH,
    SENSITIVE_TRAIN_PATH,
    SENSITIVE_TEST_PATH,
    DATACARD_PATH,
    EDA_TARGET_DISTRIBUTION_PLOT_PATH,
    EDA_NUMERIC_DISTRIBUTION_PLOT_PATH,
    EDA_MISSING_VALUES_PLOT_PATH,
    OUTLIER_REPORT_PATH,
    CLEAN_TRAIN_FULL_PATH,
    CLEAN_TEST_FULL_PATH,
)


def get_credit_data():
    """Load the credit dataset from local CSV or OpenML and save it as raw data."""
    try:
        if DATA_PATH:
            df = pd.read_csv(DATA_PATH)
            df.to_csv(RAW_DATA_PATH, index=False)
            return str(RAW_DATA_PATH)

        dataset = openml.datasets.get_dataset(OPENML_ID)
        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute,
        )

        y = y.apply(lambda x: 1 if x == "bad" else 0)
        data = pd.concat([X, y.rename("CreditRisk")], axis=1)

        data.to_csv(RAW_DATA_PATH, index=False)
        return str(RAW_DATA_PATH)

    except Exception as e:
        return f"Error loading data: {e}"


def _save_eda_artifacts(df, target_col="CreditRisk"):
    """Save target, numeric-distribution, and missing-value EDA plots."""
    if target_col in df.columns:
        plt.figure(figsize=(6, 4))
        df[target_col].value_counts().sort_index().plot(kind="bar")
        plt.title("Target Distribution")
        plt.xlabel(target_col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(EDA_TARGET_DISTRIBUTION_PLOT_PATH, dpi=200)
        plt.close()

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    if numeric_cols:
        df[numeric_cols[:8]].hist(figsize=(12, 8), bins=20)
        plt.suptitle("Numeric Feature Distributions")
        plt.tight_layout()
        plt.savefig(EDA_NUMERIC_DISTRIBUTION_PLOT_PATH, dpi=200)
        plt.close()

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    plt.figure(figsize=(8, 4))
    if len(missing) > 0:
        missing.plot(kind="bar")
        plt.ylabel("Missing values")
        plt.title("Missing Values by Column")
    else:
        plt.text(0.5, 0.5, "No missing values detected", ha="center", va="center")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(EDA_MISSING_VALUES_PLOT_PATH, dpi=200)
    plt.close()


def _save_outlier_report(df, target_col="CreditRisk"):
    """Save an IQR-based outlier analysis report."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    rows = []

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_count = int(((df[col] < lower) | (df[col] > upper)).sum())

        rows.append({
            "feature": col,
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "outlier_count": outlier_count,
            "outlier_rate": float(outlier_count / len(df)),
        })

    outlier_df = pd.DataFrame(rows)

    with open(OUTLIER_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Outlier Analysis Report\n\n")
        f.write("Outliers are detected using the IQR rule.\n\n")

        if outlier_df.empty:
            f.write("No numeric columns were available for outlier analysis.\n")
        else:
            f.write(
                outlier_df
                .sort_values("outlier_count", ascending=False)
                .to_markdown(index=False)
            )
            f.write("\n")


def _build_one_hot_encoder():
    """Create a OneHotEncoder compatible with multiple sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _clean_feature_names(feature_names):
    """Clean feature names for model compatibility and reporting."""
    return [
        name.replace("[", "")
        .replace("]", "")
        .replace("<", "less_than_")
        for name in feature_names
    ]


def _save_sensitive_feature(X_train, X_test):
    """Save the sensitive feature separately for fairness analysis."""
    if SENSITIVE_FEATURE in X_train.columns:
        X_train[[SENSITIVE_FEATURE]].to_csv(SENSITIVE_TRAIN_PATH, index=False)
        X_test[[SENSITIVE_FEATURE]].to_csv(SENSITIVE_TEST_PATH, index=False)
    else:
        pd.DataFrame({"group": ["UNKNOWN"] * len(X_train)}).to_csv(SENSITIVE_TRAIN_PATH, index=False)
        pd.DataFrame({"group": ["UNKNOWN"] * len(X_test)}).to_csv(SENSITIVE_TEST_PATH, index=False)


def _save_processed_data(
    train_features_df,
    train_target_df,
    test_features_df,
    test_target_df,
):
    """Save separate and combined processed train/test artifacts."""
    train_features_df.to_csv(TRAIN_FEATURES_PATH, index=False)
    train_target_df.to_csv(TRAIN_TARGET_PATH, index=False)

    test_features_df.to_csv(TEST_FEATURES_PATH, index=False)
    test_target_df.to_csv(TEST_TARGET_PATH, index=False)

    pd.concat(
        [train_features_df.reset_index(drop=True), train_target_df.reset_index(drop=True)],
        axis=1,
    ).to_csv(CLEAN_TRAIN_FULL_PATH, index=False)

    pd.concat(
        [test_features_df.reset_index(drop=True), test_target_df.reset_index(drop=True)],
        axis=1,
    ).to_csv(CLEAN_TEST_FULL_PATH, index=False)


def _save_datacard(
    df,
    X_train,
    X_test,
    clean_feature_names,
    model_numerical_features,
    model_categorical_features,
):
    """Save the Data Card for reproducibility and auditability."""
    datacard = {
        "status": "CLEANED",
        "raw_rows": int(len(df)),
        "raw_columns": int(len(df.columns)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features_after_encoding": int(len(clean_feature_names)),
        "numeric_features_raw": list(map(str, model_numerical_features)),
        "categorical_features_raw": list(map(str, model_categorical_features)),
        "sensitive_feature": SENSITIVE_FEATURE,
        "drop_sensitive_from_model": bool(DROP_SENSITIVE_FROM_MODEL),
        "config": current_config(),
        "eda_artifacts": {
            "target_distribution": str(EDA_TARGET_DISTRIBUTION_PLOT_PATH.name),
            "numeric_distributions": str(EDA_NUMERIC_DISTRIBUTION_PLOT_PATH.name),
            "missing_values": str(EDA_MISSING_VALUES_PLOT_PATH.name),
            "outlier_report": str(OUTLIER_REPORT_PATH.name),
        },
        "clean_artifacts": {
            "clean_train_full": str(CLEAN_TRAIN_FULL_PATH.name),
            "clean_test_full": str(CLEAN_TEST_FULL_PATH.name),
        },
    }

    with open(DATACARD_PATH, "w", encoding="utf-8") as f:
        json.dump(datacard, f, indent=2)


@tool
def data_preprocessing_tool(file_path: str):
    """Create ML-ready train/test artifacts and the Data Card."""
    try:
        df = pd.read_csv(file_path)

        _save_eda_artifacts(df, target_col="CreditRisk")
        _save_outlier_report(df, target_col="CreditRisk")

        X = df.drop("CreditRisk", axis=1)
        y = df["CreditRisk"]

        numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        model_categorical_features = [
            c for c in categorical_features
            if not (DROP_SENSITIVE_FROM_MODEL and c == SENSITIVE_FEATURE)
        ]

        model_numerical_features = [
            c for c in numerical_features
            if not (DROP_SENSITIVE_FROM_MODEL and c == SENSITIVE_FEATURE)
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", _build_one_hot_encoder(), model_categorical_features),
                ("num", StandardScaler(), model_numerical_features),
            ],
            remainder="drop",
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        _save_sensitive_feature(X_train, X_test)

        if DROP_SENSITIVE_FROM_MODEL and SENSITIVE_FEATURE in X_train.columns:
            X_train = X_train.drop(columns=[SENSITIVE_FEATURE])
            X_test = X_test.drop(columns=[SENSITIVE_FEATURE])

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        raw_feature_names = (
            list(preprocessor.named_transformers_["cat"].get_feature_names_out(model_categorical_features))
            + list(model_numerical_features)
        )

        clean_feature_names = _clean_feature_names(raw_feature_names)

        train_features_df = pd.DataFrame(X_train_processed, columns=clean_feature_names)
        test_features_df = pd.DataFrame(X_test_processed, columns=clean_feature_names)

        train_target_df = pd.DataFrame({"CreditRisk": y_train.values})
        test_target_df = pd.DataFrame({"CreditRisk": y_test.values})

        _save_processed_data(
            train_features_df,
            train_target_df,
            test_features_df,
            test_target_df,
        )

        _save_datacard(
            df=df,
            X_train=X_train,
            X_test=X_test,
            clean_feature_names=clean_feature_names,
            model_numerical_features=model_numerical_features,
            model_categorical_features=model_categorical_features,
        )

        return "Data successfully processed, feature names cleaned for XGBoost, and datasets saved."

    except Exception as e:
        return f"DATA PREPROCESSING FAILED: {e}"
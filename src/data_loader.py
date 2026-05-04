# src/data_loader.py

# Standard library import for writing the Data Card as JSON
import json

# OpenML is used as an optional external source for the credit dataset
import openml

import numpy as np

# Use a non-GUI backend for saving plots during automated pipeline runs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pandas is used for reading, transforming, and saving tabular data
import pandas as pd

# CrewAI tool decorator so preprocessing can be used by the Data Agent
from crewai.tools import tool

# Scikit-learn preprocessing utilities:
# - ColumnTransformer: apply different preprocessing to different column types
# - train_test_split: create train/test datasets
# - StandardScaler: scale numerical features
# - OneHotEncoder: encode categorical features into numeric form
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Project configuration values used during data loading and preprocessing
from src.config import (
    DATA_PATH,
    OPENML_ID,
    RANDOM_STATE,
    SENSITIVE_FEATURE,
    DROP_SENSITIVE_FROM_MODEL,
    current_config,
)

# Centralized file paths for storing raw, processed, sensitive, and documentation artifacts
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
    """
    Load the raw credit dataset and save it to the project's raw-data path.

    Behavior:
    - If DATA_PATH is provided in the configuration, use the local CSV file.
    - Otherwise, download the dataset from OpenML using OPENML_ID.
    - In the OpenML case, convert the target to a binary CreditRisk label.
    
    Returns:
        str: path to the saved raw dataset, or an error string if loading fails.
    """
    try:
        # If a local dataset path is provided, use that file directly
        if DATA_PATH:
            df = pd.read_csv(DATA_PATH)
            df.to_csv(RAW_DATA_PATH, index=False)
            return str(RAW_DATA_PATH)

        # Otherwise, fetch the dataset from OpenML
        dataset = openml.datasets.get_dataset(OPENML_ID)
        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute
        )

        # Convert the OpenML target into a binary credit-risk label
        # Here: "bad" -> 1, everything else -> 0
        y = y.apply(lambda x: 1 if x == "bad" else 0)

        # Combine input features and target into one raw dataset
        data = pd.concat([X, y.rename("CreditRisk")], axis=1)

        # Save the raw dataset into the project structure
        data.to_csv(RAW_DATA_PATH, index=False)
        return str(RAW_DATA_PATH)

    except Exception as e:
        return f"Error loading data: {e}"

def _save_eda_artifacts(df, target_col="CreditRisk"):
    """
    Save basic EDA figures:
    - target distribution
    - numeric feature distributions
    - missing-value counts
    """
    # Target distribution plot
    if target_col in df.columns:
        plt.figure(figsize=(6, 4))
        df[target_col].value_counts().sort_index().plot(kind="bar")
        plt.title("Target Distribution")
        plt.xlabel(target_col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(EDA_TARGET_DISTRIBUTION_PLOT_PATH, dpi=200)
        plt.close()

    # Numeric feature distributions
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    if numeric_cols:
        sample_cols = numeric_cols[:8]
        axes = df[sample_cols].hist(figsize=(12, 8), bins=20)
        plt.suptitle("Numeric Feature Distributions")
        plt.tight_layout()
        plt.savefig(EDA_NUMERIC_DISTRIBUTION_PLOT_PATH, dpi=200)
        plt.close()

    # Missing-value plot
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
    """
    Save a simple outlier analysis report using the IQR rule.
    """
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
            f.write(outlier_df.sort_values("outlier_count", ascending=False).to_markdown(index=False))
            f.write("\n")

@tool
def data_preprocessing_tool(file_path: str):
    """
    Process the raw credit dataset into machine-learning-ready artifacts.

    Main steps:
    - read the raw dataset
    - split features and target
    - identify numeric and categorical columns
    - handle the sensitive feature for fairness analysis
    - split into train/test sets
    - one-hot encode categorical features
    - scale numerical features
    - save processed train/test CSV files
    - save the Data Card artifact

    Args:
        file_path (str): path to the raw input CSV

    Returns:
        str: success or failure message
    """
    try:
        # Read the raw input dataset
        df = pd.read_csv(file_path)

        # Save EDA and outlier artifacts before preprocessing.
        _save_eda_artifacts(df, target_col="CreditRisk")
        _save_outlier_report(df, target_col="CreditRisk")

        # Separate features (X) and target label (y)
        X = df.drop("CreditRisk", axis=1)
        y = df["CreditRisk"]

        # Detect numerical and categorical feature columns
        numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        # Build the feature lists that will actually be used by the model.
        # If DROP_SENSITIVE_FROM_MODEL is enabled, the selected sensitive feature
        # is excluded from the model input.
        model_categorical_features = [
            c for c in categorical_features
            if not (DROP_SENSITIVE_FROM_MODEL and c == SENSITIVE_FEATURE)
        ]
        model_numerical_features = [
            c for c in numerical_features
            if not (DROP_SENSITIVE_FROM_MODEL and c == SENSITIVE_FEATURE)
        ]

        # Create a OneHotEncoder compatible with different sklearn versions
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        # Build the preprocessing pipeline:
        # - categorical columns -> one-hot encoding
        # - numerical columns -> standard scaling
        # Any remaining columns are dropped.
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", ohe, model_categorical_features),
                ("num", StandardScaler(), model_numerical_features),
            ],
            remainder="drop"
        )

        # Create train/test split with a fixed random seed for reproducibility.
        # stratify=y keeps class balance more stable across train and test.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        # Save the sensitive feature separately for fairness analysis.
        # If the configured sensitive feature is missing, create a placeholder group.
        if SENSITIVE_FEATURE in X_train.columns:
            X_train[[SENSITIVE_FEATURE]].to_csv(SENSITIVE_TRAIN_PATH, index=False)
            X_test[[SENSITIVE_FEATURE]].to_csv(SENSITIVE_TEST_PATH, index=False)
        else:
            pd.DataFrame({"group": ["UNKNOWN"] * len(X_train)}).to_csv(SENSITIVE_TRAIN_PATH, index=False)
            pd.DataFrame({"group": ["UNKNOWN"] * len(X_test)}).to_csv(SENSITIVE_TEST_PATH, index=False)

        # If configured, remove the sensitive feature from model inputs
        # after it has already been saved separately for fairness analysis.
        if DROP_SENSITIVE_FROM_MODEL and (SENSITIVE_FEATURE in X_train.columns):
            X_train = X_train.drop(columns=[SENSITIVE_FEATURE])
            X_test = X_test.drop(columns=[SENSITIVE_FEATURE])

        # Fit preprocessing on the training set and apply the same transformation to test data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Recover processed feature names from the fitted transformer:
        # one-hot encoded categorical names + numerical column names
        raw_feature_names = (
            list(preprocessor.named_transformers_["cat"].get_feature_names_out(model_categorical_features))
            + list(model_numerical_features)
        )

        # Clean feature names so they are safer for downstream model use and reporting
        clean_feature_names = [
            name.replace("[", "").replace("]", "").replace("<", "less_than_")
            for name in raw_feature_names
        ]

        # Save separate feature/target files.
        train_features_df = pd.DataFrame(X_train_processed, columns=clean_feature_names)
        test_features_df = pd.DataFrame(X_test_processed, columns=clean_feature_names)

        train_target_df = pd.DataFrame({"CreditRisk": y_train.values})
        test_target_df = pd.DataFrame({"CreditRisk": y_test.values})

        train_features_df.to_csv(TRAIN_FEATURES_PATH, index=False)
        train_target_df.to_csv(TRAIN_TARGET_PATH, index=False)

        test_features_df.to_csv(TEST_FEATURES_PATH, index=False)
        test_target_df.to_csv(TEST_TARGET_PATH, index=False)

        # Save combined clean train/test artifacts for easier inspection and reporting.
        pd.concat([train_features_df.reset_index(drop=True), train_target_df.reset_index(drop=True)], axis=1).to_csv(
            CLEAN_TRAIN_FULL_PATH,
            index=False
        )

        pd.concat([test_features_df.reset_index(drop=True), test_target_df.reset_index(drop=True)], axis=1).to_csv(
            CLEAN_TEST_FULL_PATH,
            index=False
        )
        # Build the Data Card:
        # a compact summary of preprocessing outputs and key configuration choices
        datacard = {
            "status": "CLEANED",
            "features_after_encoding": int(len(clean_feature_names)),
            "numeric_features_raw": list(map(str, model_numerical_features)),
            "categorical_features_raw": list(map(str, model_categorical_features)),
            "sensitive_feature": SENSITIVE_FEATURE,
            "drop_sensitive_from_model": bool(DROP_SENSITIVE_FROM_MODEL),
            "config": current_config(),
                        "raw_rows": int(len(df)),
            "raw_columns": int(len(df.columns)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
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

        # Save the Data Card artifact for auditing and reproducibility
        with open(DATACARD_PATH, "w", encoding="utf-8") as f:
            json.dump(datacard, f, indent=2)

        return "Data successfully processed, feature names cleaned for XGBoost, and datasets saved."

    except Exception as e:
        return f"DATA PREPROCESSING FAILED: {str(e)}"
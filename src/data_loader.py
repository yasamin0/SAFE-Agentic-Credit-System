import json
import openml
import pandas as pd
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
)


def get_credit_data():
    try:
        if DATA_PATH:
            df = pd.read_csv(DATA_PATH)
            df.to_csv(RAW_DATA_PATH, index=False)
            return str(RAW_DATA_PATH)

        dataset = openml.datasets.get_dataset(OPENML_ID)
        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute
        )

        y = y.apply(lambda x: 1 if x == "bad" else 0)

        data = pd.concat([X, y.rename("CreditRisk")], axis=1)
        data.to_csv(RAW_DATA_PATH, index=False)
        return str(RAW_DATA_PATH)

    except Exception as e:
        return f"Error loading data: {e}"


@tool
def data_preprocessing_tool(file_path: str):
    """Processes, cleans, encodes, scales, and splits the raw credit data. Saves results as clean CSV files."""
    try:
        df = pd.read_csv(file_path)

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

        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", ohe, model_categorical_features),
                ("num", StandardScaler(), model_numerical_features),
            ],
            remainder="drop"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        if SENSITIVE_FEATURE in X_train.columns:
            X_train[[SENSITIVE_FEATURE]].to_csv(SENSITIVE_TRAIN_PATH, index=False)
            X_test[[SENSITIVE_FEATURE]].to_csv(SENSITIVE_TEST_PATH, index=False)
        else:
            pd.DataFrame({"group": ["UNKNOWN"] * len(X_train)}).to_csv(SENSITIVE_TRAIN_PATH, index=False)
            pd.DataFrame({"group": ["UNKNOWN"] * len(X_test)}).to_csv(SENSITIVE_TEST_PATH, index=False)

        if DROP_SENSITIVE_FROM_MODEL and (SENSITIVE_FEATURE in X_train.columns):
            X_train = X_train.drop(columns=[SENSITIVE_FEATURE])
            X_test = X_test.drop(columns=[SENSITIVE_FEATURE])

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        raw_feature_names = (
            list(preprocessor.named_transformers_["cat"].get_feature_names_out(model_categorical_features))
            + list(model_numerical_features)
        )

        clean_feature_names = [
            name.replace("[", "").replace("]", "").replace("<", "less_than_")
            for name in raw_feature_names
        ]

        pd.DataFrame(X_train_processed, columns=clean_feature_names).to_csv(TRAIN_FEATURES_PATH, index=False)
        y_train.to_csv(TRAIN_TARGET_PATH, index=False, header=["CreditRisk"])
        pd.DataFrame(X_test_processed, columns=clean_feature_names).to_csv(TEST_FEATURES_PATH, index=False)
        y_test.to_csv(TEST_TARGET_PATH, index=False, header=["CreditRisk"])

        datacard = {
            "status": "CLEANED",
            "features_after_encoding": int(len(clean_feature_names)),
            "numeric_features_raw": list(map(str, model_numerical_features)),
            "categorical_features_raw": list(map(str, model_categorical_features)),
            "sensitive_feature": SENSITIVE_FEATURE,
            "drop_sensitive_from_model": bool(DROP_SENSITIVE_FROM_MODEL),
            "config": current_config(),
        }

        with open(DATACARD_PATH, "w", encoding="utf-8") as f:
            json.dump(datacard, f, indent=2)

        return "Data successfully processed, feature names cleaned for XGBoost, and datasets saved."

    except Exception as e:
        return f"DATA PREPROCESSING FAILED: {str(e)}"
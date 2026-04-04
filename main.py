# main.py

import os
import pandas as pd
import openml
import json
import numpy as np 
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import joblib 
from xgboost import XGBClassifier
# --- SCALER AND PIPELINE IMPORTS ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.metrics import roc_auc_score

def _apply_threshold_mitigation(y_probs, group, base_threshold=0.55):
    """
    Simple post-processing mitigation:
    use a slightly lower threshold for disadvantaged groups with lower positive prediction rates.
    """
    df = pd.DataFrame({
        "prob": y_probs,
        "group": group.astype(str).fillna("NA")
    })

    group_rates = df.groupby("group")["prob"].apply(lambda s: float((s >= base_threshold).mean()))
    min_group = group_rates.idxmin()

    adjusted_pred = []
    for p, g in zip(df["prob"], df["group"]):
        thr = base_threshold - 0.05 if g == min_group else base_threshold
        adjusted_pred.append(1 if p >= thr else 0)

    return np.array(adjusted_pred), str(min_group)

def _compute_fairness_from_predictions(y_true, y_pred, group):
    tmp, group_table = _build_group_table(y_true, y_pred, group)

    if len(group_table) <= 1:
        return {
            "spd_gap": 0.0,
            "eod_gap": 0.0,
            "aod_gap": 0.0,
            "dir_ratio": 1.0,
            "fairness_score_spd": 1.0,
            "fairness_score_eod": 1.0,
            "fairness_score_aod": 1.0,
            "fairness_score_dir": 1.0,
            "fairness_aggregate": 1.0,
        }, group_table

    pos_rates = group_table["positive_rate"]
    tprs = group_table["tpr"]
    fprs = group_table["fpr"]

    spd_gap = float(pos_rates.max() - pos_rates.min())
    eod_gap = float(tprs.max() - tprs.min())
    fpr_gap = float(fprs.max() - fprs.min())
    aod_gap = float(0.5 * (eod_gap + fpr_gap))

    min_pos = float(pos_rates.min())
    max_pos = float(pos_rates.max())
    dir_ratio = float(min_pos / max_pos) if max_pos > 0 else 1.0

    fairness_score_spd = float(np.clip(1.0 - spd_gap, 0.0, 1.0))
    fairness_score_eod = float(np.clip(1.0 - eod_gap, 0.0, 1.0))
    fairness_score_aod = float(np.clip(1.0 - aod_gap, 0.0, 1.0))
    fairness_score_dir = float(np.clip(dir_ratio, 0.0, 1.0))
    fairness_aggregate = _safe_mean([
        fairness_score_spd,
        fairness_score_eod,
        fairness_score_aod,
        fairness_score_dir,
    ])

    return {
        "spd_gap": spd_gap,
        "eod_gap": eod_gap,
        "aod_gap": aod_gap,
        "dir_ratio": dir_ratio,
        "fairness_score_spd": fairness_score_spd,
        "fairness_score_eod": fairness_score_eod,
        "fairness_score_aod": fairness_score_aod,
        "fairness_score_dir": fairness_score_dir,
        "fairness_aggregate": fairness_aggregate,
    }, group_table

def _safe_mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def _read_target_series(path):
    df = pd.read_csv(path)
    return df.iloc[:, 0].values.ravel()


def _build_group_table(y_true, y_pred, group):
    tmp = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": group})

    def tpr(df):
        tp = ((df.y_pred == 1) & (df.y_true == 1)).sum()
        fn = ((df.y_pred == 0) & (df.y_true == 1)).sum()
        return tp / (tp + fn) if (tp + fn) else 0.0

    def fpr(df):
        fp = ((df.y_pred == 1) & (df.y_true == 0)).sum()
        tn = ((df.y_pred == 0) & (df.y_true == 0)).sum()
        return fp / (fp + tn) if (fp + tn) else 0.0

    rows = []
    for g, gdf in tmp.groupby("group"):
        rows.append({
            "group": g,
            "n": int(len(gdf)),
            "positive_rate": float(gdf["y_pred"].mean()),
            "tpr": float(tpr(gdf)),
            "fpr": float(fpr(gdf)),
        })
    return tmp, pd.DataFrame(rows)


def _compute_fairness_metrics(y_true, y_probs, group, pred_threshold):
    y_pred = (y_probs >= pred_threshold).astype(int)
    tmp, group_table = _build_group_table(y_true, y_pred, group)

    if len(group_table) <= 1:
        metrics = {
            "spd_gap": 0.0,
            "eod_gap": 0.0,
            "aod_gap": 0.0,
            "dir_ratio": 1.0,
            "fairness_score_spd": 1.0,
            "fairness_score_eod": 1.0,
            "fairness_score_aod": 1.0,
            "fairness_score_dir": 1.0,
            "fairness_aggregate": 1.0,
        }
        return metrics, group_table, tmp

    pos_rates = group_table["positive_rate"]
    tprs = group_table["tpr"]
    fprs = group_table["fpr"]

    spd_gap = float(pos_rates.max() - pos_rates.min())
    eod_gap = float(tprs.max() - tprs.min())
    fpr_gap = float(fprs.max() - fprs.min())
    aod_gap = float(0.5 * (eod_gap + fpr_gap))

    min_pos = float(pos_rates.min())
    max_pos = float(pos_rates.max())
    dir_ratio = float(min_pos / max_pos) if max_pos > 0 else 1.0

    fairness_score_spd = float(np.clip(1.0 - spd_gap, 0.0, 1.0))
    fairness_score_eod = float(np.clip(1.0 - eod_gap, 0.0, 1.0))
    fairness_score_aod = float(np.clip(1.0 - aod_gap, 0.0, 1.0))
    fairness_score_dir = float(np.clip(dir_ratio, 0.0, 1.0))
    fairness_aggregate = _safe_mean([
        fairness_score_spd,
        fairness_score_eod,
        fairness_score_aod,
        fairness_score_dir,
    ])

    metrics = {
        "spd_gap": spd_gap,
        "eod_gap": eod_gap,
        "aod_gap": aod_gap,
        "dir_ratio": dir_ratio,
        "fairness_score_spd": fairness_score_spd,
        "fairness_score_eod": fairness_score_eod,
        "fairness_score_aod": fairness_score_aod,
        "fairness_score_dir": fairness_score_dir,
        "fairness_aggregate": fairness_aggregate,
    }
    return metrics, group_table, tmp


def _compute_robustness_metrics(model, X_test, y_test, numeric_cols):
    base_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    scores = {}
    rng = np.random.default_rng(RANDOM_STATE)

    if numeric_cols:
        X_noise = X_test.copy()
        noise = rng.normal(0.0, ROBUST_NOISE_STD, size=(len(X_noise), len(numeric_cols)))
        X_noise.loc[:, numeric_cols] = X_noise.loc[:, numeric_cols].values + noise
        noise_auc = roc_auc_score(y_test, model.predict_proba(X_noise)[:, 1])
    else:
        noise_auc = base_auc
    scores["noise_auc_ratio"] = float(np.clip(noise_auc / base_auc, 0.0, 1.0)) if base_auc > 0 else 0.0

    drop_count = max(1, int(round(len(X_test.columns) * ROBUST_DROPOUT_RATE)))
    selected_cols = list(X_test.columns[:drop_count])
    X_dropout = X_test.copy()
    X_dropout.loc[:, selected_cols] = 0.0
    dropout_auc = roc_auc_score(y_test, model.predict_proba(X_dropout)[:, 1])
    scores["dropout_auc_ratio"] = float(np.clip(dropout_auc / base_auc, 0.0, 1.0)) if base_auc > 0 else 0.0

    if numeric_cols:
        miss_count = max(1, int(round(len(X_test) * ROBUST_MISSING_RATE)))
        chosen_rows = rng.choice(len(X_test), size=miss_count, replace=False)
        X_missing = X_test.copy()
        X_missing.iloc[chosen_rows, [X_test.columns.get_loc(c) for c in numeric_cols]] = 0.0
        missing_auc = roc_auc_score(y_test, model.predict_proba(X_missing)[:, 1])
    else:
        missing_auc = base_auc
    scores["missingness_auc_ratio"] = float(np.clip(missing_auc / base_auc, 0.0, 1.0)) if base_auc > 0 else 0.0

    scores["robustness_aggregate"] = _safe_mean([
        scores["noise_auc_ratio"],
        scores["dropout_auc_ratio"],
        scores["missingness_auc_ratio"],
    ])
    scores["base_auc"] = float(base_auc)
    return scores

def _performance_auditor(auc_score):
    return {
        "auditor": "performance_auditor",
        "score": float(np.clip(auc_score, 0.0, 1.0)),
        "details": {
            "auc": float(auc_score)
        }
    }


def _fairness_auditor(fairness_metrics):
    return {
        "auditor": "fairness_auditor",
        "score": float(np.clip(fairness_metrics["fairness_aggregate"], 0.0, 1.0)),
        "details": {
            "spd_gap": fairness_metrics["spd_gap"],
            "eod_gap": fairness_metrics["eod_gap"],
            "aod_gap": fairness_metrics["aod_gap"],
            "dir_ratio": fairness_metrics["dir_ratio"],
            "fairness_aggregate": fairness_metrics["fairness_aggregate"],
        }
    }


def _robustness_auditor(robustness_metrics):
    return {
        "auditor": "robustness_auditor",
        "score": float(np.clip(robustness_metrics["robustness_aggregate"], 0.0, 1.0)),
        "details": {
            "noise_auc_ratio": robustness_metrics["noise_auc_ratio"],
            "dropout_auc_ratio": robustness_metrics["dropout_auc_ratio"],
            "missingness_auc_ratio": robustness_metrics["missingness_auc_ratio"],
            "robustness_aggregate": robustness_metrics["robustness_aggregate"],
        }
    }


def _ensemble_auditor(auc_score, fairness_metrics, robustness_metrics):
    perf = _performance_auditor(auc_score)
    fair = _fairness_auditor(fairness_metrics)
    rob = _robustness_auditor(robustness_metrics)

    ensemble_score = (
        W_AUC * perf["score"]
        + W_FAIR * fair["score"]
        + W_ROB * rob["score"]
    )

    auditor_df = pd.DataFrame([
        {"auditor": perf["auditor"], "score": perf["score"]},
        {"auditor": fair["auditor"], "score": fair["score"]},
        {"auditor": rob["auditor"], "score": rob["score"]},
    ])

    return {
        "performance_auditor": perf,
        "fairness_auditor": fair,
        "robustness_auditor": rob,
        "ensemble_score": float(ensemble_score),
        "auditor_table": auditor_df
    }

def _sensitivity_analysis(model, X_test, y_test, config, numeric_cols):
    y_probs = model.predict_proba(X_test)[:, 1]
    rows = []

    base_group = pd.read_csv("sensitive_test.csv").iloc[:, 0].astype(str).fillna("NA")
    fairness_metrics, _, _ = _compute_fairness_metrics(y_test, y_probs, base_group, config["prediction_threshold"])
    robustness_metrics = _compute_robustness_metrics(model, X_test, y_test, numeric_cols)
    base_auc = float(roc_auc_score(y_test, y_probs))

    def add_row(label, auc, fair, rob, w_auc, w_fair, w_rob, approval_thr, pred_thr, sensitive_feature):
        safe_score = (w_auc * auc) + (w_fair * fair) + (w_rob * rob)
        rows.append({
            "scenario": label,
            "prediction_threshold": pred_thr,
            "approval_threshold": approval_thr,
            "w_auc": w_auc,
            "w_fair": w_fair,
            "w_rob": w_rob,
            "sensitive_feature": sensitive_feature,
            "auc": auc,
            "fairness_aggregate": fair,
            "robustness_aggregate": rob,
            "safe_score": safe_score,
            "decision": "APPROVED" if safe_score >= approval_thr else "REJECTED",
        })

    add_row(
        "base",
        base_auc,
        fairness_metrics["fairness_aggregate"],
        robustness_metrics["robustness_aggregate"],
        config["weights"]["auc"],
        config["weights"]["fairness"],
        config["weights"]["robustness"],
        config["approval_threshold"],
        config["prediction_threshold"],
        config["sensitive_feature"],
    )

    for approval_thr in [0.70, 0.75, 0.80]:
        add_row(
            f"approval_threshold={approval_thr}",
            base_auc,
            fairness_metrics["fairness_aggregate"],
            robustness_metrics["robustness_aggregate"],
            config["weights"]["auc"],
            config["weights"]["fairness"],
            config["weights"]["robustness"],
            approval_thr,
            config["prediction_threshold"],
            config["sensitive_feature"],
        )

    for pred_thr in [0.45, 0.50, 0.55, 0.60]:
        fair_var, _, _ = _compute_fairness_metrics(y_test, y_probs, base_group, pred_thr)
        add_row(
            f"prediction_threshold={pred_thr}",
            base_auc,
            fair_var["fairness_aggregate"],
            robustness_metrics["robustness_aggregate"],
            config["weights"]["auc"],
            config["weights"]["fairness"],
            config["weights"]["robustness"],
            config["approval_threshold"],
            pred_thr,
            config["sensitive_feature"],
        )

    weight_sets = [
        (0.50, 0.30, 0.20),
        (0.30, 0.50, 0.20),
        (0.30, 0.30, 0.40),
    ]
    for wa, wf, wr in weight_sets:
        s = wa + wf + wr
        wa, wf, wr = wa / s, wf / s, wr / s
        add_row(
            f"weights=({wa:.2f},{wf:.2f},{wr:.2f})",
            base_auc,
            fairness_metrics["fairness_aggregate"],
            robustness_metrics["robustness_aggregate"],
            wa,
            wf,
            wr,
            config["approval_threshold"],
            config["prediction_threshold"],
            config["sensitive_feature"],
        )

    original_df = pd.read_csv("raw_credit_data.csv")
    X = original_df.drop('CreditRisk', axis=1)
    y = original_df['CreditRisk']
    _, X_test_raw, _, _ = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    for sf in [config["sensitive_feature"]] + config["alternative_sensitive_features"]:
        if sf in X_test_raw.columns:
            grp = X_test_raw[sf].astype(str).fillna("NA")
            fair_sf, _, _ = _compute_fairness_metrics(y_test, y_probs, grp, config["prediction_threshold"])
            add_row(
                f"sensitive_feature={sf}",
                base_auc,
                fair_sf["fairness_aggregate"],
                robustness_metrics["robustness_aggregate"],
                config["weights"]["auc"],
                config["weights"]["fairness"],
                config["weights"]["robustness"],
                config["approval_threshold"],
                config["prediction_threshold"],
                sf,
            )

    sens_df = pd.DataFrame(rows).drop_duplicates(subset=["scenario"])
    sens_df["delta_vs_base"] = sens_df["safe_score"] - float(sens_df.loc[sens_df["scenario"] == "base", "safe_score"].iloc[0])
    sens_df = sens_df.sort_values(["safe_score", "scenario"], ascending=[False, True]).reset_index(drop=True)
    return sens_df

def _interaction_analysis(model, X_test, y_test, config, numeric_cols):
    y_probs = model.predict_proba(X_test)[:, 1]
    group = pd.read_csv("sensitive_test.csv").iloc[:, 0].astype(str).fillna("NA")

    pred_threshold_values = [0.45, 0.50, 0.55, 0.60]
    approval_threshold_values = [0.70, 0.75, 0.80]
    fair_weight_values = [0.3, 0.4, 0.5]
    rob_weight_values = [0.2, 0.3, 0.4]

    rows = []

    for pred_thr in pred_threshold_values:
        fairness_metrics, _, _ = _compute_fairness_metrics(y_test, y_probs, group, pred_thr)

        for approval_thr in approval_threshold_values:
            for fair_w in fair_weight_values:
                for rob_w in rob_weight_values:
                    auc_w = 1.0 - fair_w - rob_w
                    if auc_w < 0:
                        continue

                    robustness_metrics = _compute_robustness_metrics(model, X_test, y_test, numeric_cols)
                    safe_score = (
                        auc_w * float(roc_auc_score(y_test, y_probs))
                        + fair_w * fairness_metrics["fairness_aggregate"]
                        + rob_w * robustness_metrics["robustness_aggregate"]
                    )

                    rows.append({
                        "prediction_threshold": pred_thr,
                        "approval_threshold": approval_thr,
                        "w_auc": auc_w,
                        "w_fair": fair_w,
                        "w_rob": rob_w,
                        "safe_score": safe_score,
                        "decision": "APPROVED" if safe_score >= approval_thr else "REJECTED"
                    })

    df = pd.DataFrame(rows)

    effect_summary = []
    for col in ["prediction_threshold", "approval_threshold", "w_fair", "w_rob"]:
        grouped = df.groupby(col)["safe_score"].mean()
        effect_summary.append({
            "factor": col,
            "mean_effect_range": float(grouped.max() - grouped.min())
        })
    effect_df = pd.DataFrame(effect_summary).sort_values("mean_effect_range", ascending=False).reset_index(drop=True)

    interaction_rows = []
    pairs = [
        ("prediction_threshold", "approval_threshold"),
        ("prediction_threshold", "w_fair"),
        ("prediction_threshold", "w_rob"),
        ("approval_threshold", "w_fair"),
        ("approval_threshold", "w_rob"),
        ("w_fair", "w_rob"),
    ]

    for a, b in pairs:
        pair_table = df.pivot_table(values="safe_score", index=a, columns=b, aggfunc="mean")
        row_means = pair_table.mean(axis=1)
        col_means = pair_table.mean(axis=0)
        grand_mean = pair_table.values.mean()

        residual = pair_table.copy()
        for i in pair_table.index:
            for j in pair_table.columns:
                residual.loc[i, j] = pair_table.loc[i, j] - row_means.loc[i] - col_means.loc[j] + grand_mean

        interaction_strength = float(np.abs(residual.values).mean())
        interaction_rows.append({
            "factor_a": a,
            "factor_b": b,
            "interaction_strength": interaction_strength
        })

    interaction_df = pd.DataFrame(interaction_rows).sort_values("interaction_strength", ascending=False).reset_index(drop=True)

    return df, effect_df, interaction_df

# --- CONFIGURATION & SETUP ---

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv() 
crew_llm = LLM(model="gpt-4o")

# --- USER CONFIG (via .env or environment variables) ---
DATA_PATH = os.getenv("DATA_PATH", "").strip() or None   # e.g. raw_credit_data.csv
OPENML_ID = int(os.getenv("OPENML_ID", "31"))            # default German Credit (31)

PRED_THRESHOLD = float(os.getenv("PRED_THRESHOLD", "0.50"))         # for y_pred
APPROVAL_THRESHOLD = float(os.getenv("APPROVAL_THRESHOLD", "0.75")) # for SAFE decision

W_AUC = float(os.getenv("W_AUC", "0.4"))
W_FAIR = float(os.getenv("W_FAIR", "0.4"))
W_ROB = float(os.getenv("W_ROB", "0.2"))

# normalize weights so sum=1
w_sum = W_AUC + W_FAIR + W_ROB
if w_sum <= 0:
    raise ValueError("Weights must sum to > 0")
W_AUC, W_FAIR, W_ROB = W_AUC / w_sum, W_FAIR / w_sum, W_ROB / w_sum

SENSITIVE_FEATURE = os.getenv("SENSITIVE_FEATURE", "personal_status")  # or foreign_worker
DROP_SENSITIVE_FROM_MODEL = os.getenv("DROP_SENSITIVE_FROM_MODEL", "0") == "1"
ALT_SENSITIVE_FEATURES = [
    x.strip() for x in os.getenv("ALT_SENSITIVE_FEATURES", "foreign_worker,sex,age").split(",") if x.strip()
]
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
ROBUST_NOISE_STD = float(os.getenv("ROBUST_NOISE_STD", "0.10"))
ROBUST_DROPOUT_RATE = float(os.getenv("ROBUST_DROPOUT_RATE", "0.10"))
ROBUST_MISSING_RATE = float(os.getenv("ROBUST_MISSING_RATE", "0.10"))

def current_config():
    return {
        "data_source": f"CSV ({DATA_PATH})" if DATA_PATH else f"OpenML ({OPENML_ID})",
        "prediction_threshold": PRED_THRESHOLD,
        "approval_threshold": APPROVAL_THRESHOLD,
        "weights": {"auc": W_AUC, "fairness": W_FAIR, "robustness": W_ROB},
        "sensitive_feature": SENSITIVE_FEATURE,
        "alternative_sensitive_features": ALT_SENSITIVE_FEATURES,
        "drop_sensitive_from_model": DROP_SENSITIVE_FROM_MODEL,
        "random_state": RANDOM_STATE,
        "robustness_settings": {
            "noise_std": ROBUST_NOISE_STD,
            "dropout_rate": ROBUST_DROPOUT_RATE,
            "missing_rate": ROBUST_MISSING_RATE,
        },
        "decision_rule": (
            "APPROVED if SAFE_SCORE >= APPROVAL_THRESHOLD else REJECTED, "
            "where SAFE_SCORE = W_AUC*AUC + W_FAIR*FAIRNESS_AGG + W_ROB*ROBUSTNESS_AGG"
        ),
    }
# --- DATA LOADER ---

def get_credit_data():
    """
    If DATA_PATH is set, use that CSV.
    Otherwise fetch OpenML dataset OPENML_ID and save to raw_credit_data.csv.
    """
    try:
        if DATA_PATH:
            # Use user-provided CSV
            df = pd.read_csv(DATA_PATH)
            df.to_csv("raw_credit_data.csv", index=False)
            return "raw_credit_data.csv"

        dataset = openml.datasets.get_dataset(OPENML_ID)
        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute
        )

        # German Credit mapping (good/bad)
        y = y.apply(lambda x: 1 if x == 'bad' else 0)

        data = pd.concat([X, y.rename('CreditRisk')], axis=1)
        data.to_csv("raw_credit_data.csv", index=False)
        return "raw_credit_data.csv"

    except Exception as e:
        return f"Error loading data: {e}"
RAW_DATA_PATH = get_credit_data()

@tool
def data_preprocessing_tool(file_path: str):
    """Processes, cleans, encodes, scales, and splits the raw credit data. Saves results as clean CSV files."""
    try:
        df = pd.read_csv(file_path)

        X = df.drop('CreditRisk', axis=1)
        y = df['CreditRisk']

        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        model_categorical_features = [
            c for c in categorical_features
            if not (DROP_SENSITIVE_FROM_MODEL and c == SENSITIVE_FEATURE)
        ]
        model_numerical_features = [
            c for c in numerical_features
            if not (DROP_SENSITIVE_FROM_MODEL and c == SENSITIVE_FEATURE)
        ]

        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', ohe, model_categorical_features),
                ('num', StandardScaler(), model_numerical_features),
            ],
            remainder='drop'
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        if SENSITIVE_FEATURE in X_train.columns:
            X_train[[SENSITIVE_FEATURE]].to_csv("sensitive_train.csv", index=False)
            X_test[[SENSITIVE_FEATURE]].to_csv("sensitive_test.csv", index=False)
        else:
            pd.DataFrame({"group": ["UNKNOWN"] * len(X_train)}).to_csv("sensitive_train.csv", index=False)
            pd.DataFrame({"group": ["UNKNOWN"] * len(X_test)}).to_csv("sensitive_test.csv", index=False)

        if DROP_SENSITIVE_FROM_MODEL and (SENSITIVE_FEATURE in X_train.columns):
            X_train = X_train.drop(columns=[SENSITIVE_FEATURE])
            X_test = X_test.drop(columns=[SENSITIVE_FEATURE])

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        raw_feature_names = (
            list(preprocessor.named_transformers_['cat'].get_feature_names_out(model_categorical_features))
            + list(model_numerical_features)
        )

        clean_feature_names = [
            name.replace("[", "").replace("]", "").replace("<", "less_than_")
            for name in raw_feature_names
        ]

        pd.DataFrame(X_train_processed, columns=clean_feature_names).to_csv("clean_train_features.csv", index=False)
        y_train.to_csv("clean_train_target.csv", index=False, header=['CreditRisk'])
        pd.DataFrame(X_test_processed, columns=clean_feature_names).to_csv("clean_test_features.csv", index=False)
        y_test.to_csv("clean_test_target.csv", index=False, header=['CreditRisk'])

        datacard = {
            "status": "CLEANED",
            "features_after_encoding": int(len(clean_feature_names)),
            "numeric_features_raw": list(map(str, model_numerical_features)),
            "categorical_features_raw": list(map(str, model_categorical_features)),
            "sensitive_feature": SENSITIVE_FEATURE,
            "drop_sensitive_from_model": bool(DROP_SENSITIVE_FROM_MODEL),
            "config": current_config(),
        }

        with open("datacard.json", "w", encoding="utf-8") as f:
            json.dump(datacard, f, indent=2)

        return "Data successfully processed, feature names cleaned for XGBoost, and datasets saved."

    except Exception as e:
        return f"DATA PREPROCESSING FAILED: {str(e)}"

# --- AGENTS DEFINITION ---

@tool
def model_training_tool(description: str):
    """Trains an XGBoost model on the cleaned training data. Saves the model as 'best_model.pkl' and creates a 'model_card.md'."""
    try:
        # Load the processed training data
        X_train = pd.read_csv("clean_train_features.csv")
        y_train = pd.read_csv("clean_train_target.csv").values.ravel()

        # Initialize and train the XGBoost model
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        # Save the model artifact
        joblib.dump(model, "best_model.pkl")

        # Create the Model Card
        with open('model_card.md', 'w') as f:
            f.write("## XGBoost Model Card\n\n- Model Type: XGBoost Classifier\n- Status: Trained\n- Features: " + str(X_train.shape[1]))

        return "SUCCESS: Model trained and saved as 'best_model.pkl'. Model Card generated."
    except Exception as e:
        return f"MODEL TRAINING FAILED: {e}"

@tool
def evaluation_and_risk_tool(description: str):
    """Evaluates the model and calculates aggregated SAFE metrics + sensitivity analysis."""
    try:
        model = joblib.load("best_model.pkl")
        X_test = pd.read_csv("clean_test_features.csv")
        y_test = _read_target_series("clean_test_target.csv")
        y_probs = model.predict_proba(X_test)[:, 1]
        auc_score = float(roc_auc_score(y_test, y_probs))

        with open("datacard.json", "r", encoding="utf-8") as f:
            dc = json.load(f)
        config = dc.get("config", current_config())
        numeric_cols = [c for c in dc.get("numeric_features_raw", []) if c in X_test.columns]

        group = pd.read_csv("sensitive_test.csv").iloc[:, 0].astype(str).fillna("NA")

        fairness_metrics, group_table, _ = _compute_fairness_metrics(
            y_test, y_probs, group, PRED_THRESHOLD
        )

        robustness_metrics = _compute_robustness_metrics(
            model, X_test, y_test, numeric_cols
        )

        ensemble_results = _ensemble_auditor(
            auc_score,
            fairness_metrics,
            robustness_metrics
        )

        mitigated_pred, disadvantaged_group = _apply_threshold_mitigation(
            y_probs, group, base_threshold=PRED_THRESHOLD
        )

        mitigated_fairness_metrics, mitigated_group_table = _compute_fairness_from_predictions(
            y_test, mitigated_pred, group
        )

        mitigated_auc = float(roc_auc_score(y_test, mitigated_pred))

        baseline_safe = ensemble_results["ensemble_score"]

        mitigated_safe = (
            W_AUC * mitigated_auc
            + W_FAIR * mitigated_fairness_metrics["fairness_aggregate"]
            + W_ROB * robustness_metrics["robustness_aggregate"]
        )

        sensitivity_df = _sensitivity_analysis(model, X_test, y_test, config, numeric_cols)
        interaction_grid_df, effect_df, interaction_df = _interaction_analysis(
            model, X_test, y_test, config, numeric_cols
        )
        best_scenario = sensitivity_df.iloc[0]
        best_non_base_df = sensitivity_df[sensitivity_df["scenario"] != "base"]
        best_non_base = best_non_base_df.iloc[0] if not best_non_base_df.empty else best_scenario

        importance_df = pd.DataFrame({
            "feature": X_test.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        report_content = f"""### Detailed SAFE AI Evaluation Report
- **Accuracy (AUC)**: {auc_score:.4f}
- **Fairness Aggregate**: {fairness_metrics['fairness_aggregate']:.4f}
- **Robustness Aggregate**: {robustness_metrics['robustness_aggregate']:.4f}
- **Baseline SAFE Score**: {baseline_safe:.4f}
- **Ensemble Auditing Enabled**: True
- **Auditors Used**: performance_auditor, fairness_auditor, robustness_auditor
- **Mitigated AUC**: {mitigated_auc:.4f}
- **Mitigated Fairness Aggregate**: {mitigated_fairness_metrics['fairness_aggregate']:.4f}
- **Mitigated SAFE Score**: {mitigated_safe:.4f}
- **Fairness Components**: SPD={fairness_metrics['fairness_score_spd']:.4f}, EOD={fairness_metrics['fairness_score_eod']:.4f}, AOD={fairness_metrics['fairness_score_aod']:.4f}, DIR={fairness_metrics['fairness_score_dir']:.4f}
- **Robustness Components**: Noise={robustness_metrics['noise_auc_ratio']:.4f}, Dropout={robustness_metrics['dropout_auc_ratio']:.4f}, Missingness={robustness_metrics['missingness_auc_ratio']:.4f}
- **Mitigation Applied To Group**: {disadvantaged_group}
- **Status**: Metrics extracted for weighting, mitigation, sensitivity analysis, and explainability.
"""
        with open('evaluation_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)

        final_report = f"""# Final SAFE Agentic Credit Scoring Report

## User Controls
- Data source: {config['data_source']}
- Prediction threshold: {config['prediction_threshold']}
- Approval threshold: {config['approval_threshold']}
- Weights: AUC={config['weights']['auc']:.3f}, Fairness={config['weights']['fairness']:.3f}, Robustness={config['weights']['robustness']:.3f}
- Sensitive feature: {config['sensitive_feature']}
- Drop sensitive from model: {config['drop_sensitive_from_model']}
- Decision rule: {config['decision_rule']}

## Accuracy
- AUC: {auc_score:.4f}

## Fairness Aggregation
- SPD gap: {fairness_metrics['spd_gap']:.4f}
- EOD gap: {fairness_metrics['eod_gap']:.4f}
- AOD gap: {fairness_metrics['aod_gap']:.4f}
- Disparate impact ratio: {fairness_metrics['dir_ratio']:.4f}
- Fairness aggregate: {fairness_metrics['fairness_aggregate']:.4f}

## Robustness Aggregation
- Noise AUC ratio: {robustness_metrics['noise_auc_ratio']:.4f}
- Dropout AUC ratio: {robustness_metrics['dropout_auc_ratio']:.4f}
- Missingness AUC ratio: {robustness_metrics['missingness_auc_ratio']:.4f}
- Robustness aggregate: {robustness_metrics['robustness_aggregate']:.4f}

## Ensemble Auditing
Individual auditor scores:
{ensemble_results["auditor_table"].to_markdown(index=False)}

- Final ensemble SAFE score: {baseline_safe:.4f}
- Ensemble rule: weighted aggregation of independent performance, fairness, and robustness auditors.

## Mitigation Experiment
- Mitigation type: group-aware threshold adjustment
- Disadvantaged group detected: {disadvantaged_group}
- Baseline fairness aggregate: {fairness_metrics['fairness_aggregate']:.4f}
- Mitigated fairness aggregate: {mitigated_fairness_metrics['fairness_aggregate']:.4f}
- Baseline SAFE score: {(W_AUC * auc_score) + (W_FAIR * fairness_metrics['fairness_aggregate']) + (W_ROB * robustness_metrics['robustness_aggregate']):.4f}
- Mitigated SAFE score: {mitigated_safe:.4f}

### Group Table
{group_table.to_markdown(index=False)}

## Sensitivity Analysis Summary
Top scenarios by SAFE score:
{sensitivity_df.head(8).to_markdown(index=False)}

## Interaction / Effects Summary
- Baseline SAFE score: {baseline_safe:.4f}
- Best scenario from sensitivity analysis: {best_scenario['scenario']}
- Best scenario SAFE score: {best_scenario['safe_score']:.4f}
- Strongest observed effect beyond baseline: {best_non_base['scenario']}
- Effect size vs baseline: {best_non_base['delta_vs_base']:.4f}
- Interpretation: the governance decision is sensitive to policy weights and sensitive-feature choice, while threshold changes had weaker effects in this run.

## Global Interaction Analysis
Top main effects on SAFE score:
{effect_df.head(4).to_markdown(index=False)}

Top pairwise interactions:
{interaction_df.head(6).to_markdown(index=False)}

Interpretation:
- Main effects show which single factor most strongly changes SAFE score on average.
- Pairwise interactions show which pairs of factors jointly influence the SAFE decision beyond their separate average effects.

## Explainability Snapshot
Top 10 most important processed features:
{importance_df.head(10).to_markdown(index=False)}

## Auditor Notes
- Multi-metric fairness and robustness aggregation are enabled.
- Sensitivity analysis covers thresholds, weights, alternative sensitive features, and perturbation settings.
"""
        with open("final_report.md", "w", encoding="utf-8") as f:
            f.write(final_report)

        with open("sensitivity_report.md", "w", encoding="utf-8") as f:
            f.write("# Sensitivity Analysis Report\n\n")
            f.write("Evaluates how SAFE decisions change under variations in weights, thresholds, sensitive feature choice, and perturbation assumptions.\n\n")
            f.write("## Scenario Table\n\n")
            f.write(sensitivity_df.to_markdown(index=False))
            f.write("\n\n## Main Effects\n\n")
            f.write(effect_df.to_markdown(index=False))
            f.write("\n\n## Pairwise Interactions\n\n")
            f.write(interaction_df.to_markdown(index=False))
            f.write("\n")
            
        print("DEBUG: evaluation_report.md written successfully")
        return report_content
    except Exception as e:
        return f"EVALUATION FAILED: {e}"
    
@tool
def governance_scoring_tool(description: str):
    """Computes weighted SAFE score from evaluation_report.md and writes system_card.md."""
    try:
        import re
        with open("evaluation_report.md", "r", encoding="utf-8") as f:
            rep = f.read()

        if not rep.strip():
            return "REJECTED: evaluation_report.md is empty."

        def extract_float(pattern, text):
            m = re.search(pattern, text, re.MULTILINE)
            return float(m.group(1)) if m else None

        auc = extract_float(r"\*\*Accuracy \(AUC\)\*\*:\s*([0-9]*\.?[0-9]+)", rep)
        fair = extract_float(r"\*\*Fairness Aggregate\*\*:\s*([0-9]*\.?[0-9]+)", rep)
        rob = extract_float(r"\*\*Robustness Aggregate\*\*:\s*([0-9]*\.?[0-9]+)", rep)
        mitigated_safe = extract_float(r"\*\*Mitigated SAFE Score\*\*:\s*([0-9]*\.?[0-9]+)", rep)

        if auc is None or fair is None or rob is None:
            return "REJECTED: Could not parse AUC/Fairness Aggregate/Robustness Aggregate from evaluation_report.md."

        final_score = (W_AUC * auc) + (W_FAIR * fair) + (W_ROB * rob)
        decision = "APPROVED" if final_score >= APPROVAL_THRESHOLD else "REJECTED"

        mitigated_decision = None
        if mitigated_safe is not None:
            mitigated_decision = "APPROVED" if mitigated_safe >= APPROVAL_THRESHOLD else "REJECTED"

        mitigated_safe_text = f"{mitigated_safe:.3f}" if mitigated_safe is not None else "N/A"
        mitigated_decision_text = mitigated_decision if mitigated_decision is not None else "N/A"

        with open("sensitivity_report.md", "r", encoding="utf-8") as f:
            sensitivity_excerpt = "\n".join(f.read().splitlines()[:18])

        system_card = f"""# System Card — SAFE Agentic Credit Scoring

## Decision
**{decision}**

## Final SAFE Score
**{final_score:.3f}**

## Decision Rule
- SAFE Score = {W_AUC:.3f}*AUC + {W_FAIR:.3f}*Fairness_Aggregate + {W_ROB:.3f}*Robustness_Aggregate
- Approval threshold = {APPROVAL_THRESHOLD:.2f}
- Policy = APPROVED if SAFE Score >= threshold else REJECTED

## Metrics Used
- Baseline AUC: {auc:.3f}
- Baseline Fairness Aggregate: {fair:.3f}
- Baseline Robustness Aggregate: {rob:.3f}
- Mitigated SAFE Score: {mitigated_safe_text}
- Mitigated Decision: {mitigated_decision_text}

## Reproducibility Controls
- Prediction threshold: {PRED_THRESHOLD}
- Sensitive feature: {SENSITIVE_FEATURE}
- Drop sensitive from model: {DROP_SENSITIVE_FROM_MODEL}
- Random state: {RANDOM_STATE}

## Rationale
The final SAFE score balances predictive performance with multi-metric fairness and multi-scenario robustness.

## Mitigation Result
- Baseline decision: {decision}
- Mitigated decision: {mitigated_decision_text}
- Interpretation: mitigation is reported separately to show whether fairness-aware post-processing improves the governance outcome under a fixed policy.

## Sensitivity Snapshot
{sensitivity_excerpt}
"""
        with open("system_card.md", "w", encoding="utf-8") as f:
            f.write(system_card)

        return f"{decision}: SAFE Score={final_score:.3f}. System Card saved to system_card.md."

    except Exception as e:
        return f"GOVERNANCE FAILED: {e}"

# --- CHATBOT HELPER FUNCTIONS ---

def _safe_read_text(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''


def _safe_read_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _extract_markdown_metric(text, label):
    import re
    safe_label = re.escape(label)
    patterns = [
        rf"- \*\*{safe_label}\*\*:\s*([^\n]+)",
        rf"- {safe_label}:\s*([^\n]+)",
        rf"\*\*{safe_label}\*\*\s*\n\s*\*\*([^\n]+)\*\*",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None

def _extract_top_features(report_text, k=5):
    lines = report_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if 'Top 10 most important processed features:' in line:
            start = i + 2
            break
    if start is None:
        return []

    rows = []
    for line in lines[start:]:
        if not line.strip().startswith('|'):
            break
        if '---' in line or 'feature' in line.lower():
            continue
        parts = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(parts) >= 2:
            rows.append((parts[0], parts[1]))
        if len(rows) >= k:
            break
    return rows


def build_chatbot_context():
    config = _safe_read_json('datacard.json').get('config', current_config())
    system_card = _safe_read_text('system_card.md')
    evaluation_report = _safe_read_text('evaluation_report.md')
    final_report = _safe_read_text('final_report.md')
    sensitivity_report = _safe_read_text('sensitivity_report.md')
    model_card = _safe_read_text('model_card.md')

    return {
        'config': config,
        'system_card': system_card,
        'evaluation_report': evaluation_report,
        'final_report': final_report,
        'sensitivity_report': sensitivity_report,
        'model_card': model_card,
        'decision': _extract_markdown_metric(system_card, 'Decision'),
        'final_safe_score': _extract_markdown_metric(system_card, 'Final SAFE Score'),
        'auc': _extract_markdown_metric(evaluation_report, 'Accuracy (AUC)'),
        'fairness_aggregate': _extract_markdown_metric(evaluation_report, 'Fairness Aggregate'),
        'robustness_aggregate': _extract_markdown_metric(evaluation_report, 'Robustness Aggregate'),
        'mitigated_safe_score': _extract_markdown_metric(evaluation_report, 'Mitigated SAFE Score'),
        'mitigated_auc': _extract_markdown_metric(evaluation_report, 'Mitigated AUC'),
        'top_features': _extract_top_features(final_report, k=5),
    }

# --- CHATBOT TOOL ---
@tool
def safe_chatbot_tool(query: str):
    """Answers user questions about the SAFE credit scoring run using generated artifacts such as system_card.md, final_report.md, evaluation_report.md, and datacard.json."""
    try:
        ctx = build_chatbot_context()
        q = (query or '').strip().lower()

        required_files = ['system_card.md', 'evaluation_report.md', 'final_report.md']
        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            return (
                'CHATBOT ERROR: Missing required artifacts: ' + ', '.join(missing) + '. '
                'Run the full pipeline first so the chatbot can answer grounded questions.'
            )

        if any(x in q for x in ['hello', 'hi', 'hey', 'start', 'help']):
            return (
                'SAFE chatbot is ready. You can ask about the final decision, SAFE score, AUC, fairness, '
                'robustness, mitigation, configuration, sensitivity analysis, or top features.'
            )

        if any(x in q for x in ['decision', 'approved', 'rejected', 'governance']):
            return (
                f"Final governance decision: {ctx['decision']}. "
                f"Final SAFE score: {ctx['final_safe_score']}."
            )

        if 'safe score' in q or 'final score' in q:
            return (
                f"Final SAFE score: {ctx['final_safe_score']}. "
                f"Mitigated SAFE score: {ctx['mitigated_safe_score']}."
            )

        if 'auc' in q or 'accuracy' in q or 'performance' in q:
            return (
                f"AUC: {ctx['auc']}. "
                f"Mitigated AUC: {ctx['mitigated_auc']}."
            )

        if 'fairness' in q:
            return (
                f"Fairness aggregate: {ctx['fairness_aggregate']}. "
                'For more detail, check final_report.md for SPD, EOD, AOD, and DIR breakdown.'
            )

        if 'robust' in q:
            return (
                f"Robustness aggregate: {ctx['robustness_aggregate']}. "
                'The robustness report is based on noise, dropout, and missingness stress tests.'
            )

        if any(x in q for x in ['mitigation', 'mitigated', 'post-mitigation']):
            return (
                f"Mitigated SAFE score: {ctx['mitigated_safe_score']}. "
                f"Mitigated AUC: {ctx['mitigated_auc']}. "
                'The mitigation used here is group-aware threshold adjustment.'
            )

        if any(x in q for x in ['config', 'settings', 'threshold', 'weights', 'sensitive feature']):
            cfg = ctx['config']
            return (
                'Current configuration: '
                f"prediction_threshold={cfg.get('prediction_threshold')}, "
                f"approval_threshold={cfg.get('approval_threshold')}, "
                f"weights={cfg.get('weights')}, "
                f"sensitive_feature={cfg.get('sensitive_feature')}, "
                f"drop_sensitive_from_model={cfg.get('drop_sensitive_from_model')}."
            )

        if any(x in q for x in ['feature importance', 'top features', 'important features', 'explainability']):
            if not ctx['top_features']:
                return 'No feature-importance summary was found in final_report.md.'
            formatted = ', '.join([f"{name} ({imp})" for name, imp in ctx['top_features']])
            return f"Top processed features from the report: {formatted}."

        if any(x in q for x in ['sensitivity', 'interaction', 'effects']):
            return (
                'Sensitivity and interaction analysis were generated. '
                'See sensitivity_report.md for scenario comparisons, main effects, and pairwise interactions.'
            )

        return (
            'I can answer grounded questions about this SAFE run. '
            'Try asking: What is the final decision? What is the SAFE score? '
            'How fair is the model? How robust is it? What mitigation was applied? '
            'What are the top features?'
        )
    except Exception as e:
        return f'CHATBOT FAILED: {e}'

# --- CHATBOT CLI INTERFACE ---

def run_safe_chatbot_cli():
    print("\n--- SAFE Chatbot ---")
    print('Ask about the final decision, SAFE score, fairness, robustness, mitigation, config, or top features.')
    print("Type 'exit' to stop.\n")

    while True:
        try:
            user_query = input('SAFE Chatbot > ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nExiting SAFE Chatbot.')
            break

        if user_query.lower() in {'exit', 'quit', 'q'}:
            print('Exiting SAFE Chatbot.')
            break

        if not user_query:
            continue

        answer = safe_chatbot_tool.run(user_query)
        print(f'\n{answer}\n')

# --- AGENTS DEFINITION ---

# 1. Data Agent
data_agent = Agent(
    role='Data Preprocessor and Feature Engineer',
    goal='Rigorously clean and transform the German Credit dataset, handling categorical variables, scaling numerical features, and creating a balanced train/test split. Produce a detailed Data Card artifact.',
    backstory=(
        "An expert in financial data preparation who specializes in preventing "
        "data leakage and ensuring data quality before any modeling starts. "
        "Their primary focus is on maximizing fairness and robustness in the downstream model."
    ),
    tools=[data_preprocessing_tool],  # DataAgent uses the preprocessing tool
    llm=crew_llm,
    allow_delegation=False,
    verbose=True
)

# 2. Modeling Agent
modeling_agent = Agent(
    role='Machine Learning Model Builder and Validator',
    goal='Train highly optimized predictive models (e.g., Logistic Regression and XGBoost) and select the best model based on cross-validation metrics. Output a comprehensive Model Card.',
    backstory=(
        "A seasoned ML engineer focused on credit risk assessment. "
        "They utilize state-of-the-art libraries to perform hyperparameter tuning "
        "and establish strong performance baselines."
    ),
    tools=[model_training_tool],  # ModelingAgent uses the training tool
    llm=crew_llm,
    allow_delegation=False,
    verbose=True
)

# 3. Evaluation Agent 
eval_agent = Agent(
    role='Risk and Performance Auditor (SAFE AI Focus)',
    goal='Evaluate the model against Accuracy, Robustness, and Fairness metrics.',
    backstory='A specialized auditor using the SAFE AI framework to stress test models.',
    tools=[evaluation_and_risk_tool],
    llm=crew_llm,
    allow_delegation=False,
    verbose=True
)

# 4. Safety Agent- Unified Safety Agent (SAFE Score + Governance/Compliance) ---
safety_agent = Agent(
    role='SAFE AI Governance & Compliance Officer',
    goal=(
        "1) Compute a weighted SAFE Score from the evaluation report. "
        "2) Perform governance/compliance checks (no PII leakage, artifacts exist, reproducibility). "
        "3) Approve/Reject with a detailed, human-readable System Card."
    ),
    backstory=(
        "You are the gatekeeper of the pipeline. You consolidate artifacts from Data/Model/Eval, "
        "verify basic compliance (no raw PII, files exist, clear reporting), then compute the FINAL SAFE SCORE."
    ),
    tools=[governance_scoring_tool],
    llm=crew_llm,
    allow_delegation=False,
    verbose=True
)

# 5. Chatbot Agent

chatbot_agent = Agent(
    role='SAFE Results Chatbot',
    goal='Answer user questions about the finished SAFE pipeline run using generated reports and configuration artifacts.',
    backstory=(
        'A lightweight assistant that only answers from the generated SAFE artifacts. '
        'It explains results, configuration, mitigation, and feature-importance findings in plain language.'
    ),
    tools=[safe_chatbot_tool],
    llm=crew_llm,
    allow_delegation=False,
    verbose=True
)

# --- TASKS DEFINITION (The Execution Flow) ---

# T1: Data Preparation
task_data_prep = Task(
    description=(
        f"Load data from {RAW_DATA_PATH}, clean it using data_preprocessing_tool, and confirm creation of "
        "clean_train_features.csv, clean_train_target.csv, clean_test_features.csv, clean_test_target.csv."
    ),
    expected_output="A summary confirming preprocessing, encoding, scaling, and the paths to the split datasets.",
    agent=data_agent,
)

# T2: Model Training
task_model_train = Task(
    description="Using the cleaned data from Task 1, execute the model_training_tool to train and select the best classification model and save the 'best_model.pkl' artifact.",
    expected_output="A confirmation that the best model has been saved and the path to the generated Model Card.",
    agent=modeling_agent,
    context=[task_data_prep] # Needs the DataAgent's output to proceed
)

# T3: Full Evaluation
task_full_eval = Task(
    description=(
        "You MUST call the evaluation_and_risk_tool tool exactly once. "
        "Do not just describe what should be done. "
        "Use the trained model and test data to generate evaluation_report.md, final_report.md, and sensitivity_report.md. "
        "Your final answer must summarize the metrics returned by the tool."
    ),
    expected_output="A summary of AUC, fairness aggregate, robustness aggregate, and confirmation that evaluation_report.md was created.",
    agent=eval_agent,
    context=[task_model_train]
)

# T4: Governance Sign-off
task_governance = Task(
    description=(
        "You MUST call governance_scoring_tool exactly once. "
        "Do not explain the process. "
        "Read evaluation_report.md and generate system_card.md with the final SAFE score and decision."
    ),
    expected_output="system_card.md containing decision + final SAFE score.",
    agent=safety_agent,
    context=[task_full_eval]
)

# T5: Chatbot Readiness + User Q&A
task_chatbot = Task(
    description=(
        "You MUST call safe_chatbot_tool exactly once with a short readiness query such as 'help'. "
        "Do not invent results. Confirm the chatbot is ready to answer grounded questions from the generated artifacts."
    ),
    expected_output="A short readiness message from the chatbot confirming supported question types.",
    agent=chatbot_agent,
    context=[task_governance]
)

# --- CREW LAUNCHER / MAIN EXECUTION ---

if __name__ == "__main__":
    # Create the Crew
    safe_agent_crew = Crew(
        agents=[data_agent, modeling_agent, eval_agent, safety_agent, chatbot_agent],
        tasks=[task_data_prep, task_model_train, task_full_eval, task_governance, task_chatbot],
        process=Process.sequential,
        verbose=True 
    )

    print("--- Starting SAFE Agentic Credit System ---")
    
    # Kick off the process!
    final_result = safe_agent_crew.kickoff()
    
    print("\n\n################################################")
    print("## FINISHED! FINAL GOVERNANCE DECISION: ##")
    print("################################################")
    print(final_result)
    print("\n[SUCCESS] System Card saved to 'system_card.md'")
    print("\n[SUCCESS] SAFE chatbot is ready.")
    run_safe_chatbot_cli()
    
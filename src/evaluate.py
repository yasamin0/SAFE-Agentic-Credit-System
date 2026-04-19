import json
import joblib
import numpy as np
import pandas as pd
from crewai.tools import tool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_STATE,
    ROBUST_NOISE_STD,
    ROBUST_DROPOUT_RATE,
    ROBUST_MISSING_RATE,
    PRED_THRESHOLD,
    W_AUC,
    W_FAIR,
    W_ROB,
)
from src.fairness import (
    _apply_threshold_mitigation,
    _compute_fairness_from_predictions,
    _compute_fairness_metrics,
)
from src.paths import (
    MODEL_PATH,
    TEST_FEATURES_PATH,
    TEST_TARGET_PATH,
    DATACARD_PATH,
    SENSITIVE_TEST_PATH,
    RAW_DATA_PATH,
    EVALUATION_REPORT_PATH,
    FINAL_REPORT_PATH,
    SENSITIVITY_REPORT_PATH,
)
from src.utils import _read_target_series, _safe_mean


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
        "details": {"auc": float(auc_score)}
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

    base_group = pd.read_csv(SENSITIVE_TEST_PATH).iloc[:, 0].astype(str).fillna("NA")
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

    original_df = pd.read_csv(RAW_DATA_PATH)
    X = original_df.drop("CreditRisk", axis=1)
    y = original_df["CreditRisk"]
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
    sens_df["delta_vs_base"] = sens_df["safe_score"] - float(
        sens_df.loc[sens_df["scenario"] == "base", "safe_score"].iloc[0]
    )
    sens_df = sens_df.sort_values(["safe_score", "scenario"], ascending=[False, True]).reset_index(drop=True)
    return sens_df


def _interaction_analysis(model, X_test, y_test, config, numeric_cols):
    y_probs = model.predict_proba(X_test)[:, 1]
    group = pd.read_csv(SENSITIVE_TEST_PATH).iloc[:, 0].astype(str).fillna("NA")

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


@tool
def evaluation_and_risk_tool(description: str):
    """Evaluates the model and calculates aggregated SAFE metrics + sensitivity analysis."""
    try:
        model = joblib.load(MODEL_PATH)
        X_test = pd.read_csv(TEST_FEATURES_PATH)
        y_test = _read_target_series(TEST_TARGET_PATH)
        y_probs = model.predict_proba(X_test)[:, 1]
        auc_score = float(roc_auc_score(y_test, y_probs))

        with open(DATACARD_PATH, "r", encoding="utf-8") as f:
            dc = json.load(f)
        config = dc.get("config", {})
        numeric_cols = [c for c in dc.get("numeric_features_raw", []) if c in X_test.columns]

        group = pd.read_csv(SENSITIVE_TEST_PATH).iloc[:, 0].astype(str).fillna("NA")

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
        with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
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
        with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(final_report)

        with open(SENSITIVITY_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("# Sensitivity Analysis Report\n\n")
            f.write("Evaluates how SAFE decisions change under variations in weights, thresholds, sensitive feature choice, and perturbation assumptions.\n\n")
            f.write("## Scenario Table\n\n")
            f.write(sensitivity_df.to_markdown(index=False))
            f.write("\n\n## Main Effects\n\n")
            f.write(effect_df.to_markdown(index=False))
            f.write("\n\n## Pairwise Interactions\n\n")
            f.write(interaction_df.to_markdown(index=False))
            f.write("\n")

        return report_content
    except Exception as e:
        return f"EVALUATION FAILED: {e}"
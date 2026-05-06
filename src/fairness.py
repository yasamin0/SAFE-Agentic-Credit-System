# src/fairness.py

import numpy as np
import pandas as pd

from src.utils import _safe_mean


def _build_group_table(y_true, y_pred, group):
    """Create group-level rates used by fairness metrics."""
    tmp = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "group": group,
    })

    def tpr(df):
        tp = ((df.y_pred == 1) & (df.y_true == 1)).sum()
        fn = ((df.y_pred == 0) & (df.y_true == 1)).sum()
        return tp / (tp + fn) if (tp + fn) else 0.0

    def fpr(df):
        fp = ((df.y_pred == 1) & (df.y_true == 0)).sum()
        tn = ((df.y_pred == 0) & (df.y_true == 0)).sum()
        return fp / (fp + tn) if (fp + tn) else 0.0

    rows = []

    for group_name, group_df in tmp.groupby("group"):
        rows.append({
            "group": group_name,
            "n": int(len(group_df)),
            "positive_rate": float(group_df["y_pred"].mean()),
            "tpr": float(tpr(group_df)),
            "fpr": float(fpr(group_df)),
        })

    return tmp, pd.DataFrame(rows)


def _perfect_fairness_metrics():
    """Fallback used when only one sensitive group is available."""
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
    }


def _score_fairness_from_group_table(group_table):
    """Convert group-rate gaps into normalized fairness scores."""
    if len(group_table) <= 1:
        return _perfect_fairness_metrics()

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
    }


def _compute_fairness_metrics(y_true, y_probs, group, pred_threshold):
    """Compute fairness metrics from predicted probabilities."""
    y_pred = (y_probs >= pred_threshold).astype(int)
    tmp, group_table = _build_group_table(y_true, y_pred, group)
    metrics = _score_fairness_from_group_table(group_table)

    return metrics, group_table, tmp


def _compute_fairness_from_predictions(y_true, y_pred, group):
    """Compute fairness metrics from already-binarized predictions."""
    _, group_table = _build_group_table(y_true, y_pred, group)
    metrics = _score_fairness_from_group_table(group_table)

    return metrics, group_table


def _apply_threshold_mitigation_search(
    y_true,
    y_probs,
    group,
    base_threshold,
    auc_score,
    robustness_aggregate,
    w_auc,
    w_fair,
    w_rob,
    deltas=None,
):
    """
    Run a group-aware threshold mitigation search.

    The model itself is not retrained.
    The predicted probabilities stay unchanged.
    Only the decision threshold is reduced for the group with the lowest
    baseline positive prediction rate.

    The best candidate is selected by SAFE score.
    """
    if deltas is None:
        deltas = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]

    group = pd.Series(group).astype(str).fillna("NA")
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    baseline_pred = (y_probs >= base_threshold).astype(int)

    baseline_metrics, baseline_group_table = _compute_fairness_from_predictions(
        y_true,
        baseline_pred,
        group,
    )

    baseline_group_rates = baseline_group_table.set_index("group")["positive_rate"]
    disadvantaged_group = str(baseline_group_rates.idxmin())

    rows = []

    for delta in deltas:
        adjusted_threshold = max(0.0, base_threshold - float(delta))

        adjusted_pred = np.array([
            1 if p >= (adjusted_threshold if g == disadvantaged_group else base_threshold) else 0
            for p, g in zip(y_probs, group)
        ])

        mitigated_metrics, mitigated_group_table = _compute_fairness_from_predictions(
            y_true,
            adjusted_pred,
            group,
        )

        positive_rate_gap = float(
            mitigated_group_table["positive_rate"].max()
            - mitigated_group_table["positive_rate"].min()
        )

        # Threshold mitigation changes binary decisions, not probability ranking.
        # Therefore, probability-based AUC remains the baseline AUC.
        safe_score = (
            w_auc * auc_score
            + w_fair * mitigated_metrics["fairness_aggregate"]
            + w_rob * robustness_aggregate
        )

        rows.append({
            "delta": float(delta),
            "base_threshold": float(base_threshold),
            "adjusted_threshold_for_disadvantaged_group": float(adjusted_threshold),
            "disadvantaged_group": disadvantaged_group,
            "auc_probability_based": float(auc_score),
            "fairness_aggregate": float(mitigated_metrics["fairness_aggregate"]),
            "spd_gap": float(mitigated_metrics["spd_gap"]),
            "eod_gap": float(mitigated_metrics["eod_gap"]),
            "aod_gap": float(mitigated_metrics["aod_gap"]),
            "dir_ratio": float(mitigated_metrics["dir_ratio"]),
            "positive_rate_gap": positive_rate_gap,
            "safe_score": float(safe_score),
        })

    search_df = pd.DataFrame(rows)

    # Main criterion: highest SAFE score.
    # Tie-breakers: higher fairness aggregate, then smaller positive-rate gap.
    search_df = search_df.sort_values(
        ["safe_score", "fairness_aggregate", "positive_rate_gap"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    best_row = search_df.iloc[0]
    best_threshold = float(best_row["adjusted_threshold_for_disadvantaged_group"])

    best_pred = np.array([
        1 if p >= (best_threshold if g == disadvantaged_group else base_threshold) else 0
        for p, g in zip(y_probs, group)
    ])

    best_metrics, best_group_table = _compute_fairness_from_predictions(
        y_true,
        best_pred,
        group,
    )

    return {
        "mitigated_pred": best_pred,
        "disadvantaged_group": disadvantaged_group,
        "best_row": best_row,
        "search_df": search_df,
        "baseline_group_table": baseline_group_table,
        "mitigated_group_table": best_group_table,
        "baseline_metrics": baseline_metrics,
        "mitigated_metrics": best_metrics,
    }
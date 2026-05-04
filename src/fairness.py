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


def _apply_threshold_mitigation(y_probs, group, base_threshold=0.55):
    """Lower the threshold for the group with the lowest positive rate."""
    df = pd.DataFrame({
        "prob": y_probs,
        "group": group.astype(str).fillna("NA"),
    })

    group_rates = df.groupby("group")["prob"].apply(
        lambda s: float((s >= base_threshold).mean())
    )

    min_group = group_rates.idxmin()

    adjusted_pred = [
        1 if p >= (base_threshold - 0.05 if g == min_group else base_threshold) else 0
        for p, g in zip(df["prob"], df["group"])
    ]

    return np.array(adjusted_pred), str(min_group)
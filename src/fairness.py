# src/fairness.py

# NumPy is used for numeric operations such as clipping and array creation
import numpy as np

# Pandas is used for grouping rows by sensitive group and computing group-level metrics
import pandas as pd

# Helper for computing the mean of multiple fairness sub-scores safely
from src.utils import _safe_mean


def _build_group_table(y_true, y_pred, group):
    """
    Build a per-group fairness summary table.

    For each sensitive group, this function computes:
    - group size
    - positive prediction rate
    - true positive rate (TPR)
    - false positive rate (FPR)

    These group-level quantities are later used to compute
    fairness gaps such as SPD, EOD, and AOD.
    """
    # Create a unified DataFrame containing true labels, predictions, and group labels
    tmp = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "group": group
    })

    def tpr(df):
        """
        True Positive Rate (TPR):
        among the true positive cases, how many were correctly predicted as positive?
        """
        tp = ((df.y_pred == 1) & (df.y_true == 1)).sum()
        fn = ((df.y_pred == 0) & (df.y_true == 1)).sum()
        return tp / (tp + fn) if (tp + fn) else 0.0

    def fpr(df):
        """
        False Positive Rate (FPR):
        among the true negative cases, how many were incorrectly predicted as positive?
        """
        fp = ((df.y_pred == 1) & (df.y_true == 0)).sum()
        tn = ((df.y_pred == 0) & (df.y_true == 0)).sum()
        return fp / (fp + tn) if (fp + tn) else 0.0

    rows = []

    # Compute fairness-related statistics for each sensitive group separately
    for g, gdf in tmp.groupby("group"):
        rows.append({
            "group": g,
            "n": int(len(gdf)),
            "positive_rate": float(gdf["y_pred"].mean()),
            "tpr": float(tpr(gdf)),
            "fpr": float(fpr(gdf)),
        })

    # Return both the row-level table and the aggregated group table
    return tmp, pd.DataFrame(rows)


def _compute_fairness_metrics(y_true, y_probs, group, pred_threshold):
    """
    Compute fairness metrics starting from predicted probabilities.

    Steps:
    1. Convert probabilities into binary predictions using pred_threshold
    2. Build the per-group fairness table
    3. Compute fairness gap metrics
    4. Convert those gaps into normalized fairness scores
    5. Aggregate the fairness scores into one fairness_aggregate
    """
    # Convert probabilities into binary predictions
    y_pred = (y_probs >= pred_threshold).astype(int)

    # Build group-level statistics
    tmp, group_table = _build_group_table(y_true, y_pred, group)

    # If only one group exists, fairness comparison is not meaningful.
    # In that case, return a default "perfect fairness" placeholder.
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

    # Extract group-level rates
    pos_rates = group_table["positive_rate"]
    tprs = group_table["tpr"]
    fprs = group_table["fpr"]

    # ------------------------------------------------------------
    # FAIRNESS GAP METRICS
    # ------------------------------------------------------------

    # SPD gap: difference between the highest and lowest positive prediction rate
    spd_gap = float(pos_rates.max() - pos_rates.min())

    # EOD gap: difference between the highest and lowest true positive rate
    eod_gap = float(tprs.max() - tprs.min())

    # FPR gap is used as part of AOD
    fpr_gap = float(fprs.max() - fprs.min())

    # AOD gap: average of TPR gap and FPR gap
    aod_gap = float(0.5 * (eod_gap + fpr_gap))

    # DIR ratio: minimum positive rate divided by maximum positive rate
    # This is often used as a fairness ratio rather than a difference
    min_pos = float(pos_rates.min())
    max_pos = float(pos_rates.max())
    dir_ratio = float(min_pos / max_pos) if max_pos > 0 else 1.0

    # ------------------------------------------------------------
    # NORMALIZED FAIRNESS SCORES
    # ------------------------------------------------------------

    # Convert gap metrics into bounded fairness scores in [0, 1]
    # Lower gaps -> higher fairness scores
    fairness_score_spd = float(np.clip(1.0 - spd_gap, 0.0, 1.0))
    fairness_score_eod = float(np.clip(1.0 - eod_gap, 0.0, 1.0))
    fairness_score_aod = float(np.clip(1.0 - aod_gap, 0.0, 1.0))

    # DIR ratio is already ratio-style, so it is clipped directly
    fairness_score_dir = float(np.clip(dir_ratio, 0.0, 1.0))

    # Final fairness aggregate = mean of all fairness sub-scores
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


def _compute_fairness_from_predictions(y_true, y_pred, group):
    """
    Compute fairness metrics starting from already-binarized predictions.

    This is similar to _compute_fairness_metrics(), but it skips the
    probability-to-prediction thresholding step.

    It is mainly used after mitigation, because mitigation directly
    outputs adjusted binary predictions.
    """
    tmp, group_table = _build_group_table(y_true, y_pred, group)

    # Same fallback behavior when fairness comparison is not meaningful
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

    # Extract group-level statistics
    pos_rates = group_table["positive_rate"]
    tprs = group_table["tpr"]
    fprs = group_table["fpr"]

    # Compute fairness gaps
    spd_gap = float(pos_rates.max() - pos_rates.min())
    eod_gap = float(tprs.max() - tprs.min())
    fpr_gap = float(fprs.max() - fprs.min())
    aod_gap = float(0.5 * (eod_gap + fpr_gap))

    # Compute disparate impact ratio
    min_pos = float(pos_rates.min())
    max_pos = float(pos_rates.max())
    dir_ratio = float(min_pos / max_pos) if max_pos > 0 else 1.0

    # Convert gap metrics into normalized fairness scores
    fairness_score_spd = float(np.clip(1.0 - spd_gap, 0.0, 1.0))
    fairness_score_eod = float(np.clip(1.0 - eod_gap, 0.0, 1.0))
    fairness_score_aod = float(np.clip(1.0 - aod_gap, 0.0, 1.0))
    fairness_score_dir = float(np.clip(dir_ratio, 0.0, 1.0))

    # Aggregate fairness sub-scores into one final fairness score
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


def _apply_threshold_mitigation(y_probs, group, base_threshold=0.55):
    """
    Apply a simple post-processing fairness mitigation strategy.

    Idea:
    - Find the group with the lowest positive prediction rate
    - Lower the decision threshold slightly for that group
    - Keep the original threshold for all other groups

    This produces adjusted binary predictions and identifies
    the group that was treated as disadvantaged in the mitigation step.
    """
    # Build a table of probabilities and group labels
    df = pd.DataFrame({
        "prob": y_probs,
        "group": group.astype(str).fillna("NA")
    })

    # Compute positive prediction rate per group under the base threshold
    group_rates = df.groupby("group")["prob"].apply(
        lambda s: float((s >= base_threshold).mean())
    )

    # Select the group with the lowest positive rate as the disadvantaged group
    min_group = group_rates.idxmin()

    adjusted_pred = []

    # Apply a slightly lower threshold only to that disadvantaged group
    for p, g in zip(df["prob"], df["group"]):
        thr = base_threshold - 0.05 if g == min_group else base_threshold
        adjusted_pred.append(1 if p >= thr else 0)

    return np.array(adjusted_pred), str(min_group)
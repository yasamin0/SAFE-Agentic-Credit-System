# src/rga.py

import numpy as np
import pandas as pd

# Use non-GUI backend for saving plots safely in automated runs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score


def _safe_auc(y_true, y_score, fallback=0.5):
    """
    Compute AUC safely.
    If only one class is present, return fallback.
    """
    if len(np.unique(y_true)) < 2:
        return float(fallback)

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float(fallback)


def compute_rga_curve(model, X_test, y_test, fractions=None):
    """
    Compute a paper-style RGA curve.

    The curve is created by progressively removing the highest-ranked
    predictions and recalculating AUC on the remaining samples.
    """
    if fractions is None:
        fractions = np.linspace(0.0, 0.9, 10)

    y_score = model.predict_proba(X_test)[:, 1]
    base_auc = _safe_auc(y_test, y_score)

    order_desc = np.argsort(y_score)[::-1]
    n = len(y_test)

    rows = []

    for frac in fractions:
        remove_count = int(round(n * float(frac)))
        keep_idx = order_desc[remove_count:]

        if len(keep_idx) < 2:
            rga_value = base_auc
        else:
            rga_value = _safe_auc(
                np.asarray(y_test)[keep_idx],
                y_score[keep_idx],
                fallback=base_auc
            )

        rows.append({
            "fraction_removed": float(frac),
            "rga_value": float(rga_value),
        })

    curve_df = pd.DataFrame(rows)

    aurga = float(np.trapz(
        y=curve_df["rga_value"].values,
        x=curve_df["fraction_removed"].values,
    ))

    # Normalize by maximum x-range so AURGA remains comparable.
    max_x = float(curve_df["fraction_removed"].max())
    if max_x > 0:
        aurga = aurga / max_x

    return curve_df, aurga


def save_rga_plot(curve_df, output_path):
    """
    Save the RGA curve plot.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(curve_df["fraction_removed"], curve_df["rga_value"], marker="o")
    plt.xlabel("Fraction of highest-ranked samples removed")
    plt.ylabel("RGA / AUC on remaining data")
    plt.title("RGA Curve — Progressive Data Removal")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
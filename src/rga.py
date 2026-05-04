# src/rga.py

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score


def _safe_auc(y_true, y_score, fallback=0.5):
    """Compute AUC safely when labels contain both classes."""
    if len(np.unique(y_true)) < 2:
        return float(fallback)

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float(fallback)


def compute_rga_curve(model, X_test, y_test, fractions=None, min_remaining_samples=30):
    """
    Compute the RGA curve.

    RGA progressively removes the highest-ranked samples and recalculates
    AUC on the remaining data.
    """
    if fractions is None:
        fractions = np.linspace(0.0, 0.9, 10)

    y_true = np.asarray(y_test)
    y_score = model.predict_proba(X_test)[:, 1]

    base_auc = _safe_auc(y_true, y_score)
    ranked_indices = np.argsort(y_score)[::-1]

    rows = []
    n_samples = len(y_true)

    for frac in fractions:
        remove_count = int(round(n_samples * float(frac)))
        keep_idx = ranked_indices[remove_count:]

        remaining_y = y_true[keep_idx]

        if len(keep_idx) < min_remaining_samples or len(np.unique(remaining_y)) < 2:
            rga_value = np.nan
        else:
            rga_value = _safe_auc(
                remaining_y,
                y_score[keep_idx],
                fallback=np.nan,
            )

        rows.append({
            "fraction_removed": float(frac),
            "remaining_samples": int(len(keep_idx)),
            "rga_value": rga_value,
        })

    curve_df = pd.DataFrame(rows)

    curve_for_auc = curve_df.dropna(subset=["rga_value"])

    if len(curve_for_auc) < 2:
        aurga = float(base_auc)
    else:
        aurga = float(np.trapz(
            y=curve_for_auc["rga_value"].values,
            x=curve_for_auc["fraction_removed"].values,
        ))

        max_x = float(curve_for_auc["fraction_removed"].max())
        if max_x > 0:
            aurga = aurga / max_x

    return curve_df, aurga


def save_rga_plot(curve_df, output_path):
    """Save the RGA curve plot."""
    plot_df = curve_df.dropna(subset=["rga_value"])

    if plot_df.empty:
        plot_df = curve_df.copy()

    plt.figure(figsize=(7, 5))
    plt.plot(
        plot_df["fraction_removed"],
        plot_df["rga_value"],
        marker="o",
    )
    plt.xlabel("Fraction of highest-ranked samples removed")
    plt.ylabel("RGA / AUC on remaining data")
    plt.title("RGA Curve — Progressive Data Removal")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
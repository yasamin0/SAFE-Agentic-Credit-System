# src/compliance.py

import numpy as np
import pandas as pd
# Use non-GUI backend for saving plots safely in automated runs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from src.rga import compute_rga_curve
from src.rgr import compute_rgr_curve
from src.rge import compute_rge_feature_importance, compute_rge_curve


def _compute_topsis(df, metric_cols):
    """
    Compute TOPSIS score for model comparison.
    Higher metric values are assumed to be better.
    """
    values = df[metric_cols].astype(float).values

    denom = np.sqrt((values ** 2).sum(axis=0))
    denom[denom == 0] = 1.0

    normalized = values / denom

    ideal = normalized.max(axis=0)
    anti_ideal = normalized.min(axis=0)

    dist_to_ideal = np.sqrt(((normalized - ideal) ** 2).sum(axis=1))
    dist_to_anti = np.sqrt(((normalized - anti_ideal) ** 2).sum(axis=1))

    return dist_to_anti / (dist_to_ideal + dist_to_anti + 1e-12)


def compute_compliance_scores(metrics_df):
    """
    Compute final Compliance Score using:
    - Arithmetic mean
    - Geometric mean
    - RMS
    - TOPSIS
    """
    df = metrics_df.copy()

    metric_cols = ["AURGA", "AURGR", "AURGE"]

    df["Compliance_Arithmetic"] = df[metric_cols].mean(axis=1)

    df["Compliance_Geometric"] = (
        df[metric_cols].clip(lower=1e-12).prod(axis=1) ** (1.0 / len(metric_cols))
    )

    df["Compliance_RMS"] = np.sqrt((df[metric_cols] ** 2).mean(axis=1))

    df["Compliance_TOPSIS"] = _compute_topsis(df, metric_cols)

    return df.sort_values("Compliance_TOPSIS", ascending=False).reset_index(drop=True)


def run_model_metric_comparison(
    all_model_paths,
    X_test,
    y_test,
    rgr_columns,
    random_state,
):
    """
    Compute AURGA, AURGR, and AURGE for all trained models.
    """
    rows = []

    for model_name, model_path in all_model_paths.items():
        model = joblib.load(model_path)

        _, aurga = compute_rga_curve(model, X_test, y_test)

        _, aurgr_gaussian = compute_rgr_curve(
            model=model,
            X_test=X_test,
            perturbation_type="gaussian",
            columns=rgr_columns,
            random_state=random_state,
        )

        _, aurgr_swapping = compute_rgr_curve(
            model=model,
            X_test=X_test,
            perturbation_type="swapping",
            columns=rgr_columns,
            random_state=random_state,
        )

        rge_importance_df = compute_rge_feature_importance(model, X_test)
        _, aurge = compute_rge_curve(model, X_test, rge_importance_df)

        rows.append({
            "Model": model_name,
            "AURGA": float(aurga),
            "AURGR_Gaussian": float(aurgr_gaussian),
            "AURGR_Swapping": float(aurgr_swapping),
            "AURGR": float(np.mean([aurgr_gaussian, aurgr_swapping])),
            "AURGE": float(aurge),
        })

    return pd.DataFrame(rows)


def save_compliance_plot(compliance_df, output_path):
    """
    Save a bar plot of TOPSIS Compliance Score by model.
    """
    plot_df = compliance_df.sort_values("Compliance_TOPSIS", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["Model"], plot_df["Compliance_TOPSIS"])
    plt.xlabel("Compliance TOPSIS Score")
    plt.ylabel("Model")
    plt.title("SAFE AI Compliance Score Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
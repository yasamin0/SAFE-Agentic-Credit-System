# src/rge.py

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rank_graduation_similarity(reference_scores, comparison_scores):
    """Measure how stable prediction rankings remain after a change."""
    ref = np.asarray(reference_scores, dtype=float)
    comp = np.asarray(comparison_scores, dtype=float)

    n = len(ref)
    if n <= 1:
        return 1.0

    weights = np.arange(1, n + 1, dtype=float)

    order_ref_asc = np.argsort(ref)
    order_ref_desc = order_ref_asc[::-1]
    order_comp_asc = np.argsort(comp)

    denominator = (
        np.sum(weights * ref[order_ref_asc])
        - np.sum(weights * ref[order_ref_desc])
    )

    if np.isclose(denominator, 0.0):
        return 1.0

    numerator = (
        np.sum(weights * ref[order_comp_asc])
        - np.sum(weights * ref[order_ref_desc])
    )

    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _remove_features(X, features_to_remove, replacement_value=0.0):
    """Replace selected processed features with a neutral baseline."""
    X_removed = X.copy()

    for feature in features_to_remove:
        if feature in X_removed.columns:
            X_removed[feature] = replacement_value

    return X_removed


def compute_rge_feature_importance(model, X_test):
    """
    Compute RGE feature importance.

    Importance is defined as:
    importance = 1 - ranking similarity after removing the feature.
    """
    base_scores = model.predict_proba(X_test)[:, 1]
    rows = []

    for feature in X_test.columns:
        X_removed = _remove_features(X_test, [feature])
        removed_scores = model.predict_proba(X_removed)[:, 1]

        similarity = _rank_graduation_similarity(
            reference_scores=base_scores,
            comparison_scores=removed_scores,
        )

        rows.append({
            "feature": feature,
            "rge_similarity_after_removal": float(similarity),
            "rge_importance": float(1.0 - similarity),
        })

    importance_df = pd.DataFrame(rows)

    # Order features from least important to most important.
    importance_df = importance_df.sort_values(
        "rge_importance",
        ascending=True,
    ).reset_index(drop=True)

    importance_df["importance_rank_least_to_most"] = np.arange(
        1,
        len(importance_df) + 1,
    )

    return importance_df


def compute_rge_curve(model, X_test, importance_df):
    """
    Build the RGE curve.

    Features are progressively removed from least important to most important.
    """
    base_scores = model.predict_proba(X_test)[:, 1]
    ordered_features = importance_df["feature"].tolist()

    rows = [{
        "num_removed_features": 0,
        "fraction_removed": 0.0,
        "removed_features": "none",
        "rge_curve_value": 1.0,
    }]

    removed_features = []

    for feature in ordered_features:
        removed_features.append(feature)

        X_removed = _remove_features(X_test, removed_features)
        removed_scores = model.predict_proba(X_removed)[:, 1]

        similarity = _rank_graduation_similarity(
            reference_scores=base_scores,
            comparison_scores=removed_scores,
        )

        rows.append({
            "num_removed_features": len(removed_features),
            "fraction_removed": len(removed_features) / len(ordered_features),
            "removed_features": "; ".join(removed_features),
            "rge_curve_value": float(similarity),
        })

    curve_df = pd.DataFrame(rows)

    # Use np.trapz for compatibility with older NumPy versions.
    aurge = float(np.trapz(
        y=curve_df["rge_curve_value"].values,
        x=curve_df["fraction_removed"].values,
    ))

    return curve_df, aurge


def save_rge_curve_plot(curve_df, output_path):
    """Save the RGE curve plot."""
    plt.figure(figsize=(7, 5))
    plt.plot(
        curve_df["fraction_removed"],
        curve_df["rge_curve_value"],
        marker="o",
    )
    plt.xlabel("Fraction of removed features")
    plt.ylabel("RGE curve value")
    plt.title("RGE Curve — Progressive Feature Removal")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_rge_importance_plot(importance_df, output_path, top_k=15):
    """Save a bar plot of top RGE-important features."""
    top_df = importance_df.sort_values(
        "rge_importance",
        ascending=False,
    ).head(top_k)

    plt.figure(figsize=(9, 6))
    plt.barh(top_df["feature"], top_df["rge_importance"])
    plt.xlabel("RGE importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_k} Features by RGE Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
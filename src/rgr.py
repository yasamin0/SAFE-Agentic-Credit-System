# src/rgr.py

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rank_graduation_similarity(reference_scores, comparison_scores):
    """
    Measure ranking stability between clean and perturbed predictions.

    A score near 1 means the ranking stayed stable.
    A score near 0 means the ranking changed strongly.
    """
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


def apply_gaussian_noise(X, intensity, columns, random_state=42):
    """Add Gaussian noise to selected columns."""
    X_noisy = X.copy()

    if intensity <= 0 or not columns:
        return X_noisy

    rng = np.random.default_rng(random_state)

    for col in columns:
        col_std = float(X_noisy[col].std())

        if np.isclose(col_std, 0.0):
            col_std = 1.0

        noise = rng.normal(
            loc=0.0,
            scale=float(intensity) * col_std,
            size=len(X_noisy),
        )

        X_noisy[col] = X_noisy[col].values + noise

    return X_noisy


def apply_percentile_swapping(X, intensity, columns, random_state=42):
    """
    Swap lower-tail and upper-tail values in selected columns.

    This creates a stronger distributional perturbation than Gaussian noise.
    """
    X_swapped = X.copy()

    if intensity <= 0 or not columns:
        return X_swapped

    rng = np.random.default_rng(random_state)
    percentile = min(max(float(intensity), 0.0), 0.5)

    for col in columns:
        values = X_swapped[col].values.copy()

        low_threshold = np.quantile(values, percentile)
        high_threshold = np.quantile(values, 1.0 - percentile)

        low_idx = np.where(values <= low_threshold)[0]
        high_idx = np.where(values >= high_threshold)[0]

        swap_count = min(len(low_idx), len(high_idx))
        if swap_count == 0:
            continue

        low_sample = rng.choice(low_idx, size=swap_count, replace=False)
        high_sample = rng.choice(high_idx, size=swap_count, replace=False)

        temp = values[low_sample].copy()
        values[low_sample] = values[high_sample]
        values[high_sample] = temp

        X_swapped[col] = values

    return X_swapped


def _apply_perturbation(X_test, perturbation_type, intensity, columns, random_state):
    """Apply one supported perturbation type."""
    if perturbation_type == "gaussian":
        return apply_gaussian_noise(
            X=X_test,
            intensity=float(intensity),
            columns=columns,
            random_state=random_state,
        )

    if perturbation_type == "swapping":
        return apply_percentile_swapping(
            X=X_test,
            intensity=float(intensity),
            columns=columns,
            random_state=random_state,
        )

    raise ValueError("perturbation_type must be 'gaussian' or 'swapping'")


def compute_rgr_curve(
    model,
    X_test,
    perturbation_type,
    columns,
    intensities=None,
    random_state=42,
):
    """
    Compute RGR across increasing perturbation intensities.

    Returns the RGR curve and AURGR.
    """
    if intensities is None:
        intensities = np.linspace(0.0, 0.5, 11)

    base_scores = model.predict_proba(X_test)[:, 1]
    max_intensity = float(max(intensities)) if max(intensities) > 0 else 1.0

    rows = []

    for intensity in intensities:
        X_perturbed = _apply_perturbation(
            X_test=X_test,
            perturbation_type=perturbation_type,
            intensity=intensity,
            columns=columns,
            random_state=random_state,
        )

        perturbed_scores = model.predict_proba(X_perturbed)[:, 1]

        rgr = _rank_graduation_similarity(
            reference_scores=base_scores,
            comparison_scores=perturbed_scores,
        )

        rows.append({
            "intensity": float(intensity),
            "normalized_intensity": float(intensity / max_intensity),
            "rgr": float(rgr),
        })

    curve_df = pd.DataFrame(rows)

    # Use np.trapz for compatibility with older NumPy versions.
    aurgr = float(np.trapz(
        y=curve_df["rgr"].values,
        x=curve_df["normalized_intensity"].values,
    ))

    return curve_df, aurgr


def save_rgr_plot(curve_df, output_path, title):
    """Save an RGR curve plot."""
    plt.figure(figsize=(7, 5))
    plt.plot(
        curve_df["normalized_intensity"],
        curve_df["rgr"],
        marker="o",
    )
    plt.xlabel("Normalized perturbation intensity")
    plt.ylabel("RGR")
    plt.title(title)
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
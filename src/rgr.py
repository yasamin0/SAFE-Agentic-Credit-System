# ------------------------------------------------------------
# src/rgr.py
# ------------------------------------------------------------
# This file implements Rank Graduation Robustness (RGR).
#
# Goal:
# RGR checks whether the ranking of model predictions remains stable
# when the input data is progressively perturbed.
#
# Example:
# - First, the model predicts probabilities on the original clean test set.
# - Then we add noise or swap percentile values in the test set.
# - The model predicts again on the perturbed data.
# - RGR compares the original prediction ranking with the perturbed ranking.
#
# Output:
# - RGR curve for Gaussian noise
# - RGR curve for percentile swapping
# - AURGR = Area Under the RGR Curve
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _rank_graduation_similarity(reference_scores, comparison_scores):
    """
    Compute a rank-based similarity score inspired by Rank Graduation Robustness.

    Parameters
    ----------
    reference_scores:
        Predicted probabilities from the model on the original clean test data.

    comparison_scores:
        Predicted probabilities from the model on perturbed test data.

    Interpretation
    --------------
    - score close to 1.0:
        The perturbed prediction ranking is very similar to the original ranking.

    - score close to 0.5:
        The perturbed prediction ranking behaves approximately like random ranking.

    - score close to 0.0:
        The perturbed prediction ranking is close to the reverse of the original ranking.

    Why ranking?
    ------------
    In credit scoring, the exact probability value is important, but the ranking
    is also very important because customers are often ordered from low risk to
    high risk or from high approval likelihood to low approval likelihood.
    """

    # Convert inputs to NumPy arrays so we can do numerical operations safely.
    ref = np.asarray(reference_scores, dtype=float)
    comp = np.asarray(comparison_scores, dtype=float)

    # Number of test samples.
    n = len(ref)

    # If there is only one sample, ranking is meaningless.
    # In that special case, we return perfect stability.
    if n <= 1:
        return 1.0

    # Weights represent ranking positions: 1, 2, 3, ..., n.
    # These are used to compute weighted rank-based sums.
    weights = np.arange(1, n + 1, dtype=float)

    # Indices that sort the original predictions from low to high.
    order_ref_asc = np.argsort(ref)

    # Indices that sort the original predictions from high to low.
    # This is the reverse ranking.
    order_ref_desc = order_ref_asc[::-1]

    # Indices that sort the perturbed predictions from low to high.
    # This tells us the ranking after perturbation.
    order_comp_asc = np.argsort(comp)

    # Denominator:
    # Difference between the best/original rank structure and the reversed one.
    # This normalizes the score to approximately [0, 1].
    denominator = (
        np.sum(weights * ref[order_ref_asc])
        - np.sum(weights * ref[order_ref_desc])
    )

    # If all predictions are almost identical, the denominator can become zero.
    # In that case, there is no meaningful rank difference, so we return 1.0.
    if np.isclose(denominator, 0.0):
        return 1.0

    # Numerator:
    # Compares the perturbed ranking against the reversed original ranking.
    numerator = (
        np.sum(weights * ref[order_comp_asc])
        - np.sum(weights * ref[order_ref_desc])
    )

    # Final rank similarity score.
    score = numerator / denominator

    # Clip the result to keep it inside [0, 1].
    # This prevents small numerical issues from producing invalid values.
    return float(np.clip(score, 0.0, 1.0))


def apply_gaussian_noise(X, intensity, columns, random_state=42):
    """
    Add Gaussian noise to selected columns.

    Parameters
    ----------
    X:
        Test feature DataFrame.

    intensity:
        Noise intensity.
        - 0.00 means no perturbation.
        - 0.10 means mild perturbation.
        - 0.50 means strong perturbation.

    columns:
        Columns where noise should be applied.
        Usually these are numeric columns.

    random_state:
        Random seed for reproducibility.

    What this does
    --------------
    For each selected column:
    noise ~ Normal(0, intensity * column_standard_deviation)

    Then:
    X_noisy[column] = X[column] + noise
    """

    # Make a copy so the original test data is not modified.
    X_noisy = X.copy()

    # If intensity is zero or no columns are provided, return unchanged data.
    if intensity <= 0 or not columns:
        return X_noisy

    # Create a reproducible random generator.
    rng = np.random.default_rng(random_state)

    # Add noise column by column.
    for col in columns:
        # Compute standard deviation of this column.
        col_std = float(X_noisy[col].std())

        # If the column has zero variance, use 1.0 to avoid zero noise scale.
        if np.isclose(col_std, 0.0):
            col_std = 1.0

        # Generate Gaussian noise for this column.
        noise = rng.normal(
            loc=0.0,
            scale=intensity * col_std,
            size=len(X_noisy)
        )

        # Add noise to the column.
        X_noisy[col] = X_noisy[col].values + noise

    return X_noisy


def apply_percentile_swapping(X, intensity, columns, random_state=42):
    """
    Apply percentile swapping perturbation.

    Parameters
    ----------
    X:
        Test feature DataFrame.

    intensity:
        Swapping intensity.
        Example:
        - 0.10 means swap values between roughly bottom 10% and top 10%.
        - 0.50 means a very strong tail swapping perturbation.

    columns:
        Columns where percentile swapping should be applied.

    random_state:
        Random seed for reproducibility.

    What this does
    --------------
    For each selected column:
    1. Find the lower tail values.
    2. Find the upper tail values.
    3. Swap values between the lower and upper tails.

    Why this matters
    ----------------
    Gaussian noise creates small continuous changes.
    Percentile swapping creates stronger distributional changes because
    low values and high values are exchanged.
    """

    # Make a copy so the original test data is not modified.
    X_swapped = X.copy()

    # If intensity is zero or no columns are provided, return unchanged data.
    if intensity <= 0 or not columns:
        return X_swapped

    # Create a reproducible random generator.
    rng = np.random.default_rng(random_state)

    # Keep intensity inside the valid range [0, 0.5].
    # 0.5 means bottom 50% and top 50%, which is the maximum meaningful swap.
    p = min(max(float(intensity), 0.0), 0.5)

    # Apply swapping column by column.
    for col in columns:
        # Copy column values as a NumPy array.
        values = X_swapped[col].values.copy()

        # Compute lower and upper percentile thresholds.
        low_threshold = np.quantile(values, p)
        high_threshold = np.quantile(values, 1.0 - p)

        # Find row indices in the lower tail.
        low_idx = np.where(values <= low_threshold)[0]

        # Find row indices in the upper tail.
        high_idx = np.where(values >= high_threshold)[0]

        # We can only swap the same number of low and high values.
        swap_count = min(len(low_idx), len(high_idx))

        # If there are no values to swap, skip this column.
        if swap_count == 0:
            continue

        # Randomly sample rows from lower and upper tails.
        low_sample = rng.choice(low_idx, size=swap_count, replace=False)
        high_sample = rng.choice(high_idx, size=swap_count, replace=False)

        # Swap the sampled lower-tail and upper-tail values.
        temp = values[low_sample].copy()
        values[low_sample] = values[high_sample]
        values[high_sample] = temp

        # Save the swapped values back into the DataFrame.
        X_swapped[col] = values

    return X_swapped


def compute_rgr_curve(
    model,
    X_test,
    perturbation_type,
    columns,
    intensities=None,
    random_state=42,
):
    """
    Compute the RGR curve across increasing perturbation intensities.

    Parameters
    ----------
    model:
        Trained classifier with predict_proba().

    X_test:
        Clean test feature DataFrame.

    perturbation_type:
        Either:
        - "gaussian"
        - "swapping"

    columns:
        Columns to perturb.

    intensities:
        List or array of perturbation intensities.
        If None, we use 0.00, 0.05, ..., 0.50.

    random_state:
        Random seed for reproducibility.

    Returns
    -------
    curve_df:
        DataFrame containing:
        - intensity
        - normalized_intensity
        - rgr

    aurgr:
        Area Under the RGR Curve.
    """

    # Default perturbation levels:
    # 0.00, 0.05, 0.10, ..., 0.50
    if intensities is None:
        intensities = np.linspace(0.0, 0.5, 11)

    # Original model predictions on clean test data.
    # These are the reference predictions.
    base_scores = model.predict_proba(X_test)[:, 1]

    # Store RGR values for each perturbation intensity.
    rows = []

    # Compute RGR for every intensity level.
    for intensity in intensities:
        # Apply Gaussian noise perturbation.
        if perturbation_type == "gaussian":
            X_perturbed = apply_gaussian_noise(
                X=X_test,
                intensity=float(intensity),
                columns=columns,
                random_state=random_state,
            )

        # Apply percentile swapping perturbation.
        elif perturbation_type == "swapping":
            X_perturbed = apply_percentile_swapping(
                X=X_test,
                intensity=float(intensity),
                columns=columns,
                random_state=random_state,
            )

        # Reject invalid perturbation type.
        else:
            raise ValueError("perturbation_type must be 'gaussian' or 'swapping'")

        # Model predictions after perturbation.
        perturbed_scores = model.predict_proba(X_perturbed)[:, 1]

        # Compare original ranking with perturbed ranking.
        rgr = _rank_graduation_similarity(
            reference_scores=base_scores,
            comparison_scores=perturbed_scores,
        )

        # Save one row of the RGR curve.
        rows.append({
            "intensity": float(intensity),
            "normalized_intensity": float(intensity / max(intensities)),
            "rgr": float(rgr),
        })

    # Convert results into a DataFrame.
    curve_df = pd.DataFrame(rows)

    # AURGR:
    # Area under the RGR curve using trapezoidal integration.
    # x-axis: normalized perturbation intensity
    # y-axis: RGR score
    aurgr = float(np.trapz(
        y=curve_df["rgr"].values,
        x=curve_df["normalized_intensity"].values,
    ))

    return curve_df, aurgr


def save_rgr_plot(curve_df, output_path, title):
    """
    Save an RGR curve plot as a PNG file.

    Parameters
    ----------
    curve_df:
        DataFrame returned by compute_rgr_curve().

    output_path:
        Where to save the plot.

    title:
        Plot title.
    """

    # Create a new figure.
    plt.figure(figsize=(7, 5))

    # Plot RGR against normalized perturbation intensity.
    plt.plot(
        curve_df["normalized_intensity"],
        curve_df["rgr"],
        marker="o"
    )

    # Label the x-axis.
    plt.xlabel("Normalized perturbation intensity")

    # Label the y-axis.
    plt.ylabel("RGR")

    # Add title.
    plt.title(title)

    # Keep y-axis inside valid RGR range.
    plt.ylim(0.0, 1.05)

    # Add a light grid to make the curve easier to read.
    plt.grid(True, alpha=0.3)

    # Prevent label/title overlap.
    plt.tight_layout()

    # Save the plot.
    plt.savefig(output_path, dpi=200)

    # Close the figure to avoid memory issues when running many plots.
    plt.close()
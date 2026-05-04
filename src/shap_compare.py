# src/shap_compare.py

import numpy as np
import pandas as pd


def compare_rge_with_shap(
    model,
    X_test,
    rge_importance_df,
    output_csv_path,
    output_report_path,
    sample_size=200,
):
    """
    Compare RGE feature ranking with SHAP ranking.

    This version uses XGBoost's built-in SHAP contribution method
    instead of shap.TreeExplainer. It is more stable for this project
    because the governance model is XGBoost.
    """
    try:
        import xgboost as xgb

        X_sample = X_test.copy()

        # Use a small sample to keep SHAP computation fast.
        if len(X_sample) > sample_size:
            X_sample = X_sample.sample(sample_size, random_state=42)

        # Convert all columns to numeric to avoid string conversion errors.
        X_sample = X_sample.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # XGBoost built-in SHAP values.
        # pred_contribs=True returns one contribution per feature plus one bias column.
        booster = model.get_booster()
        dmatrix = xgb.DMatrix(X_sample)
        shap_contribs = booster.predict(dmatrix, pred_contribs=True)

        # Last column is the bias term, so we remove it.
        shap_values = shap_contribs[:, :-1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        shap_df = pd.DataFrame({
            "feature": X_sample.columns,
            "mean_abs_shap": mean_abs_shap,
        })

        merged = rge_importance_df.merge(shap_df, on="feature", how="inner")

        merged["rge_rank"] = merged["rge_importance"].rank(
            ascending=False,
            method="average"
        )

        merged["shap_rank"] = merged["mean_abs_shap"].rank(
            ascending=False,
            method="average"
        )

        spearman_corr = float(
            merged["rge_importance"].corr(
                merged["mean_abs_shap"],
                method="spearman"
            )
        )

        merged = merged.sort_values(
            "rge_importance",
            ascending=False
        ).reset_index(drop=True)

        merged.to_csv(output_csv_path, index=False)

        with open(output_report_path, "w", encoding="utf-8") as f:
            f.write("# RGE vs SHAP Comparison Report\n\n")
            f.write("This report compares RGE feature ranking with SHAP feature ranking.\n\n")
            f.write("SHAP values were computed using XGBoost built-in TreeSHAP contributions.\n\n")
            f.write(f"- Spearman correlation: {spearman_corr:.4f}\n")
            f.write(f"- Compared features: {len(merged)}\n\n")

            f.write("## Top Features by RGE\n\n")
            f.write(
                merged[
                    ["feature", "rge_importance", "mean_abs_shap", "rge_rank", "shap_rank"]
                ].head(15).to_markdown(index=False)
            )
            f.write("\n\n")

            f.write("## Top Features by SHAP\n\n")
            f.write(
                merged.sort_values("mean_abs_shap", ascending=False)[
                    ["feature", "rge_importance", "mean_abs_shap", "rge_rank", "shap_rank"]
                ].head(15).to_markdown(index=False)
            )
            f.write("\n")

        return {
            "status": "completed",
            "spearman_corr": spearman_corr,
            "compared_features": int(len(merged)),
        }

    except Exception as e:
        with open(output_report_path, "w", encoding="utf-8") as f:
            f.write("# RGE vs SHAP Comparison Report\n\n")
            f.write("SHAP comparison could not be completed.\n\n")
            f.write(f"Reason: {e}\n")

        return {
            "status": "failed",
            "spearman_corr": None,
            "compared_features": 0,
            "error": str(e),
        }
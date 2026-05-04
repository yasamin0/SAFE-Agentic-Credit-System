# Rank Graduation Robustness Report

This report implements the paper-style RGR robustness analysis.

## Method
- Original model predictions are computed on the clean test set.
- Perturbation intensity is increased step by step.
- At each intensity level, predictions are recomputed.
- RGR measures how much the perturbed prediction ranking preserves the original ranking.
- AURGR is the area under the RGR curve.

## Results
- AURGR Gaussian Noise: 0.9664
- AURGR Percentile Swapping: 0.8162
- RGR Aggregate: 0.8913

## Output Files
- Gaussian curve CSV: rgr_gaussian_curve.csv
- Swapping curve CSV: rgr_swapping_curve.csv
- Gaussian curve plot: rgr_gaussian_curve.png
- Swapping curve plot: rgr_swapping_curve.png

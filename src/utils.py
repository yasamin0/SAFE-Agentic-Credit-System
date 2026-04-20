# src/utils.py

# Standard library imports
import os
import json
import re

# Numerical and tabular utilities
import numpy as np
import pandas as pd


def _safe_mean(values):
    """
    Return the mean of a list of values safely.

    If the input list is empty, return 0.0 instead of raising an error.
    This is useful when aggregating metrics such as fairness or robustness
    sub-scores.
    """
    return float(np.mean(values)) if len(values) else 0.0


def _read_target_series(path):
    """
    Read a target CSV file and return it as a flattened 1D NumPy array.

    This is used for files like clean_train_target.csv or clean_test_target.csv,
    where the target is stored as a single-column CSV.
    """
    df = pd.read_csv(path)
    return df.iloc[:, 0].values.ravel()


def _safe_read_text(path):
    """
    Safely read a text file.

    If the file exists, return its text content.
    If it does not exist, return an empty string instead of failing.
    This is useful for optional artifacts such as reports or cards.
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def _safe_read_json(path):
    """
    Safely read a JSON file.

    If the file exists, return the parsed JSON object.
    If it does not exist, return an empty dictionary.
    This is useful for optional artifacts such as datacard.json.
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _extract_markdown_metric(text, label):
    """
    Extract the value of a named metric from markdown-like report text.

    This helper supports multiple markdown patterns because generated reports
    may format values in slightly different ways, for example:
    - **Label**: value
    - Label: value
    - heading + bold value

    Returns:
        str | None: the extracted metric value as text, or None if not found
    """
    safe_label = re.escape(label)

    # Different possible markdown/report layouts supported by this parser
    patterns = [
        rf"- \*\*{safe_label}\*\*:\s*([^\n]+)",
        rf"- {safe_label}:\s*([^\n]+)",
        rf"\*\*{safe_label}\*\*\s*\n\s*\*\*([^\n]+)\*\*",
        rf"#+\s*{safe_label}\s*\n\s*\*\*([^\n]+)\*\*",
        rf"#+\s*{safe_label}\s*\n\s*([^\n*][^\n]*)",
    ]

    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    return None


def _extract_top_features(report_text, k=5):
    """
    Extract the top-k feature-importance rows from the markdown table
    inside the final SAFE report.

    The function looks for the section:
    'Top 10 most important processed features:'
    and then parses the markdown table that follows it.

    Args:
        report_text (str): full text of the report
        k (int): number of top features to return

    Returns:
        list[tuple[str, str]]: list of (feature_name, importance_value)
    """
    lines = report_text.splitlines()
    start = None

    # Find the start of the feature-importance section
    for i, line in enumerate(lines):
        if "Top 10 most important processed features:" in line:
            start = i + 2
            break

    if start is None:
        return []

    rows = []

    # Parse markdown table rows until the table ends
    for line in lines[start:]:
        if not line.strip().startswith("|"):
            break
        if "---" in line or "feature" in line.lower():
            continue

        parts = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(parts) >= 2:
            rows.append((parts[0], parts[1]))

        if len(rows) >= k:
            break

    return rows


def _safe_str(x):
    """
    Convert a value to string safely.

    If the value is None, return 'N/A' instead.
    This is especially useful when building chatbot answers from artifacts
    that may have missing fields.
    """
    return "N/A" if x is None else str(x)
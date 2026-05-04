# src/utils.py

import json
import os
import re

import numpy as np
import pandas as pd


def _safe_mean(values):
    """Return mean safely; return 0.0 for empty inputs."""
    return float(np.mean(values)) if len(values) else 0.0


def _read_target_series(path):
    """Read a single-column target CSV as a flat NumPy array."""
    df = pd.read_csv(path)
    return df.iloc[:, 0].values.ravel()


def _safe_read_text(path):
    """Read a text file if it exists; otherwise return an empty string."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    return ""


def _safe_read_json(path):
    """Read a JSON file if it exists; otherwise return an empty dictionary."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def _extract_markdown_metric(text, label):
    """
    Extract a metric value from markdown-like report text.

    Supports common formats such as:
    - **Label**: value
    - Label: value
    - heading followed by a bold value
    """
    safe_label = re.escape(label)

    patterns = [
        rf"- \*\*{safe_label}\*\*:\s*([^\n]+)",
        rf"- {safe_label}:\s*([^\n]+)",
        rf"\*\*{safe_label}\*\*\s*\n\s*\*\*([^\n]+)\*\*",
        rf"#+\s*{safe_label}\s*\n\s*\*\*([^\n]+)\*\*",
        rf"#+\s*{safe_label}\s*\n\s*([^\n*][^\n]*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def _extract_top_features(report_text, k=5):
    """
    Extract top feature-importance rows from the final report table.

    Looks for the section:
    'Top 10 most important processed features:'
    """
    lines = report_text.splitlines()
    start = None

    for i, line in enumerate(lines):
        if "Top 10 most important processed features:" in line:
            start = i + 2
            break

    if start is None:
        return []

    rows = []

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


def _safe_str(value):
    """Convert None to N/A; otherwise convert value to string."""
    return "N/A" if value is None else str(value)
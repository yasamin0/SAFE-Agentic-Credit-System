import os
import json
import re
import numpy as np
import pandas as pd


def _safe_mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def _read_target_series(path):
    df = pd.read_csv(path)
    return df.iloc[:, 0].values.ravel()


def _safe_read_text(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def _safe_read_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _extract_markdown_metric(text, label):
    safe_label = re.escape(label)
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


def _safe_str(x):
    return "N/A" if x is None else str(x)
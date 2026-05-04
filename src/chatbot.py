# src/chatbot.py

import os
import re
from pathlib import Path
from textwrap import dedent
from datetime import datetime

import pandas as pd
from crewai.tools import tool

from src.config import current_config, crew_llm

from src.paths import (
    DATACARD_PATH,
    MODEL_CARD_PATH,
    SYSTEM_CARD_PATH,
    EVALUATION_REPORT_PATH,
    FINAL_REPORT_PATH,
    SENSITIVITY_REPORT_PATH,
    CHATBOT_LOG_PATH,
)

from src.utils import (
    _safe_read_text,
    _safe_read_json,
    _extract_markdown_metric,
)


CHATBOT_HISTORY = []

SAMPLE_QUESTIONS = [
    "Why was the model approved or rejected?",
    "What is the weakest dimension?",
    "Which model has the best compliance?",
    "Which variables are most important?",
    "How robust is the model to noise?",
    "What does the model card say about training?",
    "Does the report mention calibration?",
]


def _extract_system_card_field(text, field_name):
    """
    Extract fields from system_card.md.

    Supports:
    **Decision:** REJECTED
    **Final SAFE Score:** 0.692
    """
    if not text:
        return None

    pattern_inline = rf"\*\*{re.escape(field_name)}:\*\*\s*([^\n]+)"
    match = re.search(pattern_inline, text)
    if match:
        return match.group(1).strip()

    pattern_inline_alt = rf"\*\*{re.escape(field_name)}\*\*:\s*([^\n]+)"
    match = re.search(pattern_inline_alt, text)
    if match:
        return match.group(1).strip()

    pattern_section = rf"##\s+{re.escape(field_name)}\s*\n\s*\*\*([^\*]+)\*\*"
    match = re.search(pattern_section, text)
    if match:
        return match.group(1).strip()

    return None


def _read_csv_as_markdown(path, max_rows=30):
    """
    Read a generated CSV artifact and return a compact markdown preview.
    """
    try:
        if Path(path).exists():
            df = pd.read_csv(path)
            return df.head(max_rows).to_markdown(index=False)
    except Exception as e:
        return f"Could not read {Path(path).name}: {e}"

    return ""


def build_chatbot_context():
    """
    Load all generated SAFE artifacts used by the chatbot.

    The chatbot is grounded only in generated files.
    """
    config = _safe_read_json(DATACARD_PATH).get("config", current_config())

    system_card = _safe_read_text(SYSTEM_CARD_PATH)
    evaluation_report = _safe_read_text(EVALUATION_REPORT_PATH)
    final_report = _safe_read_text(FINAL_REPORT_PATH)
    sensitivity_report = _safe_read_text(SENSITIVITY_REPORT_PATH)
    model_card = _safe_read_text(MODEL_CARD_PATH)

    model_comparison_report = _safe_read_text("reports/model_comparison_report.md")
    cv_results = _read_csv_as_markdown("reports/cv_results.csv", max_rows=40)

    csv_artifacts = {
        "classification_metrics.csv": _read_csv_as_markdown("reports/classification_metrics.csv"),
        "confusion_matrix.csv": _read_csv_as_markdown("reports/confusion_matrix.csv"),
        "calibration_curve.csv": _read_csv_as_markdown("reports/calibration_curve.csv"),
        "rga_curve.csv": _read_csv_as_markdown("reports/rga_curve.csv"),
        "rgr_gaussian_curve.csv": _read_csv_as_markdown("reports/rgr_gaussian_curve.csv"),
        "rgr_swapping_curve.csv": _read_csv_as_markdown("reports/rgr_swapping_curve.csv"),
        "rge_feature_importance.csv": _read_csv_as_markdown("reports/rge_feature_importance.csv", max_rows=25),
        "rge_curve.csv": _read_csv_as_markdown("reports/rge_curve.csv"),
        "shap_rge_comparison.csv": _read_csv_as_markdown("reports/shap_rge_comparison.csv", max_rows=25),
        "model_metrics_comparison.csv": _read_csv_as_markdown("reports/model_metrics_comparison.csv"),
        "compliance_score_comparison.csv": _read_csv_as_markdown("reports/compliance_score_comparison.csv"),
    }

    markdown_artifacts = {
        "system_card.md": system_card,
        "evaluation_report.md": evaluation_report,
        "final_report.md": final_report,
        "sensitivity_report.md": sensitivity_report,
        "model_card.md": model_card,
        "model_comparison_report.md": model_comparison_report,
        "rga_report.md": _safe_read_text("reports/rga_report.md"),
        "rgr_report.md": _safe_read_text("reports/rgr_report.md"),
        "rge_report.md": _safe_read_text("reports/rge_report.md"),
        "shap_rge_report.md": _safe_read_text("reports/shap_rge_report.md"),
        "safe_paper_metrics_report.md": _safe_read_text("reports/safe_paper_metrics_report.md"),
        "outlier_analysis_report.md": _safe_read_text("reports/outlier_analysis_report.md"),
    }

    return {
        "config": config,

        "decision": (
            _extract_system_card_field(system_card, "Decision")
            or _extract_system_card_field(system_card, "Final Governance Decision")
            or _extract_markdown_metric(system_card, "Decision")
        ),
        "final_safe_score": (
            _extract_system_card_field(system_card, "Final SAFE Score")
            or _extract_markdown_metric(system_card, "Final SAFE Score")
        ),
        "approval_threshold": config.get("approval_threshold"),

        "auc": _extract_markdown_metric(evaluation_report, "Accuracy (AUC)"),
        "fairness_aggregate": _extract_markdown_metric(evaluation_report, "Fairness Aggregate"),
        "robustness_aggregate": _extract_markdown_metric(evaluation_report, "Robustness Aggregate"),

        "markdown_artifacts": markdown_artifacts,
        "csv_artifacts": csv_artifacts,
        "cv_results": cv_results,
    }


def _format_chat_history(chat_history, max_turns=6):
    """
    Convert recent chat history into compact text.
    """
    if not chat_history:
        return "No prior conversation."

    trimmed = chat_history[-max_turns:]
    lines = []

    for turn in trimmed:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")

    return "\n".join(lines)


def _split_query_terms(query):
    """
    Convert the user question into useful retrieval terms.
    """
    stopwords = {
        "the", "is", "are", "was", "were", "what", "which", "why", "how",
        "does", "do", "did", "about", "from", "with", "this", "that",
        "tell", "me", "please", "give", "show", "explain", "say", "says",
        "has", "have", "had", "been", "into", "for", "and", "or", "to",
        "of", "in", "on", "a", "an",
    }

    cleaned = (
        query.lower()
        .replace("?", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace("-", " ")
        .replace("_", " ")
    )

    terms = []

    for raw in cleaned.split():
        term = raw.strip()

        if len(term) <= 2:
            continue

        if term in stopwords:
            continue

        terms.append(term)

    return terms


def _expand_query_terms(query, terms):
    """
    Add domain synonyms so retrieval works for natural questions.
    """
    q = query.lower()
    expanded = set(terms)

    if any(x in q for x in ["weakest", "weakness", "lowest", "dimension"]):
        expanded.update([
            "weakest",
            "fairness",
            "aggregate",
            "auc",
            "robustness",
            "decision",
            "safe",
        ])

    if any(x in q for x in ["compliance", "topsis", "best model", "best compliance"]):
        expanded.update([
            "compliance",
            "topsis",
            "arithmetic",
            "geometric",
            "rms",
            "aurga",
            "aurgr",
            "aurge",
            "model",
        ])

    if any(x in q for x in ["important", "variables", "features", "explainability"]):
        expanded.update([
            "feature",
            "features",
            "importance",
            "rge",
            "shap",
            "rge_importance",
            "mean_abs_shap",
            "duration",
            "credit_amount",
        ])

    if any(x in q for x in ["noise", "robust", "robustness", "gaussian"]):
        expanded.update([
            "noise",
            "gaussian",
            "robustness",
            "rgr",
            "aurgr",
            "dropout",
            "missingness",
        ])

    if any(x in q for x in ["calibration", "brier", "probability"]):
        expanded.update([
            "calibration",
            "brier",
            "probability",
            "fraction",
            "positives",
            "curve",
        ])

    if any(x in q for x in ["training", "trained", "model card", "cross validation", "cv"]):
        expanded.update([
            "training",
            "trained",
            "cross",
            "validation",
            "cv",
            "best_cv_auc",
            "hyperparameter",
            "model",
        ])

    if any(x in q for x in ["approved", "rejected", "decision", "safe score"]):
        expanded.update([
            "decision",
            "rejected",
            "approved",
            "safe",
            "score",
            "threshold",
            "fairness",
        ])

    return list(expanded)


def _score_line(line, query_terms):
    """
    Score one artifact line by query-term overlap.
    """
    line_lower = line.lower()
    return sum(1 for term in query_terms if term.lower() in line_lower)


def _include_artifact_by_intent(query, artifact_name):
    """
    Include important whole-table artifacts for known broad intents.

    This is not answering by shortcut. It only decides which generated files
    should be shown to the LLM as evidence.
    """
    q = query.lower()
    name = artifact_name.lower()

    if "compliance" in q or "topsis" in q:
        return name in [
            "compliance_score_comparison.csv",
            "model_metrics_comparison.csv",
            "safe_paper_metrics_report.md",
        ]

    if "important" in q or "variables" in q or "features" in q or "explainability" in q:
        return name in [
            "rge_feature_importance.csv",
            "shap_rge_comparison.csv",
            "rge_report.md",
            "shap_rge_report.md",
        ]

    if "noise" in q or "robust" in q or "gaussian" in q:
        return name in [
            "evaluation_report.md",
            "rgr_report.md",
            "rgr_gaussian_curve.csv",
            "rgr_swapping_curve.csv",
            "safe_paper_metrics_report.md",
        ]

    if "calibration" in q or "brier" in q:
        return name in [
            "evaluation_report.md",
            "classification_metrics.csv",
            "calibration_curve.csv",
            "final_report.md",
        ]

    if "training" in q or "model card" in q or "cross-validation" in q or "cross validation" in q:
        return name in [
            "model_card.md",
            "model_comparison_report.md",
            "cv_results.csv",
        ]

    if "weakest" in q or "decision" in q or "rejected" in q or "approved" in q:
        return name in [
            "system_card.md",
            "evaluation_report.md",
            "final_report.md",
        ]

    return False


def _retrieve_artifact_evidence(query, ctx, max_lines=90):
    """
    Retrieve relevant evidence from generated artifacts.

    This is the main RAG retrieval step.
    """
    terms = _split_query_terms(query)
    terms = _expand_query_terms(query, terms)

    if not terms:
        return ""

    artifact_texts = {}

    artifact_texts.update(ctx.get("markdown_artifacts", {}))
    artifact_texts.update(ctx.get("csv_artifacts", {}))

    if ctx.get("cv_results"):
        artifact_texts["cv_results.csv"] = ctx["cv_results"]

    selected_blocks = []
    scored_lines = []

    for artifact_name, text in artifact_texts.items():
        if not text:
            continue

        # For important artifacts, include the first part of the whole artifact/table.
        if _include_artifact_by_intent(query, artifact_name):
            preview_lines = text.splitlines()[:35]
            selected_blocks.append(
                f"\n### {artifact_name}\n" + "\n".join(preview_lines)
            )

        # Also score individual lines for more focused evidence.
        for line in text.splitlines():
            clean_line = line.strip()

            if not clean_line:
                continue

            score = _score_line(clean_line, terms)

            if score > 0:
                scored_lines.append((score, artifact_name, clean_line))

    scored_lines = sorted(scored_lines, key=lambda x: x[0], reverse=True)

    evidence_lines = []

    if selected_blocks:
        evidence_lines.extend(selected_blocks)

    for score, artifact_name, line in scored_lines[:max_lines]:
        evidence_lines.append(f"[{artifact_name}] {line}")

    if not evidence_lines:
        return ""

    return "\n".join(evidence_lines)


def _build_rag_prompt(query, evidence, chat_history):
    """
    Build a grounded RAG prompt for the LLM.
    """
    return dedent(f"""
    You are a SAFE AI results chatbot.

    You MUST answer only using the retrieved evidence below.
    Do not invent numbers, files, experiments, model behavior, or conclusions.
    If the evidence is insufficient, say:
    "I could not find enough evidence in the generated artifacts."

    Recent conversation:
    {_format_chat_history(chat_history)}

    Retrieved evidence from generated artifacts:
    {evidence}

    User question:
    {query}

    Instructions:
    - Answer clearly and directly.
    - Mention the relevant numbers when available.
    - Explain the result in simple words.
    - If multiple generated artifacts disagree, say that the artifacts are inconsistent.
    - Do not use outside knowledge.
    """).strip()


def append_chatbot_log(user_query, assistant_answer, log_path=CHATBOT_LOG_PATH):
    """
    Save one chatbot exchange into the Markdown log.
    """
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("## User\n")
        f.write(f"{user_query}\n\n")
        f.write("## Assistant\n")
        f.write(f"{assistant_answer}\n\n")
        f.write("---\n\n")


def _run_sample_qa():
    """
    Run sample questions and save Q&A for the report/paper.
    """
    output_path = Path("reports") / "chatbot_sample_qa.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# SAFE Chatbot Sample Q&A\n\n")
        f.write("These examples demonstrate artifact-grounded chatbot answers.\n\n")

        for question in SAMPLE_QUESTIONS:
            answer = answer_safe_chatbot_query(question)

            f.write(f"## Question\n{question}\n\n")
            f.write(f"## Answer\n{answer}\n\n")
            f.write("---\n\n")

    return f"Sample Q&A saved to {output_path}."


def answer_safe_chatbot_query(query: str) -> str:
    """
    Main chatbot answering function.

    This version avoids hard-coded question shortcuts.
    It retrieves evidence from generated artifacts and lets the LLM answer
    only from that evidence.
    """
    global CHATBOT_HISTORY

    ctx = build_chatbot_context()

    q = (query or "").strip()
    q_lower = q.lower()

    required_files = [
        str(SYSTEM_CARD_PATH),
        str(EVALUATION_REPORT_PATH),
        str(FINAL_REPORT_PATH),
    ]

    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        answer = (
            "CHATBOT ERROR: Missing required artifacts: " + ", ".join(missing) + ". "
            "Run the full pipeline first so the chatbot can answer grounded questions."
        )
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    # Only basic commands are handled directly.
    if q_lower in ["hello", "hi", "hey", "start", "help"]:
        answer = (
            "SAFE chatbot is ready. I answer only from generated project artifacts.\n\n"
            "Sample questions:\n"
            + "\n".join(f"- {question}" for question in SAMPLE_QUESTIONS)
            + "\n\nType `run sample qa` to save sample Q&A for the report."
        )
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    if q_lower in ["run sample qa", "save sample qa", "sample qa"]:
        answer = _run_sample_qa()
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    evidence = _retrieve_artifact_evidence(q, ctx)

    if not evidence.strip():
        answer = (
            "I could not find enough evidence in the generated artifacts to answer this question. "
            "I am restricted to saved SAFE reports, model cards, system card, and generated CSV outputs."
        )
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    prompt = _build_rag_prompt(
        query=q,
        evidence=evidence,
        chat_history=CHATBOT_HISTORY,
    )

    llm_answer = crew_llm.call(prompt)

    answer = llm_answer.strip() if llm_answer else (
        "I could not produce a grounded answer from the retrieved artifacts."
    )

    CHATBOT_HISTORY.append({"user": q, "assistant": answer})
    append_chatbot_log(q, answer)

    return answer


@tool
def safe_chatbot_tool(query: str):
    """
    CrewAI-compatible wrapper around the SAFE chatbot.
    """
    try:
        return answer_safe_chatbot_query(query)
    except Exception as e:
        return f"CHATBOT FAILED: {e}"


def run_safe_chatbot_cli():
    """
    Interactive command-line SAFE chatbot.
    """
    print("\n--- SAFE Chatbot ---")
    print("Ask grounded questions about generated SAFE artifacts.")
    print("Type 'help' for examples, 'clear' to reset history, or 'exit' to stop.\n")

    while True:
        try:
            user_query = input("SAFE Chatbot > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting SAFE Chatbot.")
            break

        if user_query.lower() in {"exit", "quit", "q"}:
            print("Exiting SAFE Chatbot.")
            break

        if user_query.lower() == "clear":
            CHATBOT_HISTORY.clear()
            print("Chat history cleared.\n")
            continue

        if not user_query:
            continue

        answer = answer_safe_chatbot_query(user_query)
        print(f"\n{answer}\n")
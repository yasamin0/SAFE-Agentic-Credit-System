# src/chatbot.py

# Standard library imports:
# - os: used to check whether required artifact files exist
# - dedent: used to format the grounded LLM prompt more cleanly
import os
from textwrap import dedent

# CrewAI tool decorator so the chatbot can also be used as a Crew tool
from crewai.tools import tool

# Project-level configuration:
# - current_config(): returns the current SAFE pipeline configuration
# - crew_llm: shared LLM used for grounded question answering
from src.config import current_config, crew_llm

# Centralized artifact paths used by the chatbot
from src.paths import (
    DATACARD_PATH,
    MODEL_CARD_PATH,
    SYSTEM_CARD_PATH,
    EVALUATION_REPORT_PATH,
    FINAL_REPORT_PATH,
    SENSITIVITY_REPORT_PATH,
    CHATBOT_LOG_PATH,
)

# Helper utilities for safely reading files and extracting values
from src.utils import (
    _safe_read_text,
    _safe_read_json,
    _extract_markdown_metric,
    _extract_top_features,
    _safe_str,
)

# Stores recent conversation turns for the current chatbot session
# This is used both for context and for logging
CHATBOT_HISTORY = []


def build_chatbot_context():
    """
    Build a grounded context dictionary from the generated SAFE artifacts.

    The chatbot does not answer from raw imagination. Instead, it reads the
    saved outputs of the pipeline (data card, system card, evaluation report,
    final report, sensitivity report, and model card) and extracts the fields
    that are most useful for answering user questions.
    """
    # Load config from the data card if available; otherwise fall back to current config
    config = _safe_read_json(DATACARD_PATH).get("config", current_config())

    # Read the main report artifacts as plain text
    system_card = _safe_read_text(SYSTEM_CARD_PATH)
    evaluation_report = _safe_read_text(EVALUATION_REPORT_PATH)
    final_report = _safe_read_text(FINAL_REPORT_PATH)
    sensitivity_report = _safe_read_text(SENSITIVITY_REPORT_PATH)
    model_card = _safe_read_text(MODEL_CARD_PATH)

    # Return both raw texts and extracted summary values
    return {
        "config": config,
        "system_card": system_card,
        "evaluation_report": evaluation_report,
        "final_report": final_report,
        "sensitivity_report": sensitivity_report,
        "model_card": model_card,

        # Core final outputs
        "decision": _extract_markdown_metric(system_card, "Decision"),
        "final_safe_score": _extract_markdown_metric(system_card, "Final SAFE Score"),

        # Main evaluation metrics
        "auc": _extract_markdown_metric(evaluation_report, "Accuracy (AUC)"),
        "fairness_aggregate": _extract_markdown_metric(evaluation_report, "Fairness Aggregate"),
        "robustness_aggregate": _extract_markdown_metric(evaluation_report, "Robustness Aggregate"),

        # Mitigation-related metrics
        "mitigated_safe_score": _extract_markdown_metric(evaluation_report, "Mitigated SAFE Score"),
        "mitigated_auc": _extract_markdown_metric(evaluation_report, "Mitigated AUC"),
        "baseline_safe_score": _extract_markdown_metric(evaluation_report, "Baseline SAFE Score"),
        "mitigated_fairness_aggregate": _extract_markdown_metric(evaluation_report, "Mitigated Fairness Aggregate"),
        "mitigation_group": _extract_markdown_metric(evaluation_report, "Mitigation Applied To Group"),

        # Short explainability summary
        "top_features": _extract_top_features(final_report, k=5),

        # Important config values the chatbot may need to explain
        "approval_threshold": config.get("approval_threshold"),
        "prediction_threshold": config.get("prediction_threshold"),
        "weights": config.get("weights"),
        "sensitive_feature": config.get("sensitive_feature"),
    }


def _build_baseline_vs_mitigated_summary(ctx):
    """
    Build a short comparison between the baseline run and the mitigated run.
    This is used for questions such as:
    - compare baseline vs mitigated
    - what changed after mitigation?
    """
    return (
        f"Baseline SAFE score: {_safe_str(ctx.get('baseline_safe_score'))}. "
        f"Mitigated SAFE score: {_safe_str(ctx.get('mitigated_safe_score'))}. "
        f"Baseline AUC: {_safe_str(ctx.get('auc'))}. "
        f"Mitigated AUC: {_safe_str(ctx.get('mitigated_auc'))}. "
        f"Baseline fairness aggregate: {_safe_str(ctx.get('fairness_aggregate'))}. "
        f"Mitigated fairness aggregate: {_safe_str(ctx.get('mitigated_fairness_aggregate'))}. "
        f"Mitigation group: {_safe_str(ctx.get('mitigation_group'))}."
    )


def _build_decision_explanation(ctx):
    """
    Build a simple natural-language explanation of why the final
    governance decision was approved or rejected.
    """
    return (
        f"The final decision is {_safe_str(ctx.get('decision'))} because the final SAFE score "
        f"({_safe_str(ctx.get('final_safe_score'))}) is below the approval threshold "
        f"({_safe_str(ctx.get('approval_threshold'))}). "
        f"The main contributing metrics were AUC={_safe_str(ctx.get('auc'))}, "
        f"fairness aggregate={_safe_str(ctx.get('fairness_aggregate'))}, and "
        f"robustness aggregate={_safe_str(ctx.get('robustness_aggregate'))}. "
        f"In this run, fairness is the weakest of the three major aggregates."
    )


def _is_out_of_scope_query(q):
    """
    Check whether a user query is outside the SAFE chatbot's intended scope.

    This chatbot is intentionally narrow: it should answer only questions
    related to the generated SAFE artifacts and the run results.
    """
    allowed_keywords = [
        "decision", "safe score", "auc", "accuracy", "fairness", "robust", "robustness",
        "mitigation", "mitigated", "threshold", "weights", "config", "settings",
        "sensitive feature", "top features", "feature importance", "explainability",
        "sensitivity", "interaction", "effects", "baseline", "compare", "comparison",
        "why", "reason", "approved", "rejected", "model", "report"
    ]
    return not any(k in q for k in allowed_keywords)


def _format_chat_history(chat_history, max_turns=6):
    """
    Convert recent chat history into a compact text block
    for inclusion in the grounded LLM prompt.
    """
    if not chat_history:
        return "No prior conversation."

    trimmed = chat_history[-max_turns:]
    lines = []

    for turn in trimmed:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")

    return "\n".join(lines)


def _build_grounded_prompt(query, ctx, chat_history):
    """
    Build the prompt used for the fallback LLM answer path.

    The prompt includes:
    - recent chat history
    - core extracted metrics
    - full report texts
    - the user question

    This is designed to keep the LLM grounded in actual saved artifacts.
    """
    return dedent(f"""
    You are a SAFE AI results assistant.
    Answer ONLY using the grounded information below.
    Do not invent experiments, metrics, files, or conclusions that are not present.
    If the answer is not supported by the grounded context, say that clearly.

    Conversation history:
    {_format_chat_history(chat_history)}

    Grounded context:
    Decision: {_safe_str(ctx.get('decision'))}
    Final SAFE score: {_safe_str(ctx.get('final_safe_score'))}
    Approval threshold: {_safe_str(ctx.get('approval_threshold'))}
    AUC: {_safe_str(ctx.get('auc'))}
    Fairness aggregate: {_safe_str(ctx.get('fairness_aggregate'))}
    Robustness aggregate: {_safe_str(ctx.get('robustness_aggregate'))}
    Baseline SAFE score: {_safe_str(ctx.get('baseline_safe_score'))}
    Mitigated SAFE score: {_safe_str(ctx.get('mitigated_safe_score'))}
    Mitigated AUC: {_safe_str(ctx.get('mitigated_auc'))}
    Mitigated fairness aggregate: {_safe_str(ctx.get('mitigated_fairness_aggregate'))}
    Mitigation group: {_safe_str(ctx.get('mitigation_group'))}
    Sensitive feature: {_safe_str(ctx.get('sensitive_feature'))}
    Weights: {_safe_str(ctx.get('weights'))}
    Prediction threshold: {_safe_str(ctx.get('prediction_threshold'))}
    Top features: {_safe_str(ctx.get('top_features'))}

    System card:
    {ctx.get('system_card', '')}

    Evaluation report:
    {ctx.get('evaluation_report', '')}

    Final report:
    {ctx.get('final_report', '')}

    Sensitivity report:
    {ctx.get('sensitivity_report', '')}

    User question:
    {query}

    Answer in a concise, helpful way.
    If relevant, explain the result in simple words.
    """).strip()


def append_chatbot_log(user_query, assistant_answer, log_path=CHATBOT_LOG_PATH):
    """
    Append one user/assistant exchange to the chatbot log file.
    This creates a simple Markdown conversation history.
    """
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("## User\n")
        f.write(f"{user_query}\n\n")
        f.write("## Assistant\n")
        f.write(f"{assistant_answer}\n\n")
        f.write("---\n\n")


def answer_safe_chatbot_query(query: str) -> str:
    """
    Main chatbot answering function.

    Workflow:
    1. Build grounded context from artifacts
    2. Normalize user input
    3. Check required files exist
    4. Handle common supported question types with deterministic answers
    5. If needed, fall back to a grounded LLM prompt
    6. Save the result into history + log
    """
    global CHATBOT_HISTORY

    # Build context from generated SAFE artifacts
    ctx = build_chatbot_context()

    # Normalize user input
    q = (query or "").strip()
    q_lower = q.lower()

    # Required artifacts for meaningful grounded answers
    required_files = [
        str(SYSTEM_CARD_PATH),
        str(EVALUATION_REPORT_PATH),
        str(FINAL_REPORT_PATH),
    ]

    # If core artifacts are missing, the chatbot should stop early
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        answer = (
            "CHATBOT ERROR: Missing required artifacts: " + ", ".join(missing) + ". "
            "Run the full pipeline first so the chatbot can answer grounded questions."
        )
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    # Greeting/help message
    if any(x in q_lower for x in ["hello", "hi", "hey", "start", "help"]):
        answer = (
            "SAFE chatbot is ready. You can ask about the final decision, SAFE score, AUC, fairness, "
            "robustness, mitigation, configuration, sensitivity analysis, top features, "
            "or compare baseline vs mitigated results."
        )
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    # Baseline vs mitigated comparison
    if any(x in q_lower for x in ["compare", "comparison", "baseline vs mitigated", "baseline and mitigated"]):
        answer = _build_baseline_vs_mitigated_summary(ctx)
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    # Explanation of final decision
    if any(x in q_lower for x in ["why rejected", "why approved", "why", "reason", "explain decision"]):
        answer = _build_decision_explanation(ctx)
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    # Final governance decision
    if any(x in q_lower for x in ["decision", "approved", "rejected", "governance"]):
        answer = (
            f"Final governance decision: {ctx['decision']}. "
            f"Final SAFE score: {ctx['final_safe_score']}."
        )
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    # SAFE score questions
    if "safe score" in q_lower or "final score" in q_lower:
        answer = (
            f"Final SAFE score: {ctx['final_safe_score']}. "
            f"Baseline SAFE score: {ctx['baseline_safe_score']}. "
            f"Mitigated SAFE score: {ctx['mitigated_safe_score']}."
        )
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    # Performance questions
    if "auc" in q_lower or "accuracy" in q_lower or "performance" in q_lower:
        answer = (
            f"AUC: {ctx['auc']}. "
            f"Mitigated AUC: {ctx['mitigated_auc']}."
        )
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    # Fairness questions
    if "fairness" in q_lower:
        answer = (
            f"Fairness aggregate: {ctx['fairness_aggregate']}. "
            f"Mitigated fairness aggregate: {ctx['mitigated_fairness_aggregate']}. "
            "For more detail, check final_report.md for SPD, EOD, AOD, and DIR breakdown."
        )
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    # Robustness questions
    if "robust" in q_lower:
        answer = (
            f"Robustness aggregate: {ctx['robustness_aggregate']}. "
            "The robustness report is based on noise, dropout, and missingness stress tests."
        )
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    # Mitigation questions
    if any(x in q_lower for x in ["mitigation", "mitigated", "post-mitigation"]):
        answer = (
            f"Mitigated SAFE score: {ctx['mitigated_safe_score']}. "
            f"Mitigated AUC: {ctx['mitigated_auc']}. "
            f"Mitigated fairness aggregate: {ctx['mitigated_fairness_aggregate']}. "
            f"Mitigation group: {ctx['mitigation_group']}. "
            "The mitigation used here is group-aware threshold adjustment."
        )
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    # Configuration/policy questions
    if any(x in q_lower for x in ["config", "settings", "threshold", "weights", "sensitive feature"]):
        cfg = ctx["config"]
        answer = (
            "Current configuration: "
            f"prediction_threshold={cfg.get('prediction_threshold')}, "
            f"approval_threshold={cfg.get('approval_threshold')}, "
            f"weights={cfg.get('weights')}, "
            f"sensitive_feature={cfg.get('sensitive_feature')}, "
            f"drop_sensitive_from_model={cfg.get('drop_sensitive_from_model')}."
        )
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    # Feature importance / explainability questions
    if any(x in q_lower for x in ["feature importance", "top features", "important features", "explainability"]):
        if not ctx["top_features"]:
            answer = "No feature-importance summary was found in final_report.md."
            CHATBOT_HISTORY.append({"user": q, "assistant": answer})
            append_chatbot_log(q, answer)
            return answer

        formatted = ", ".join([f"{name} ({imp})" for name, imp in ctx["top_features"]])
        answer = f"Top processed features from the report: {formatted}."
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    # Sensitivity / interaction questions
    if any(x in q_lower for x in ["sensitivity", "interaction", "effects"]):
        answer = (
            "Sensitivity and interaction analysis were generated. "
            "See sensitivity_report.md for scenario comparisons, main effects, and pairwise interactions."
        )
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    # If the question is outside the SAFE chatbot scope, return a narrow fallback
    if _is_out_of_scope_query(q_lower):
        answer = (
            "I am limited to grounded questions about this SAFE run. "
            "Ask me about decision, SAFE score, AUC, fairness, robustness, mitigation, "
            "configuration, sensitivity analysis, or baseline vs mitigated comparison."
        )
        CHATBOT_HISTORY.append({"user": q, "assistant": answer})
        append_chatbot_log(q, answer)
        return answer

    # Fallback path:
    # If no deterministic rule matched, ask the shared LLM to answer
    # using the grounded prompt built from real artifacts only.
    prompt = _build_grounded_prompt(q, ctx, CHATBOT_HISTORY)
    llm_answer = crew_llm.call(prompt)

    answer = llm_answer.strip() if llm_answer else (
        "I could not produce a grounded answer from the current SAFE artifacts."
    )

    CHATBOT_HISTORY.append({"user": q, "assistant": answer})
    append_chatbot_log(q, answer)
    return answer


@tool
def safe_chatbot_tool(query: str):
    """
    CrewAI-compatible tool wrapper around the SAFE chatbot.

    This allows the chatbot to be invoked inside the multi-agent pipeline,
    for example to confirm readiness after reports are generated.
    """
    try:
        return answer_safe_chatbot_query(query)
    except Exception as e:
        return f"CHATBOT FAILED: {e}"


def run_safe_chatbot_cli():
    """
    Interactive command-line interface for the SAFE chatbot.

    This is the standalone chatbot loop used after the pipeline finishes.
    It allows the user to ask multiple grounded questions in sequence.
    """
    print("\n--- SAFE Chatbot ---")
    print("Ask about the final decision, SAFE score, fairness, robustness, mitigation, config, top features, or compare baseline vs mitigated.")
    print("Type 'clear' to reset chat history or 'exit' to stop.\n")

    while True:
        try:
            user_query = input("SAFE Chatbot > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting SAFE Chatbot.")
            break

        # Exit commands
        if user_query.lower() in {"exit", "quit", "q"}:
            print("Exiting SAFE Chatbot.")
            break

        # Reset current conversation memory
        if user_query.lower() == "clear":
            CHATBOT_HISTORY.clear()
            print("Chat history cleared.\n")
            continue

        # Ignore empty input
        if not user_query:
            continue

        # Answer the user question and print the result
        answer = answer_safe_chatbot_query(user_query)
        print(f"\n{answer}\n")
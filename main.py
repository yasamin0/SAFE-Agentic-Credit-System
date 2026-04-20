# main.py

# CrewAI core classes:
# - Agent: defines a role-based worker in the pipeline
# - Task: defines a unit of work assigned to an agent
# - Crew: orchestrates multiple agents and tasks
# - Process: defines the execution order (here: sequential)
from crewai import Agent, Task, Crew, Process

# Shared LLM configuration used by all agents
from src.config import crew_llm

# Utility to make sure all required folders exist before the pipeline starts
from src.paths import ensure_directories

# Data stage:
# - get_credit_data() loads or fetches the raw dataset
# - data_preprocessing_tool cleans, encodes, scales, and splits the data
from src.data_loader import get_credit_data, data_preprocessing_tool

# Training stage:
# trains the classifier and saves the model artifact + model card
from src.train import model_training_tool

# Evaluation stage:
# computes SAFE-related metrics such as AUC, fairness, robustness,
# and generates evaluation artifacts/reports
from src.evaluate import evaluation_and_risk_tool

# Governance stage:
# reads evaluation outputs, computes the final SAFE score,
# and produces the final system card with approval/rejection decision
from src.reporting import governance_scoring_tool

# Chatbot stage:
# - safe_chatbot_tool is used inside the Crew pipeline for readiness checking
# - run_safe_chatbot_cli is the interactive chatbot loop (now launched separately)
from src.chatbot import safe_chatbot_tool, run_safe_chatbot_cli

# Path for chatbot log file
from src.paths import CHATBOT_LOG_PATH

# Used to launch the standalone chatbot as a separate process after the Crew finishes
import subprocess
import sys


# ------------------------------------------------------------
# INITIAL SETUP
# ------------------------------------------------------------

# Create all required project folders if they do not already exist
ensure_directories()

# Load the raw credit dataset and store its path
# This may come from a local CSV or from OpenML depending on configuration
RAW_DATA_PATH = get_credit_data()


# ------------------------------------------------------------
# AGENT DEFINITIONS
# ------------------------------------------------------------

# 1) DATA AGENT
# Responsible for preparing raw credit data for machine learning.
# This agent performs cleaning, encoding, scaling, train/test split,
# and generates the Data Card artifact.
data_agent = Agent(
    role="Data Preprocessor and Feature Engineer",
    goal="Rigorously clean and transform the German Credit dataset, handling categorical variables, scaling numerical features, and creating a balanced train/test split. Produce a detailed Data Card artifact.",
    backstory=(
        "An expert in financial data preparation who specializes in preventing "
        "data leakage and ensuring data quality before any modeling starts. "
        "Their primary focus is on maximizing fairness and robustness in the downstream model."
    ),
    tools=[data_preprocessing_tool],
    llm=crew_llm,
    allow_delegation=False,
    verbose=True
)

# 2) MODELING AGENT
# Responsible for training the machine learning classifier
# and generating the model artifact and model card.
modeling_agent = Agent(
    role="Machine Learning Model Builder and Validator",
    goal="Train highly optimized predictive models (e.g., Logistic Regression and XGBoost) and select the best model based on cross-validation metrics. Output a comprehensive Model Card.",
    backstory=(
        "A seasoned ML engineer focused on credit risk assessment. "
        "They utilize state-of-the-art libraries to perform hyperparameter tuning "
        "and establish strong performance baselines."
    ),
    tools=[model_training_tool],
    llm=crew_llm,
    allow_delegation=False,
    verbose=True
)

# 3) EVALUATION AGENT
# Responsible for SAFE-oriented auditing of the trained model.
# It evaluates performance, fairness, robustness,
# and generates evaluation/sensitivity/final reports.
eval_agent = Agent(
    role="Risk and Performance Auditor (SAFE AI Focus)",
    goal="Evaluate the model against Accuracy, Robustness, and Fairness metrics.",
    backstory="A specialized auditor using the SAFE AI framework to stress test models.",
    tools=[evaluation_and_risk_tool],
    llm=crew_llm,
    allow_delegation=False,
    verbose=True
)

# 4) GOVERNANCE AGENT
# Responsible for converting technical evaluation results into a final
# governance-level SAFE score and approval/rejection decision.
# Also generates the final System Card.
safety_agent = Agent(
    role="SAFE AI Governance & Compliance Officer",
    goal=(
        "1) Compute a weighted SAFE Score from the evaluation report. "
        "2) Perform governance/compliance checks (no PII leakage, artifacts exist, reproducibility). "
        "3) Approve/Reject with a detailed, human-readable System Card."
    ),
    backstory=(
        "You are the gatekeeper of the pipeline. You consolidate artifacts from Data/Model/Eval, "
        "verify basic compliance (no raw PII, files exist, clear reporting), then compute the FINAL SAFE SCORE."
    ),
    tools=[governance_scoring_tool],
    llm=crew_llm,
    allow_delegation=False,
    verbose=True
)

# 5) CHATBOT AGENT
# Responsible for confirming that the grounded SAFE chatbot is ready.
# This agent does not do general chat; it answers only from generated artifacts.
chatbot_agent = Agent(
    role="SAFE Results Chatbot",
    goal="Answer user questions about the finished SAFE pipeline run using generated reports and configuration artifacts.",
    backstory=(
        "A lightweight assistant that only answers from the generated SAFE artifacts. "
        "It explains results, configuration, mitigation, and feature-importance findings in plain language."
    ),
    tools=[safe_chatbot_tool],
    llm=crew_llm,
    allow_delegation=False,
    verbose=True
)


# ------------------------------------------------------------
# TASK DEFINITIONS
# ------------------------------------------------------------

# TASK 1: DATA PREPARATION
# This task asks the Data Agent to preprocess the raw dataset
# and produce cleaned train/test files.
task_data_prep = Task(
    description=(
        f"Load data from {RAW_DATA_PATH}, clean it using data_preprocessing_tool, and confirm creation of "
        "clean_train_features.csv, clean_train_target.csv, clean_test_features.csv, clean_test_target.csv."
    ),
    expected_output="A summary confirming preprocessing, encoding, scaling, and the paths to the split datasets.",
    agent=data_agent,
)

# TASK 2: MODEL TRAINING
# This task depends on Task 1 because training requires processed data.
task_model_train = Task(
    description="Using the cleaned data from Task 1, execute the model_training_tool to train and select the best classification model and save the 'best_model.pkl' artifact.",
    expected_output="A confirmation that the best model has been saved and the path to the generated Model Card.",
    agent=modeling_agent,
    context=[task_data_prep]
)

# TASK 3: FULL SAFE EVALUATION
# This task depends on model training.
# It generates the evaluation report, final report, and sensitivity report.
task_full_eval = Task(
    description=(
        "You MUST call the evaluation_and_risk_tool tool exactly once. "
        "Do not just describe what should be done. "
        "Use the trained model and test data to generate evaluation_report.md, final_report.md, and sensitivity_report.md. "
        "Your final answer must summarize the metrics returned by the tool."
    ),
    expected_output="A summary of AUC, fairness aggregate, robustness aggregate, and confirmation that evaluation_report.md was created.",
    agent=eval_agent,
    context=[task_model_train]
)

# TASK 4: GOVERNANCE DECISION
# This task depends on the evaluation output.
# It computes the final SAFE score and writes the system card.
task_governance = Task(
    description=(
        "You MUST call governance_scoring_tool exactly once. "
        "Do not explain the process. "
        "Read evaluation_report.md and generate system_card.md with the final SAFE score and decision."
    ),
    expected_output="system_card.md containing decision + final SAFE score.",
    agent=safety_agent,
    context=[task_full_eval]
)

# TASK 5: CHATBOT READINESS
# This final task confirms that the grounded SAFE chatbot is ready
# to answer questions based on the generated artifacts.
task_chatbot = Task(
    description=(
        "You MUST call safe_chatbot_tool exactly once with a short readiness query such as 'help'. "
        "Do not invent results. Confirm the chatbot is ready to answer grounded questions from the generated artifacts."
    ),
    expected_output="A short readiness message from the chatbot confirming supported question types.",
    agent=chatbot_agent,
    context=[task_governance]
)


# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------

if __name__ == "__main__":
    # Create the Crew:
    # all agents are included, and tasks are executed in sequence
    safe_agent_crew = Crew(
        agents=[data_agent, modeling_agent, eval_agent, safety_agent, chatbot_agent],
        tasks=[task_data_prep, task_model_train, task_full_eval, task_governance, task_chatbot],
        process=Process.sequential,
        verbose=True
    )

    print("--- Starting SAFE Agentic Credit System ---")

    # Run the full multi-agent pipeline
    final_result = safe_agent_crew.kickoff()

    # Print final summary after the Crew finishes
    print("\n\n################################################")
    print("## FINISHED! FINAL GOVERNANCE DECISION: ##")
    print("################################################")
    print(final_result)
    print("\n[SUCCESS] System Card saved.")
    print("\n[SUCCESS] Launching standalone SAFE chatbot...")

    # Launch the chatbot in a separate process so that
    # interactive chat does not get mixed with Crew execution traces
    subprocess.run([sys.executable, "-m", "src.chat_cli"])
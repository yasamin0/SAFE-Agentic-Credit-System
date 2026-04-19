# SAFE-Agentic-Credit-System

A modular agentic AI pipeline for credit risk assessment with integrated governance scoring, fairness auditing, robustness analysis, sensitivity analysis, and a grounded results chatbot.

## Overview

This project implements a SAFE agentic credit scoring system that goes beyond standard prediction pipelines by combining:

- **Predictive performance evaluation**
- **Fairness assessment**
- **Robustness stress testing**
- **Governance-based weighted SAFE scoring**
- **Artifact-based grounded chatbot interaction**

The system uses a sequential multi-agent workflow to preprocess data, train a model, evaluate risk and safety metrics, compute a final governance decision, and expose the results through a lightweight chatbot interface.

## Main Idea

Instead of treating credit scoring as a single black-box prediction task, this project structures the workflow into multiple stages:

1. **Data preparation**
2. **Model training**
3. **SAFE evaluation**
4. **Governance scoring**
5. **Grounded chatbot interaction**

This makes the pipeline more modular, interpretable, and easier to audit.

## Project Structure

```text
SAFE-Agentic-Credit-System/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sensitive/
├── docs/
│   ├── datacard.json
│   ├── model_card.md
│   └── system_card.md
├── models/
│   └── best_model.pkl
├── reports/
│   ├── evaluation_report.md
│   ├── final_report.md
│   ├── sensitivity_report.md
│   └── chatbot_log.md
├── src/
│   ├── __init__.py
│   ├── chatbot.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── fairness.py
│   ├── model.py
│   ├── paths.py
│   ├── preprocessing.py
│   ├── reporting.py
│   ├── train.py
│   └── utils.py
├── .env.example
├── .gitignore
└── main.py
```

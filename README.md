# SAFE Agentic Credit System

A modular agentic AI pipeline for credit-risk assessment with SAFE governance scoring, fairness auditing, robustness analysis, sensitivity analysis, multi-model compliance scoring, and an artifact-grounded chatbot.

## Overview

This repository implements a SAFE agentic credit-lending system for machine learning governance. The goal is to evaluate credit-risk models not only by predictive performance, but also by fairness, robustness, explainability, and governance compliance.

The system follows a sequential multi-agent workflow. Each agent produces reusable artifacts such as cleaned datasets, trained models, evaluation reports, rank-based metric outputs, compliance score tables, a System Card, and chatbot logs.

The project is evaluated on the German Credit dataset and is designed as a reproducible prototype for transparent and auditable AI governance in credit scoring.

## Main Workflow

The pipeline is organized into five main stages:

1. **Data Agent**  
   Loads the German Credit dataset, performs EDA, outlier analysis, preprocessing, train-test splitting, feature encoding, scaling, sensitive-feature extraction, and Data Card generation.

2. **Modeling Agent**  
   Trains candidate models including Logistic Regression, Random Forest, XGBoost, Voting Ensemble, Stacking Ensemble, and a Random Baseline.

3. **Evaluation Agent**  
   Computes predictive performance, fairness metrics, robustness metrics, rank-based SAFE AI metrics, SHAP--RGE explainability comparison, mitigation results, sensitivity analysis, and compliance score outputs.

4. **Governance Agent**  
   Computes the final SAFE governance score, compares it with the approval threshold, identifies the weakest SAFE dimension, and writes the System Card.

5. **Chatbot Agent**  
   Provides an artifact-grounded conversational interface for explaining the selected model, SAFE score, fairness results, mitigation effects, compliance score comparison, and governance decision.

## SAFE Metrics

The project is inspired by the SAFE AI and Rank Graduation Box framework. It uses rank-based metrics including:

- **RGA**: Rank Graduation Accuracy
- **RGR**: Rank Graduation Robustness
- **RGE**: Rank Graduation Explainability
- **AURGA, AURGR, AURGE**: area-under-curve summaries of the corresponding rank-based curves

The final SAFE governance score combines:

```text
SAFE Score =
W_RGA  × AURGA
+ W_RGR  × AURGR
+ W_RGE  × AURGE
+ W_Fair × Fairness Aggregate
```

The Compliance Score layer is separate from the final governance decision. It compares candidate models using AURGA, AURGR, and AURGE with aggregation methods such as arithmetic mean, geometric mean, RMS, and TOPSIS.

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
├── reports/
│   ├── evaluation_report.md
│   ├── final_report.md
│   ├── sensitivity_report.md
│   ├── mitigation_report.md
│   └── figures/
├── src/
│   ├── chatbot.py
│   ├── chat_cli.py
│   ├── compliance.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── fairness.py
│   ├── model.py
│   ├── paths.py
│   ├── preprocessing.py
│   ├── reporting.py
│   ├── rga.py
│   ├── rge.py
│   ├── rgr.py
│   ├── shap_compare.py
│   ├── train.py
│   └── utils.py
├── main.py
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Meisam7/SAFE-Agentic-Credit-System.git
cd SAFE-Agentic-Credit-System
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run the full pipeline:

```bash
python main.py
```

Run the chatbot interface separately:

```bash
python src/chat_cli.py
```

## Generated Artifacts

The system generates:

- cleaned train/test datasets
- sensitive-feature files
- trained model artifacts
- Data Card, Model Card, and System Card
- evaluation and final reports
- fairness, robustness, calibration, and mitigation outputs
- RGA, RGR, and RGE metric files and plots
- compliance score tables
- chatbot logs

## Citation

This project is inspired by the SAFE AI and Rank Graduation Box framework:

```text
Babaei, G., Giudici, P., & Raffinetti, E. (2024). A Rank Graduation Box for SAFE AI. Expert Systems with Applications, 125239.

Giudici, P., & Raffinetti, E. (2024). RGA: a unified measure of predictive accuracy. Advances in Data Analysis and Classification.

Raffinetti, E. (2023). A rank graduation accuracy measure to mitigate artificial intelligence risks. Quality & Quantity, 57(Suppl 2), 131–150.
```

## Authors

- Yasamin Hosseinizadeh Sani, University of Pavia
- Golnoosh Babaei, University of Pavia
- Paolo Giudici, University of Pavia

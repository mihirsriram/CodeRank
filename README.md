# CodeRank — LangChain + LangGraph + HITL (Pairwise) with AstraDB & Offline Reranker

This is a **from‑scratch implementation** of the Python Coding Assistant using **LangChain** for LLM chains and **LangGraph** for **human‑in‑the‑loop (HITL)** pairwise ranking. It stores feedback, responses, reranker_scores, Eval_scores in **AstraDB** and includes an **offline fine‑tuning** script for the reranker.

## 🚀 Overview

CodeRank enables:
- Multi-agent code generation using **Hugging Face models**
- **Pairwise human feedback** collection (A vs B)
- **Automated reranker scoring** using `CrossEncoder`
- **Evaluation and fine-tuning** of the reranker
- Persistent storage in **AstraDB**
- Interactive dashboard built with **Streamlit**

---

## 🧩 System Architecture

![System Architecture](https://imgur.com/a/hvssAnC)



### Flow Summary
1. **User Input:** A coding problem or Python request is entered.
2. **Agents Generate Solutions:** Three models (Concise, Explainer, Optimizer) generate responses.
3. **Human Evaluation:** User picks the better response.
4. **Data Logging:** Responses, feedback, and scores are stored in AstraDB.
5. **Automated Reranking:** The reranker model ranks responses.
6. **Offline Fine-Tuning:** Human feedback data is used to improve reranker alignment.

---

## 🧠 Agent Types

| Agent | Description | Model |
|--------|-------------|--------|
| 🩵 **Concise Fixer** | Produces clean, correct, and efficient Python code. | `microsoft/phi-3-mini-128k-instruct` |
| 🧾 **Explainer** | Generates readable code with detailed step-by-step commentary and complexity explanation. | `microsoft/phi-3-mini-128k-instruct` |
| ⚡ **Optimizer** | Produces high-performance, optimized solutions with time and space complexity analysis. | `microsoft/phi-3-mini-128k-instruct` |

---

##  Architecture Components

| Module | Purpose |
|--------|----------|
| `lc_agents.py` | Handles code generation through Hugging Face inference endpoints. |
| `astra_store.py` | Connects and manages collections in AstraDB (responses, feedback, reranker_scores, evaluation_results). |
| `reranker.py` | Implements `CrossEncoder` for scoring responses. |
| `evaluation.py` | Evaluates how well the reranker aligns with human preferences and logs metrics. |
| `prompts.py` | Defines the system prompts for each agent type. |
| `streamlit_app.py` | Streamlit interface for human feedback and evaluation visualization. |

---

## AstraDB Collections

| Collection | Description |
|-------------|-------------|
| `responses` | Stores raw outputs from each agent. |
| `feedback` | Logs pairwise human feedback (A vs B). |
| `reranker_scores` | Stores reranker-assigned scores for each agent response. |
| `evaluation_results` | Logs reranker alignment metrics after evaluation runs. |

---

## Metrics Tracked

| Metric | Description |
|---------|--------------|
| **Accuracy** | Percentage of times reranker agrees with human feedback. |
| **Kendall’s τ** | Rank correlation coefficient between model and human scores. |
| **Spearman’s ρ** | Spearman correlation between paired scores. |
| **Pairs Evaluated** | Total feedback pairs analyzed. |
| **Model Used** | Reranker model path or checkpoint. |

---

## Optional Enhancements

Add a RAG pipeline for contextual grounding (e.g., documentation or code retrieval)

Personalize agent behavior based on user feedback history

Deploy fine-tuning loop with real-time reranker updates or use LLM-as-a-judge

Introduce automated evaluation dashboards

## Quickstart

### Setup Environment

python -m venv .venv
source .venv/bin/activate   # For Linux/Mac
.venv\Scripts\activate      # For Windows

pip install -r requirements.txt
cp .env.example .env        # Fill in Astra & Hugging Face credentials

python coderank_lc/scripts/export_feedback.py
python coderank_lc/scripts/offline_finetune_reranker.py --data data/pairs.jsonl --out models/reranker-ft/offline-$(date +%Y%m%d-%H%M%S)

 After training, set RERANKER_LOAD_DIR in .env to the saved folder
 Example:
 RERANKER_LOAD_DIR=models/reranker-ft/offline-2025-10-13

streamlit run coderank_lc/ui/streamlit_app.py



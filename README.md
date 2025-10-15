# CodeRank — LangChain + LangGraph + HITL (Pairwise) with AstraDB & Offline Reranker

This is a **from‑scratch implementation** of the Python Coding Assistant using **LangChain** for LLM chains and **LangGraph** for **human‑in‑the‑loop (HITL)** pairwise ranking. It stores feedback in **AstraDB** and includes an **offline fine‑tuning** script for the reranker.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill Astra creds; set HF endpoint (optional)

# run Streamlit UI
streamlit run coderank_lc/ui/streamlit_app.py
```

## Docker

```bash
docker build -t coderank-lc .
docker run -p 8501:8501 --env-file .env coderank-lc
```

## Offline Fine‑tune Reranker

```bash
python coderank_lc/scripts/export_feedback.py
python coderank_lc/scripts/offline_finetune_reranker.py --data data/pairs.jsonl --out models/reranker-ft/offline-$(date +%Y%m%d-%H%M%S)

# then set RERANKER_LOAD_DIR in your .env to the saved folder and restart the UI
```

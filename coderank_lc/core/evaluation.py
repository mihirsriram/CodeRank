# ==============================================
# Evaluation Script — Reranker Alignment + Astra Logging
# ==============================================

import time
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from coderank_lc.core.astra_store import list_recent_feedback
from coderank_lc.core.reranker import score_batch
from coderank_lc.core.astra_store import ensure_collection_exists
from coderank_lc.core.settings import RERANKER_BASE, RERANKER_LOAD_DIR
from astrapy import DataAPIClient
import os


# ==============================================
# Astra Setup
# ==============================================
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")

client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
db = client.get_database(ASTRA_DB_API_ENDPOINT)

ensure_collection_exists("evaluation_results")
evaluation_coll = db.get_collection("evaluation_results", keyspace=ASTRA_DB_KEYSPACE)


# ==============================================
# Evaluation Function
# ==============================================
def evaluate_reranker_alignment(limit: int = 1000):
    """
    Evaluate reranker alignment with human feedback and store results in AstraDB.
    Returns the evaluation DataFrame.
    """
    feedback_docs = list_recent_feedback(limit=limit)
    if not feedback_docs:
        print("⚠️ No feedback data found in AstraDB.")
        return None

    results = []
    for doc in feedback_docs:
        query = doc.get("query")
        a_text = doc.get("text_a")
        b_text = doc.get("text_b")
        preferred = doc.get("preferred")

        if not query or not a_text or not b_text:
            continue

        # Score both responses using current reranker
        scores = score_batch(query, [a_text, b_text])
        if not scores or len(scores) < 2:
            continue

        score_a, score_b = scores[0], scores[1]
        reranker_pref = "A" if score_a > score_b else "B" if score_a != score_b else "Tie"
        match = reranker_pref == preferred

        results.append({
            "query": query,
            "agent_a_score": score_a,
            "agent_b_score": score_b,
            "human_preferred": preferred,
            "reranker_preferred": reranker_pref,
            "match": match
        })

    df = pd.DataFrame(results)
    if df.empty:
        print("⚠️ No valid feedback pairs found.")
        return None

    # --- Compute Metrics ---
    accuracy = df["match"].mean() * 100
    total_pairs = len(df)
    print(f"✅ Reranker–Human Agreement: {accuracy:.2f}% ({total_pairs} pairs)")

    if df["agent_a_score"].nunique() > 1 or df["agent_b_score"].nunique() > 1:
        tau, _ = kendalltau(df["agent_a_score"], df["agent_b_score"])
        rho, _ = spearmanr(df["agent_a_score"], df["agent_b_score"])
    else:
        tau = rho = 0.0
        print("ℹ️ All scores identical — correlation not meaningful.")

    # --- Log evaluation to AstraDB ---
    eval_doc = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": round(accuracy, 3),
        "pairs_evaluated": total_pairs,
        "kendall_tau": round(tau, 3),
        "spearman_rho": round(rho, 3),
        "model_used": RERANKER_LOAD_DIR or RERANKER_BASE,
    }

    try:
        evaluation_coll.insert_one(eval_doc)
        print(f"📊 Stored evaluation result in AstraDB: {eval_doc}")
    except Exception as e:
        print(f"⚠️ Failed to store evaluation results: {e}")

    return df


# ==============================================
# Entry Point
# ==============================================
if __name__ == "__main__":
    print("🧠 Evaluating reranker alignment with human feedback...")
    df = evaluate_reranker_alignment(limit=200)
    if df is not None:
        print("\n📋 Sample results:")
        print(df.head(5))

import json
from sentence_transformers import CrossEncoder
from coderank_lc.core.astra_store import list_recent_feedback

reranker = CrossEncoder("models/reranker-ft/offline-2025-10-13")
rows = list_recent_feedback(limit=500)

correct, total = 0, 0
for r in rows:
    q = r["query"]
    a = r["text_a"]
    b = r["text_b"]
    pref = r["preferred"]
    scores = reranker.predict([(q, a), (q, b)])
    chosen = "A" if scores[0] > scores[1] else "B"
    correct += (chosen == pref)
    total += 1

acc = correct / total if total else 0
print(f"✅ Validation accuracy: {acc*100:.2f}% over {total} samples")

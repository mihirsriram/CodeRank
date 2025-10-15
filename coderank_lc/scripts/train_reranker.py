# coderank_lc/scripts/train_reranker.py
import json, os
from pathlib import Path
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from coderank_lc.core.astra_store import list_recent_feedback

out_dir = Path("models/reranker-ft/offline-2025-10-13")
out_dir.mkdir(parents=True, exist_ok=True)

rows = list_recent_feedback(limit=5000)
train_examples = []
for r in rows:
    q = r.get("query")
    pos = r.get("text_a") if r.get("preferred") == "A" else r.get("text_b")
    neg = r.get("text_b") if r.get("preferred") == "A" else r.get("text_a")
    if not (q and pos and neg): continue
    train_examples.append(InputExample(texts=[q, pos], label=1.0))
    train_examples.append(InputExample(texts=[q, neg], label=0.0))

print(f"🧩 Training on {len(train_examples)} examples")

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)

model.fit(train_dataloader=train_dataloader, epochs=2, warmup_steps=50)
model.save(str(out_dir))
print(f"✅ Fine-tuned model saved to {out_dir}")

# Export AstraDB HITL feedback to JSONL/CSV for offline fine‑tuning.
import os, json, csv
from pathlib import Path
from coderank_lc.core.astra_store import list_recent_feedback

OUT_DIR = Path(os.environ.get("OUT_DIR", "data"))
LIMIT = int(os.environ.get("EXPORT_LIMIT", 20000))
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = list_recent_feedback(LIMIT)
jsonl_path = OUT_DIR / "pairs.jsonl"
csv_path = OUT_DIR / "pairs.csv"

n = 0
with jsonl_path.open("w", encoding="utf-8") as jf, csv_path.open("w", newline="", encoding="utf-8") as cf:
    cw = csv.writer(cf); cw.writerow(["query","pos","neg","agent_pos","agent_neg"])
    for r in rows:
        q = (r.get("query") or "").strip()
        a = (r.get("text_a") or r.get("resp_A") or "").strip()
        b = (r.get("text_b") or r.get("resp_B") or "").strip()
        pref = (r.get("preferred") or "A").upper()
        if not (q and a and b):
            continue
        if pref == "A":
            pos, neg = a, b; ap, an = r.get("agent_a","A"), r.get("agent_b","B")
        else:
            pos, neg = b, a; ap, an = r.get("agent_b","B"), r.get("agent_a","A")
        rec = {"query": q, "pos": pos, "neg": neg, "agent_pos": ap, "agent_neg": an}
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        cw.writerow([q, pos, neg, ap, an]); n += 1
print(f"Exported {n} pairs → {jsonl_path} and {csv_path}")

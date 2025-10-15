import os
from typing import List
from sentence_transformers import CrossEncoder
from .settings import RERANKER_BASE, RERANKER_LOAD_DIR

_model_path = RERANKER_LOAD_DIR if (RERANKER_LOAD_DIR and os.path.isdir(RERANKER_LOAD_DIR)) else RERANKER_BASE
_cross = CrossEncoder(_model_path)

def score(query: str, resp: str) -> float:
    return float(_cross.predict([(query, resp)])[0])

def score_batch(query: str, responses: List[str]) -> List[float]:
    return [float(s) for s in _cross.predict([(query, r) for r in responses])]

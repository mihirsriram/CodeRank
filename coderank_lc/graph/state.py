from typing import Dict, Optional, Tuple, List, Any
from pydantic import BaseModel, Field

class GraphState(BaseModel):
    query: str
    responses: Dict[str, str] = Field(default_factory=dict)
    pair: Optional[Tuple[Tuple[str, str], Tuple[str, str]]] = None  # ✅ fixed type
    human_choice: Optional[str] = None  # "A" or "B"
    ranked: Optional[List[Dict[str, Any]]] = None
    awaiting_human: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

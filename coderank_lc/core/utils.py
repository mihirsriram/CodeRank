from typing import Dict, Tuple
import random

def pick_pair(responses: Dict[str, str]) -> Tuple[Tuple[str,str], Tuple[str,str]]:
    items = list(responses.items())
    a, b = random.sample(items, 2)
    return (a[0], a[1]), (b[0], b[1])

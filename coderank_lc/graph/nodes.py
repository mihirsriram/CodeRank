from coderank_lc.agents.lc_agents import generate_all
from coderank_lc.core.utils import pick_pair
from coderank_lc.core.astra_store import store_response, store_feedback
from coderank_lc.core.reranker import score_batch

# --- Node: generate multiple agent responses ---
def node_generate(state):
    query = state.query
    texts = generate_all(query)
    # persist
    for agent, text in texts.items():
        store_response({"query": query, "agent": agent, "text": text})
    state.responses = texts
    return state


# --- Node: pick A/B pair ---
def node_pick_pair(state):
    a, b = pick_pair(state.responses)
    state.pair = (a, b)
    state.awaiting_human = True
    return state


# --- Node: receive human feedback (HITL) ---
def node_wait_for_human(state):
    # Properly pause until human_choice is filled in
    if getattr(state, "human_choice", None) is None:
        print("⏸ Waiting for human feedback...")
        state.awaiting_human = True
        raise RuntimeError("HITL pause")  # STOP graph here cleanly
    # Once human choice exists, continue to record feedback
    return {"awaiting_human": False}



# --- Node: record feedback to AstraDB ---
def node_record_feedback(state):
    (a_name, a_text), (b_name, b_text) = state.pair
    choice = state.human_choice
    store_feedback({
        "query": state.query,
        "agent_a": a_name,
        "text_a": a_text,
        "agent_b": b_name,
        "text_b": b_text,
        "preferred": choice,
    })
    return state


# --- Node: rerank all responses using reranker ---
def node_rerank(state):
    agents = list(state.responses.keys())
    texts = [state.responses[a] for a in agents]
    scores = score_batch(state.query, texts)
    ranked = sorted(
        [{"agent": a, "text": t, "score": s} for a, t, s in zip(agents, texts, scores)],
        key=lambda x: x["score"],
        reverse=True,
    )
    state.ranked = ranked
    return state

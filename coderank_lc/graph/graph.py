from langgraph.graph import StateGraph, END
from coderank_lc.graph.state import GraphState

from coderank_lc.graph.nodes import (
    node_generate,
    node_pick_pair,
    node_wait_for_human,
    node_record_feedback,
    node_rerank,
)

# Build a LangGraph with a HITL interrupt

def build_graph():
    g = StateGraph(GraphState)

    g.add_node("generate", node_generate)
    g.add_node("pick_pair", node_pick_pair)
    g.add_node("wait_human", node_wait_for_human)
    g.add_node("record", node_record_feedback)
    g.add_node("rerank", node_rerank)

    g.set_entry_point("generate")
    g.add_edge("generate", "pick_pair")

    # After picking a pair, we wait for human. If no choice yet, remain; once set, continue to record
    g.add_edge("pick_pair", "wait_human")
    g.add_conditional_edges(
        "wait_human",
        lambda s: "have_choice" if getattr(s, "human_choice", None) is not None else "pause",
        {"pause": END, "have_choice": "record"},
    )




    g.add_edge("record", "rerank")
    g.add_edge("rerank", END)

    return g.compile()

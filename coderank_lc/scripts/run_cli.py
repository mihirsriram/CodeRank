# Minimal terminal HITL loop for LangGraph
from coderank_lc.graph.graph import build_graph
from coderank_lc.graph.state import GraphState

if __name__ == "__main__":
    graph = build_graph()
    query = input("Python problem: ")
    s = GraphState(query=query).model_dump()
    s = graph.invoke(s)  # runs to wait_human
    (a_name, a_text), (b_name, b_text) = s["pair"]
    print("\nA —", a_name, "\n", a_text)
    print("\nB —", b_name, "\n", b_text)
    choice = input("Pick A or B: ").strip().upper()
    s["human_choice"] = "A" if choice != "B" else "B"
    s = graph.invoke(s)
    print("\nTop ranked:")
    for row in s["ranked"][:3]:
        print(f"- {row['agent']} (score {row['score']:.3f})")

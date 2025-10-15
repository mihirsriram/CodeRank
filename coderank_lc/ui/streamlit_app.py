import streamlit as st
import random
from dotenv import load_dotenv
from coderank_lc.agents.lc_agents import generate_all
from coderank_lc.core.astra_store import store_feedback, store_response, store_reranker_score
from coderank_lc.core.reranker import score_batch

# ==========================================================
# INITIAL SETUP
# ==========================================================
load_dotenv()
st.set_page_config(page_title="🧠 CodeRank — Pairwise HITL", layout="wide")
st.title("🧠 CodeRank — Pairwise Ranking (Streamlit HITL + Reranker Logging)")

# ==========================================================
# INPUT SECTION
# ==========================================================
query = st.text_area(
    "Your Python request:",
    "Write a Python function to find the second largest number in a list."
)

# ==========================================================
# GENERATE RESPONSES
# ==========================================================
if st.button("Generate Responses"):
    with st.spinner("Generating agent responses..."):
        responses = generate_all(query)
        for agent, text in responses.items():
            store_response({"query": query, "agent": agent, "text": text})

        st.session_state.responses = responses
        st.session_state.new_generation = True

    st.success("✅ Agent responses generated and stored!")

# ==========================================================
# PAIRWISE COMPARISON (HITL)
# ==========================================================
if "responses" in st.session_state and st.session_state.responses:
    # Pick consistent random pair only once
    if "pair" not in st.session_state or st.session_state.get("new_generation", False):
        a_name, a_text = random.choice(list(st.session_state.responses.items()))
        b_name, b_text = random.choice(list(st.session_state.responses.items()))
        while b_name == a_name:
            b_name, b_text = random.choice(list(st.session_state.responses.items()))
        st.session_state.pair = (a_name, a_text, b_name, b_text)
        st.session_state.new_generation = False

    a_name, a_text, b_name, b_text = st.session_state.pair

    st.subheader("Pairwise Comparison (HITL)")
    st.caption(f"Comparing: **{a_name}** vs **{b_name}**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**A — {a_name}**")
        st.code(a_text, language="python")
    with col2:
        st.markdown(f"**B — {b_name}**")
        st.code(b_text, language="python")

    choice = st.radio("Which is better?", ["A", "B"], horizontal=True)

    # ==========================================================
    # SUBMIT FEEDBACK + RERANK LOGGING
    # ==========================================================
    if st.button("Submit Choice"):
        preferred = "A" if choice == "A" else "B"

        # --- Store HITL feedback
        store_feedback({
            "query": query,
            "agent_a": a_name, "text_a": a_text,
            "agent_b": b_name, "text_b": b_text,
            "preferred": preferred
        })
        st.success("✅ Feedback recorded!")

        # --- Run reranker scoring
        with st.spinner("📊 Scoring all responses using reranker..."):
            agents = list(st.session_state.responses.keys())
            texts = list(st.session_state.responses.values())
            scores = score_batch(query, texts)

            # Convert to floats and sort
            ranked = sorted(
                zip(agents, texts, [
                    float(s) if isinstance(s, (float, int, str)) and str(s).replace('.', '', 1).isdigit() else 0.0
                    for s in scores
                ]),
                key=lambda x: x[2],
                reverse=True
            )

            # --- Log reranker results into AstraDB 🔥
            for agent, text, score in ranked:
                store_reranker_score({
                    "query": query,
                    "agent": agent,
                    "score": float(score),
                    "text": text
                })
    
        # --- Display results
        st.subheader("🏆 Reranked Responses (with Scores)")
        for agent, text, score in ranked:
            with st.expander(f"{agent} — score {float(score):.3f}"):
                st.code(text, language="python")

        st.info("📥 All reranker scores have been logged for offline fine-tuning.")
    if st.sidebar.button("Evaluate Reranker Performance"):
        from coderank_lc.core.evaluation import evaluate_reranker_alignment
        df = evaluate_reranker_alignment(limit=200)
        st.dataframe(df)

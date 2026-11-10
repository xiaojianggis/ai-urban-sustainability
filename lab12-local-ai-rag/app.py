"""
A ChatGPT-style web page for your local assistant.

    streamlit run app.py

Opens in your browser at http://localhost:8501 — but everything still runs on your laptop.
"""
import streamlit as st

import config
from agent import run_agent
from rag import answer

st.set_page_config(page_title="Elmwood Planning Assistant", page_icon="🏙️")
st.title("Elmwood Planning Assistant")
st.caption("Local model · local documents · nothing leaves this computer")

with st.sidebar:
    mode = st.radio("Mode", ["RAG", "Agent"],
                    help="RAG: always search once, then answer. Agent: the model decides when and what to search.")
    top_k = st.slider("Top-k (chunks per search)", 1, 10, config.TOP_K)
    st.markdown(f"**Chat model:** `{config.AGENT_MODEL if mode == 'Agent' else config.LLM_MODEL}`  \n"
                f"**Embedding model:** `{config.EMBED_MODEL}`")
    if st.button("Clear chat"):
        st.session_state.history = []

st.session_state.setdefault("history", [])

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

examples = ["What is the maximum height in the R-1 district?",
            "Can I build an ADU, and how big can it be?",
            "What is the city's sales tax rate?"]
st.write("Try:", " · ".join(f"*{e}*" for e in examples))

if q := st.chat_input("Ask about Elmwood's planning documents"):
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking on your laptop..."):
            config.TOP_K = top_k  # the agent's search tool also uses this
            out = answer(q, k=top_k) if mode == "RAG" else run_agent(q, verbose=False)
        st.markdown(out["answer"])
        if mode == "RAG":
            with st.expander("Sources used"):
                for h in out["sources"]:
                    st.markdown(f"**{h['source']}** · similarity {h['score']}")
                    st.text(h["text"])
        else:
            with st.expander(f"Agent steps ({len(out['steps'])})"):
                for s in out["steps"]:
                    st.markdown(f"**Step {s['step']}** · `{s['tool']}({s['args']})`")
                    st.text(s["result"][:1500])
    st.session_state.history.append({"role": "assistant", "content": out["answer"]})

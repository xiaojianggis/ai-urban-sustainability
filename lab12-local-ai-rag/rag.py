"""
STEPS 2 and 3 of RAG — RETRIEVE, then GENERATE.

    python rag.py "What is the maximum building height in the R-1 district?"
"""
import sys

import config
from ingest import get_collection
from ollama_client import chat, embed

SYSTEM_PROMPT = """You are a careful assistant for city planning staff.
Answer using ONLY the context provided below.
If the answer is not in the context, say: "I don't have that information in the provided documents."
Quote numbers EXACTLY as written. Do not guess.
After each fact, cite the source document in square brackets, e.g. [01_zoning_ordinance.md]."""


def retrieve(question, k=None, db_dir=None):
    """Find the k chunks whose meaning is closest to the question."""
    collection = get_collection(db_dir)
    if collection.count() == 0:
        raise SystemExit("The database is empty. Run:  python ingest.py")
    q_vec = embed([question], kind="query")[0]
    res = collection.query(query_embeddings=[q_vec], n_results=k or config.TOP_K)
    return [
        {"text": text, "source": meta["source"], "score": round(1 - dist, 3)}
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def build_prompt(question, hits):
    """This is the whole 'magic': paste the retrieved text in front of the question."""
    context = "\n\n".join(
        f"[Context {i} | source: {h['source']}]\n{h['text']}" for i, h in enumerate(hits, 1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
    ]


def answer(question, k=None, db_dir=None, model=None):
    hits = retrieve(question, k, db_dir)
    reply = chat(build_prompt(question, hits), model=model)
    return {"answer": reply["content"], "sources": hits}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the maximum building height in the R-1 district?"
    result = answer(q)
    print(f"Q: {q}\n\nA: {result['answer']}\n\nSources used:")
    for h in result["sources"]:
        print(f"  {h['score']:.3f}  {h['source']}")

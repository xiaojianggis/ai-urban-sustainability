"""
RAG in about 40 lines — no database, no framework.

    python minimal_rag.py
    python minimal_rag.py "Do I need a parking space near the train station?"

Needs:  ollama pull llama3.2   and   ollama pull nomic-embed-text
"""
import math
import sys

import requests

OLLAMA = "http://127.0.0.1:11434"

# Our whole "library": three short rules from a fictional city.
docs = [
    "R-1 district: maximum building height 35 feet (2.5 stories). Minimum lot size 6,000 square feet.",
    "A detached ADU may not exceed 800 square feet or 50 percent of the main house, whichever is smaller.",
    "Within one-quarter mile of a rail station there is no minimum parking requirement.",
]


def embed(texts):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": "nomic-embed-text", "input": texts})
    return r.json()["embeddings"]


def similarity(a, b):  # cosine similarity: 1 = same meaning, 0 = unrelated
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


question = sys.argv[1] if len(sys.argv) > 1 else "How big can a backyard cottage be?"

# 1) INGEST: turn every document into a vector
doc_vectors = embed(docs)

# 2) RETRIEVE: turn the question into a vector and rank the documents by closeness
q_vec = embed([question])[0]
ranked = sorted(zip(docs, doc_vectors), key=lambda d: similarity(q_vec, d[1]), reverse=True)
print("Ranking:")
for text, vec in ranked:
    print(f"  {similarity(q_vec, vec):.3f}  {text[:70]}...")
context = ranked[0][0]

# 3) GENERATE: paste the best passage into the prompt and ask the model
prompt = (f"Answer using ONLY this context. If the answer is not there, say so.\n\n"
          f"Context: {context}\n\nQuestion: {question}")
r = requests.post(f"{OLLAMA}/api/chat", json={
    "model": "llama3.2", "stream": False,
    "messages": [{"role": "user", "content": prompt}]})
print("\nAnswer:", r.json()["message"]["content"])

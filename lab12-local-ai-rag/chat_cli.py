"""
Chat with your documents in the terminal.

    python chat_cli.py            # plain RAG
    python chat_cli.py --agent    # agentic RAG (needs a tool-calling model)

Type a question and press Enter. Type 'quit' to stop.
"""
import sys

from agent import run_agent
from rag import answer

use_agent = "--agent" in sys.argv
print(f"Elmwood planning assistant ({'agent' if use_agent else 'RAG'} mode). Type 'quit' to stop.\n")

while True:
    q = input("You: ").strip()
    if q.lower() in {"quit", "exit", "q"}:
        break
    if not q:
        continue
    if use_agent:
        out = run_agent(q)
        print(f"\nAssistant: {out['answer']}\n")
    else:
        out = answer(q)
        print(f"\nAssistant: {out['answer']}")
        print("Sources:", ", ".join(sorted({h["source"] for h in out["sources"]})), "\n")

"""
Agentic RAG: retrieval becomes a TOOL that the model decides when to use.

    ollama pull qwen3:4b
    python agent.py "My house is 1,400 square feet. How big can a detached backyard cottage be?"

Same loop as the Lecture 10 agent: ask the model -> run the tool it asks for ->
send the result back -> repeat until it answers.
"""
import ast
import json
import operator
import sys

import config
from ollama_client import chat
from rag import retrieve

MAX_STEPS = 6

SYSTEM_PROMPT = """You are a careful assistant for Elmwood city planning staff.
Use the search_documents tool to look up rules before answering. Never answer from memory.
Use the calculate tool for any arithmetic.
If the documents do not contain the answer, say so.
Cite the source document in square brackets after each fact."""


# ---------------------------------------------------------------------------
# 1) Tools are ordinary Python functions
# ---------------------------------------------------------------------------
def search_documents(query, **_ignored):   # small models sometimes add extra arguments
    hits = retrieve(query, config.TOP_K)
    return "\n\n".join(f"[source: {h['source']}]\n{h['text']}" for h in hits)


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.Mod: operator.mod}
_FUNCS = {"min": min, "max": max, "round": round, "abs": abs}


def calculate(expression, **_ignored):
    """Safe arithmetic: numbers, + - * / ** %, and min/max/round/abs. Nothing else."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in _FUNCS:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError("unsupported expression")
    try:
        return str(ev(ast.parse(str(expression), mode="eval").body))
    except Exception as e:  # tell the model what went wrong so it can retry
        return f"Error: {e}. Use plain numbers, e.g. min(800, 0.5 * 1400)"


TOOL_FUNCTIONS = {"search_documents": search_documents, "calculate": calculate}

# ---------------------------------------------------------------------------
# 2) Tool descriptions tell the model what each tool is for (this is a prompt!)
# ---------------------------------------------------------------------------
TOOLS = [
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Search Elmwood's planning documents (zoning, ADUs, parking, climate plan, permits). "
                       "Use specific search terms; search again with different words if the first "
                       "result does not contain the answer.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "What to search for"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression, e.g. 'min(800, 0.5 * 1400)'.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
]


# ---------------------------------------------------------------------------
# 3) The loop
# ---------------------------------------------------------------------------
def run_agent(question, model=None, verbose=True):
    model = model or config.AGENT_MODEL
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    steps = []
    for step in range(1, MAX_STEPS + 1):
        msg = chat(messages, model=model, tools=TOOLS)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:                                   # no tool requested -> final answer
            return {"answer": msg.get("content", ""), "steps": steps}
        for call in calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args or "{}")
            func = TOOL_FUNCTIONS.get(name)
            result = func(**args) if func else f"Error: unknown tool {name}"
            steps.append({"step": step, "tool": name, "args": args, "result": result})
            if verbose:
                print(f"  [step {step}] {name}({json.dumps(args)})")
            messages.append({"role": "tool", "tool_name": name, "content": result})
    return {"answer": "Stopped after too many steps — try a simpler question.", "steps": steps}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or \
        "My house is 1,400 square feet. How big can a detached backyard cottage be?"
    print(f"Q: {q}\n")
    out = run_agent(q)
    print(f"\n{out['answer']}")

"""
A tiny helper for talking to Ollama.

Ollama is a small web server on your own computer (http://127.0.0.1:11434).
We send it JSON and it sends JSON back. That is all this file does.
"""
import requests

import config


def _check(r, model=None):
    """Turn Ollama's errors into messages that say exactly what to do."""
    if r.ok:
        return
    try:
        detail = r.json().get("error", r.text)
    except ValueError:
        detail = r.text
    if r.status_code == 404 and model:
        raise SystemExit(f"\nOllama does not have the model '{model}' yet.\n"
                         f"Fix: run   ollama pull {model}   in a terminal, then try again.\n"
                         f"(Ollama said: {detail})")
    if "does not support tools" in str(detail):
        raise SystemExit(f"\nThe model '{model}' cannot call tools, so it cannot run the agent.\n"
                         f"Fix: set AGENT_MODEL in config.py to a tool-calling model, e.g. qwen3:4b or llama3.1:8b.")
    raise SystemExit(f"\nOllama returned an error ({r.status_code}): {detail}")


def chat(messages, model=None, tools=None, temperature=None):
    """Send a conversation to the chat model and return its reply message (a dict)."""
    body = {
        "model": model or config.LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": config.TEMPERATURE if temperature is None else temperature},
    }
    if tools:
        body["tools"] = tools
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=body, timeout=600)
    _check(r, body["model"])
    return r.json()["message"]


def embed(texts, kind="document"):
    """Turn a list of texts into a list of vectors.

    kind = "document" when storing chunks, "query" when embedding a question.
    nomic-embed-text was trained with these two prefixes, and works better with them.
    """
    if config.EMBED_MODEL.startswith("nomic-embed-text"):
        prefix = "search_query: " if kind == "query" else "search_document: "
        texts = [prefix + t for t in texts]
    r = requests.post(f"{config.OLLAMA_URL}/api/embed",
                      json={"model": config.EMBED_MODEL, "input": texts}, timeout=600)
    _check(r, config.EMBED_MODEL)
    return r.json()["embeddings"]


def list_models():
    """Names of the models you have pulled, e.g. ['llama3.2:latest', ...]."""
    r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json()["models"]]

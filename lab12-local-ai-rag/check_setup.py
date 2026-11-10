"""
Checks that everything is ready and tells you exactly what to fix.

    python check_setup.py
"""
import importlib

import config

ok = True


def good(msg):
    print(f"  [ OK ]  {msg}")


def bad(msg, fix):
    global ok
    ok = False
    print(f"  [FAIL]  {msg}\n          fix: {fix}")


print("1. Python packages")
for pkg in ["requests", "chromadb", "streamlit", "pypdf"]:
    try:
        importlib.import_module(pkg)
        good(pkg)
    except ImportError:
        bad(f"{pkg} is not installed", "pip install -r requirements.txt")

print("\n2. Ollama server")
try:
    from ollama_client import list_models
    models = list_models()
    good(f"Ollama is running at {config.OLLAMA_URL}")
except Exception:
    models = None
    bad("Cannot reach Ollama", "open the Ollama app (Mac/Windows) or run `ollama serve` (Linux)")

print("\n3. Models")
if models is not None:
    def have(name):
        return any(m == name or m.split(":")[0] == name or m == f"{name}:latest" for m in models)
    for name, why in [(config.LLM_MODEL, "chat model"), (config.EMBED_MODEL, "embedding model"),
                      (config.AGENT_MODEL, "agent model (only for agent.py)")]:
        if have(name):
            good(f"{name}  ({why})")
        else:
            bad(f"{name} not downloaded ({why})", f"ollama pull {name}")

print("\n4. Documents and database")
docs = [p for p in config.DATA_DIR.glob("*") if p.suffix.lower() in {".md", ".txt", ".pdf"}]
good(f"{len(docs)} documents in data/") if docs else bad("data/ is empty", "add .md/.txt/.pdf files")
if config.DB_DIR.exists():
    good("chroma_db/ exists")
else:
    bad("No vector database yet", "python ingest.py")

print("\nAll set! Try:  python rag.py" if ok else "\nFix the items marked FAIL, then run this again.")

"""
All the settings for this lab live here.
Change a value, save the file, and re-run the script you are working on.
"""
from pathlib import Path

HERE = Path(__file__).parent

# --- Where Ollama is running (your own computer) -----------------------------
OLLAMA_URL = "http://127.0.0.1:11434"

# --- Models ------------------------------------------------------------------
LLM_MODEL = "llama3.2"            # chat model for plain RAG (~2 GB)
EMBED_MODEL = "nomic-embed-text"  # turns text into vectors (~275 MB)
AGENT_MODEL = "qwen3:4b"          # needs "tools" support: try qwen3:4b, llama3.1:8b, granite4:3b

# If you change EMBED_MODEL you MUST run `python ingest.py` again.
# Vectors from two different embedding models cannot be compared.

# --- Files -------------------------------------------------------------------
DATA_DIR = HERE / "data"          # put your documents here (.md, .txt, .pdf)
DB_DIR = HERE / "chroma_db"       # the vector database is saved here
COLLECTION = "planning_docs"

# --- The four knobs ----------------------------------------------------------
CHUNK_SIZE = 900      # characters per chunk      (re-run ingest.py after changing)
CHUNK_OVERLAP = 150   # characters shared by neighbouring chunks (re-run ingest.py)
TOP_K = 4             # how many chunks go into the prompt
TEMPERATURE = 0.1     # low = stick closely to the retrieved text

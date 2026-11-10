"""
Proof #1: the answers come from YOUR files, not from the model's memory.

    python demo_grounding.py

1. Ask the bare model (no documents) about fictional Elmwood -> it invents an answer.
2. Ask the RAG system -> it answers from the ordinance, with a citation.
3. Change "35 feet" to "42 feet" in a COPY of the ordinance, re-ingest the copy,
   and ask again -> the answer follows the file.
"""
import shutil
import tempfile
from pathlib import Path

import config
from ingest import ingest
from ollama_client import chat
from rag import answer

Q = "What is the maximum building height in Elmwood's R-1 district?"

print("=" * 70, "\n1) BARE MODEL — no documents\n")
print(chat([{"role": "user", "content": Q}])["content"])

print("\n" + "=" * 70, "\n2) RAG — with Elmwood's documents\n")
print(answer(Q)["answer"])

print("\n" + "=" * 70, "\n3) RAG after editing the ordinance: 35 feet -> 42 feet\n")
tmp = Path(tempfile.mkdtemp())
try:
    data_copy, db_copy = tmp / "data", tmp / "db"
    shutil.copytree(config.DATA_DIR, data_copy)
    f = data_copy / "01_zoning_ordinance.md"
    f.write_text(f.read_text(encoding="utf-8").replace("35 feet", "42 feet"), encoding="utf-8")
    ingest(data_copy, db_copy, verbose=False)
    print(answer(Q, db_dir=db_copy)["answer"])
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\nIf the answer followed the file, the system is reading the file.")
print("(Your real data/ folder was not changed.)")

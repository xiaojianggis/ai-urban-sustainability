# Lab 12 · Local AI and RAG: a private planning assistant on your laptop

In this lab you will run a language model **on your own computer** and teach it to answer
questions from **your own documents** — the zoning code of a fictional city called Elmwood —
with citations, and without sending a single word to the internet.

```
 your question ──► search your documents ──► best passages + question ──► local model ──► answer + sources
                   (retrieve)                                              (generate)
```

That pattern is called **RAG — Retrieval-Augmented Generation**: *search first, then write*.

**Time:** about 90 minutes · **Level:** beginner (you only need to run Python scripts) ·
**Cost:** free · **Internet:** only for the one-time downloads

---

## Contents

1. [Set up (once)](#1-set-up-once)
2. [Run a model on your laptop](#2-run-a-model-on-your-laptop)
3. [RAG in 40 lines](#3-rag-in-40-lines)
4. [The Elmwood planning assistant](#4-the-elmwood-planning-assistant)
5. [RAG inside an agent](#5-rag-inside-an-agent)
6. [Don't trust it — prove it](#6-dont-trust-it--prove-it)
7. [The four knobs](#7-the-four-knobs)
8. [Exercises](#8-exercises)
9. [Troubleshooting](#9-troubleshooting)
10. [Reference: models, files, glossary](#10-reference)

---

## 1. Set up (once)

**You need** a laptop with at least 8 GB of RAM (16 GB is better) and about 5 GB of free disk.

**a) Install Ollama** from <https://ollama.com> and open the app.
Ollama downloads models and runs them on your computer. (Linux: also run `ollama serve` in a terminal.)

**b) Download two models** — open a terminal (Mac: *Terminal*; Windows: *Anaconda Prompt*):

```bash
ollama pull llama3.2            # the chat model        (~2 GB)
ollama pull nomic-embed-text    # the embedding model   (~275 MB) — used for search
ollama pull qwen3:4b            # optional: the agent model for Section 5 (~2.5 GB)
```

**c) Get the code and install the Python packages**

```bash
git clone https://github.com/xiaojianggis/ai-urban-sustainability.git
cd ai-urban-sustainability/lab12-local-ai-rag
conda activate geospatial          # or any Python 3.10+ environment
pip install -r requirements.txt
```

**d) Check everything**

```bash
python check_setup.py
```

It prints `[ OK ]` or `[FAIL]` for each item, and tells you the exact command that fixes each failure.

> **Watch out — not every Ollama model is local.** Models whose names end in `:cloud`
> (for example `gpt-oss:120b-cloud`) run on Ollama's servers. For private data, use only normal
> models, or start Ollama with `OLLAMA_NO_CLOUD=1`.

---

## 2. Run a model on your laptop

### In the terminal

```bash
ollama run llama3.2
>>> Explain floor-area ratio to a resident in two sentences.
```

Type `/bye` to quit. Now **turn your Wi-Fi off and ask again** — it still works.
That is what *local AI* means: the model file is on your disk, and your text is processed on your computer.

Other useful commands:

```bash
ollama list            # models you have downloaded
ollama ps              # models loaded in memory right now
ollama show llama3.2   # size, quantization, context length, capabilities ("tools"?)
```

### From Python

Ollama is a small web server on your own computer at `http://127.0.0.1:11434`.
Python talks to it by sending JSON:

```python
import requests
r = requests.post("http://127.0.0.1:11434/api/chat", json={
    "model": "llama3.2", "stream": False,
    "messages": [{"role": "user", "content": "What is a floor-area ratio?"}]})
print(r.json()["message"]["content"])
```

> **Code from Lab 10/11 works too.** Ollama speaks the same API as OpenAI. Change two lines:
> ```python
> client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
> MODEL = "qwen3:4b"
> ```

### Why bother? (the planner's view)

| Cloud AI (ChatGPT, Claude, Gemini) | Local AI (Ollama) |
|---|---|
| Your text goes to a company's servers | Your text never leaves your computer |
| Strongest reasoning, longest documents | Smaller models: weaker at hard reasoning and arithmetic |
| The model can change without notice | A fixed model file you can cite and reproduce |
| Pay per use | Free after the download |

**Rule of thumb: match the tool to the sensitivity of the data.** Public documents → cloud is fine.
Draft staff reports, public comments with names and addresses, 311 complaints → local.

---

## 3. RAG in 40 lines

Ask the bare model: *"What is the maximum building height in Elmwood's R-1 district?"*
It confidently answers something like "35 feet". But Elmwood is **fictional** — the model never
read its code. It produced a *plausible* number. What the model lacks is not intelligence; it is
**access** to your documents. RAG gives it that access.

Open [`minimal_rag.py`](minimal_rag.py). It contains the whole idea in three steps:

```python
doc_vectors = embed(docs)                       # 1. INGEST   – turn each document into numbers
q_vec = embed([question])[0]                    # 2. RETRIEVE – turn the question into numbers,
ranked = sorted(..., key=similarity, ...)       #               find the closest document
prompt = f"Answer using ONLY this context ..."  # 3. GENERATE – paste it in and ask the model
```

Run it:

```bash
python minimal_rag.py
python minimal_rag.py "Do I need a parking space near the train station?"
```

Look at the **ranking**. "Backyard cottage" matched the ADU rule even though the rule never says
"cottage". That is the job of the **embedding model**: it turns text into a list of numbers
(a point in space) so that **texts with similar meaning land close together**.

```
"maximum building height"  ↔  "how tall can I build?"      → close   (same meaning)
"maximum building height"  ↔  "urban tree canopy goals"    → far     (different meaning)
```

> **Key idea:** the model never "searches". Ordinary Python does the search.
> The model only reads the text we paste in front of it.

---

## 4. The Elmwood planning assistant

Same idea, organized for real use, with a database and a web page.

```
data/ (5 documents) ──► ingest.py ──► chroma_db/ ──► rag.py ──► app.py (chat in your browser)
                        chunk+embed     vectors      retrieve +
                                                     generate
```

```bash
python ingest.py        # 5 documents → 17 chunks → chroma_db/
python rag.py "What is the maximum building height in the R-1 district?"
streamlit run app.py    # opens a chat page in your browser
```

`ingest.py` cuts each document into **chunks** (~900 characters, overlapping so a rule is never cut
in half), embeds every chunk, and saves them in **Chroma**, a vector database that is just a folder
on your disk. You run it once, and again whenever the documents change.

`rag.py` does the rest for every question. Here is **everything the model actually receives**:

```
SYSTEM: You are a careful assistant for city planning staff.
        Answer using ONLY the context provided below.
        If the answer is not in the context, say: "I don't have that information..."
        Quote numbers EXACTLY as written. Do not guess.
        After each fact, cite the source document in square brackets.

USER:   [Context 1 | source: 01_zoning_ordinance.md]
        ### Section 4.2 R-1 Low-Density Residential District ...
        - Maximum building height: 35 feet (2.5 stories). ...

        [Context 2 | source: 02_adu_regulations.md] ...

        Question: What is the maximum building height in the R-1 district?
```

Two details make this work: the prompt **allows** the model to say "I don't know" (without that
permission it guesses), and the **file name travels with every chunk**, which makes citation possible.

### Six questions to try

Open **Sources used** under every answer in the app.

| Question | What a good answer looks like |
|---|---|
| What is the maximum height in the R-1 district? | 35 feet (2.5 stories) [01] |
| Can I build an ADU, and how big can it be? | Yes; detached ≤ 800 sq ft or 50% of the house, whichever is smaller [02] |
| How does Elmwood reduce parking near transit? | No minimums within ¼ mile of rail; maximums in MU-T; 50% cut near frequent buses [03] |
| What are the city's climate and canopy goals? | Canopy 24% → 35% by 2045; GHG −50% by 2030 [04] |
| Summarize the rules for short-term rentals. | Primary residence, 90 nights, $150 permit [01, 05] |
| What is the city's sales tax rate? | "I don't have that information" — **the most important answer of all** |

Prefer the terminal? `python chat_cli.py`

---

## 5. RAG inside an agent

In Lab 11 you built an **agent**: a model in a loop that can call **tools**.
**Agentic RAG** simply makes search one of those tools. Instead of always searching once with your
exact words, the model decides **whether** to search, **what** to search for, **how many times**, and
what else to do (for example, arithmetic — which small models are bad at).

```bash
python agent.py "My house is 1,400 square feet. How big can a detached backyard cottage be?"
```

A typical run:

```
  [step 1] search_documents({"query": "detached ADU maximum size"})
  [step 2] calculate({"expression": "min(800, 0.5 * 1400)"})

A detached ADU can be at most 700 square feet — 800 sq ft or 50% of the main house,
whichever is smaller [02_adu_regulations.md].
```

Open [`agent.py`](agent.py): the tools are ordinary Python functions, the tool **descriptions** are
prompts, and the loop is the same shape as Lab 11. In the app, choose **Agent** in the sidebar and
open **Agent steps**.

| | Plain RAG | Agentic RAG |
|---|---|---|
| Searches per question | always 1 | 0, 1 or several |
| Can use other tools | no | yes (calculator, GIS lookup, database) |
| Predictable | yes | less |
| Needs a tool-calling model | no | **yes** — `ollama show <model>` must list *tools* |

Use plain RAG for a predictable front-counter bot; agentic RAG when questions need several lookups or a calculation.

---

## 6. Don't trust it — prove it

```bash
python demo_grounding.py    # bare model vs RAG; then edits "35 feet" → "42 feet" in a COPY of the
                            # ordinance — the answer follows the file
python demo_privacy.py      # lists every network address used while answering: all 127.0.0.1
```

Then switch your Wi-Fi off and run `python rag.py` again.
**Claims about privacy should be testable. Ask for the test, not the promise.**

---

## 7. The four knobs

All settings live in [`config.py`](config.py).

| Knob | Default | What it does | After changing |
|---|---|---|---|
| `CHUNK_SIZE` | 900 | Characters per chunk. Bigger = more context per hit, blunter search | re-run `ingest.py` |
| `CHUNK_OVERLAP` | 150 | Text shared by neighbouring chunks, so rules aren't split | re-run `ingest.py` |
| `TOP_K` | 4 | Chunks pasted into the prompt. Too few misses facts; too many buries them | nothing |
| `TEMPERATURE` | 0.1 | How much the model improvises. Low = sticks to the text | nothing |

Changing `LLM_MODEL` is free. Changing `EMBED_MODEL` means you **must** re-run `ingest.py`.

---

## 8. Exercises

1. **Your own city (core).** Put a real zoning code (`.md`, `.txt` or `.pdf`) in `data/`, run
   `python ingest.py`, and ask three questions. Report one success and one failure, with a guess *why* it failed.
2. **Knobs.** Compare `CHUNK_SIZE` 300 vs 2000 and `TOP_K` 1 vs 8 on the ADU and short-term-rental
   questions. When do bigger chunks help, and when do they hurt?
3. **Model comparison.** Answer the six questions with `llama3.2:1b`, `llama3.2`, and one newer model
   (`gemma4:e4b` or `qwen3.5:4b`). Record correct answers and seconds per answer.
4. **Public comments.** Copy `exercises/public_comments_synthetic.md` into `data/`, re-ingest, and ask
   *"What are the main concerns in the public comments?"* Which comments did the summary leave out?
   (Hint: look at comments 12, 16 and 21.) What does that mean for equity?
5. **Plain RAG vs agent.** Write ten questions with known answers — two needing arithmetic, two needing
   two documents. Score both modes and count the agent's tool calls.
6. **Add a tool (advanced).** Give the agent a `lookup_zoning(address)` tool using a small CSV
   (address → district), so it can answer *"How tall can I build at 12 Linden St?"*

### Before you trust any answer

- **A missed search looks exactly like a good answer.** Always open the sources.
- **"Cited" is not "correct".** The citation shows where it looked, not that it read correctly.
- **Old documents give old answers.** Grounding inherits your filing discipline.
- **Small models drop negations and miscalculate.** "Not permitted" is easy to lose.
- **Summaries flatten dissent.** A minority concern can vanish into "residents were supportive".

RAG reduces made-up answers. It does **not** transfer responsibility: a person still signs the staff report.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot reach Ollama` / connection refused | Open the Ollama app (Linux: `ollama serve`) |
| `model "llama3.2" not found` | `ollama pull llama3.2` |
| `The database is empty` | `python ingest.py` |
| Nonsense search results after changing models | You changed `EMBED_MODEL` — re-run `python ingest.py` |
| Very slow or out of memory | Use `llama3.2:1b` in `config.py`; close other apps |
| Agent never calls a tool / errors about tools | The model can't call tools. Use `qwen3:4b`, `llama3.1:8b` or `granite4:3b` |
| A PDF gives "no text found" | It is a scanned image — it needs OCR first |

---

## 10. Reference

### Which model fits my laptop? (as of autumn 2026 — check <https://ollama.com/library>)

| Your computer | Chat model | For the agent (tool calling) |
|---|---|---|
| 8 GB RAM | `llama3.2:1b`, `gemma4:e2b` | `qwen3:1.7b` (simple tools only) |
| 16 GB RAM (most laptops) | `llama3.2`, `gemma4:e4b`, `qwen3.5:4b` | `qwen3:4b`, `granite4:3b`, `llama3.1:8b` |
| 32 GB+ RAM or a 16 GB GPU | `qwen3:14b`, `gpt-oss:20b`, `gemma4:26b` | `gpt-oss:20b`, `qwen3:14b` |

**Why does an 8-billion-parameter model fit in 5 GB?** *Quantization*: Ollama stores each number in
about 4 bits instead of 16, which makes the model about four times smaller with little loss in quality.

**Start small. Move up only when you can name the question the small model got wrong.**

### Files in this folder

| File | What it is |
|---|---|
| `config.py` | All settings: models and the four knobs |
| `ollama_client.py` | Three small functions that talk to Ollama: `chat`, `embed`, `list_models` |
| `minimal_rag.py` | RAG in ~40 lines, no database |
| `ingest.py` | Step 1: chunk, embed, store in Chroma |
| `rag.py` | Steps 2–3: retrieve, build the prompt, generate |
| `agent.py` | Agentic RAG: search and calculator tools in a loop |
| `app.py` | Streamlit chat page (RAG or Agent mode) |
| `chat_cli.py` | The same in the terminal |
| `check_setup.py` | Tells you what is missing |
| `demo_grounding.py`, `demo_privacy.py` | The two proofs |
| `data/` | Five fictional Elmwood planning documents |
| `exercises/` | Synthetic public comments for Exercise 4 |
| [`READING_LIST.md`](READING_LIST.md) | Papers and documentation for going deeper |

### Glossary

| Term | Meaning |
|---|---|
| **Local AI** | The model runs on hardware you control; your data stays there |
| **Open-weight model** | A model whose file you can download and run (Llama, Gemma, Qwen, Mistral, gpt-oss) |
| **Embedding** | A list of numbers that represents the meaning of a text |
| **Vector database** | Stores embeddings and finds the nearest ones quickly (Chroma) |
| **Chunk** | A piece of a document; the unit that is searched |
| **Top-k** | How many chunks are retrieved per question |
| **RAG** | Retrieve relevant text, then generate an answer from it |
| **Agentic RAG** | Search offered as a tool that an agent decides when and how to use |
| **Quantization** | Storing model numbers with fewer bits to save memory |

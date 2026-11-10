# Lab 12 · Local LLMs and RAG

Run a language model **on your own laptop**, then have it answer questions from **your own documents**
(the zoning code of a fictional city, Elmwood), with citations and without sending anything to the internet.

```
 your question ──► search your documents ──► best passages + question ──► local model ──► answer + sources
```

This is **RAG (Retrieval-Augmented Generation)**: *search first, then write*.


---

## 1. Set up

You need a laptop with 8 GB of RAM or more, and about 5 GB of free disk space.

1. Install **Ollama** from <https://ollama.com> and open the app. (Linux: run `ollama serve`.)
2. Download the models. In a terminal (Mac: *Terminal*; Windows: *Anaconda Prompt*):

   ```bash
   ollama pull llama3.2           # chat model, ~2 GB
   ollama pull nomic-embed-text   # search (embedding) model, ~275 MB
   ollama pull qwen3:4b           # agent model for Step 5, ~2.5 GB
   ```

3. Get the code and install the packages:

   ```bash
   git clone https://github.com/xiaojianggis/ai-urban-sustainability.git
   cd ai-urban-sustainability/lab12-local-ai-rag
   conda activate geospatial        # or any Python 3.10+ environment
   pip install -r requirements.txt
   python check_setup.py            # shows [ OK ] or [FAIL], and how to fix each problem
   ```

---

## 2. Chat with a model on your laptop

```bash
ollama run llama3.2
>>> Explain floor-area ratio to a resident in two sentences.
```

Type `/bye` to quit. Then **turn your Wi-Fi off and ask again**: it still works, because the model is on your disk.

**How does Python reach it?** Ollama runs a small web server on your computer at
`http://127.0.0.1:11434`. `127.0.0.1` means "this computer", and `11434` is Ollama's port. Ollama
starts the server for you, so there's nothing to set up. To check it, open <http://127.0.0.1:11434> in a browser.
It should say *"Ollama is running"*. Our scripts send it requests with Python's `requests` library
(see [`ollama_client.py`](ollama_client.py)).

---

## 3. RAG in 40 lines

A bare model asked about Elmwood's height limit will confidently *invent* a number, because it has never seen Elmwood.
RAG fixes that by giving it the right passage first. Open [`minimal_rag.py`](minimal_rag.py) and run:

```bash
python minimal_rag.py
python minimal_rag.py "Do I need a parking space near the train station?"
```

The script does three things:

1. **Ingest:** turn each document into a list of numbers (an *embedding*).
2. **Retrieve:** turn the question into numbers and find the closest document.
3. **Generate:** paste that document into the prompt and ask the model.

Look at the printed ranking. *"Backyard cottage"* matched the ADU rule even though the rule never says
"cottage". Embeddings capture **meaning**, not spelling. Also note that the model never searches;
plain Python does the search.

---

## 4. The Elmwood planning assistant

The same idea, with five real-length documents in [`data/`](data/) and a chat page:

```bash
python ingest.py        # cut 5 documents into 17 chunks, embed them, save to chroma_db/
streamlit run app.py    # opens a chat page in your browser (it still runs on your laptop)
```

Ask these questions and open **Sources used** under each answer:

- *What is the maximum height in the R-1 district?* → 35 feet [01_zoning_ordinance.md]
- *Can I build an ADU, and how big can it be?* → 800 sq ft or 50% of the house, whichever is smaller
- *What is the city's sales tax rate?* → **"I don't have that information."** The documents don't cover it, so a good system says so instead of guessing.

The settings (chunk size, top-k, models) are all in [`config.py`](config.py).

---

## 5. RAG inside an agent

In Lab 11, an agent was a model in a loop that calls tools. Here, **search is one of the tools**, along with a calculator,
and the model decides when to use them:

```bash
python agent.py "My house is 1,400 square feet. How big can a detached backyard cottage be?"
```

```
  [step 1] search_documents({"query": "detached ADU maximum size"})
  [step 2] calculate({"expression": "min(800, 0.5 * 1400)"})

A detached ADU can be at most 700 square feet ... [02_adu_regulations.md]
```

In the chat page, switch the sidebar to **Agent** and open **Agent steps** to watch it work.

**Done.** You ran a private, cited planning assistant on your own laptop.

---

## If something goes wrong

| Message | Fix |
|---|---|
| `Cannot reach Ollama` / connection refused | Open the Ollama app (Linux: `ollama serve`) |
| `Ollama does not have the model ...` or `404 ... /api/chat` | `ollama pull <that model>` |
| `The database is empty` | `python ingest.py` |
| Slow or out of memory | In `config.py`, set `LLM_MODEL = "llama3.2:1b"` |
| The agent never uses its tools | Use a tool-calling model: `qwen3:4b` or `llama3.1:8b` |

> **Optional, reusing OpenAI code from earlier labs.** Ollama also accepts the OpenAI API format. With the
> `openai` package (`pip install openai`), point existing code at your laptop:
> `client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")` (the key is ignored), and set `MODEL = "qwen3:4b"`.

> **Not every Ollama model is local.** Models ending in `:cloud` run on Ollama's servers. For private data,
> avoid them, or start Ollama with `OLLAMA_NO_CLOUD=1`.

---

## Readings

**Start here**

- Lewis et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). The paper that named RAG; read the abstract and Figure 1.
- Anthropic (2024). [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval). Why simple retrieval fails, and how to fix it, with no math.
- Kapoor, Bommasani et al. (2024). [On the Societal Impact of Open Foundation Models](https://arxiv.org/abs/2403.07918). The case for and against open-weight models.
- Sanchez, T. (2023). [Planning With Artificial Intelligence](https://planning.org/publications/report/9270237/). APA PAS Report 604, the planning profession's introduction to AI.

**How RAG works, and how to make it better**

- Gao et al. (2024). [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997). The standard map of the field.
- Liu et al. (2024). [Lost in the Middle](https://arxiv.org/abs/2307.03172). Why pasting in *more* passages can make answers worse.
- Karpukhin et al. (2020). [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906). Where embedding-based search came from.
- Edge et al. (2024). [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130). For "summarize everything" questions, such as all public comments.
- Es et al. (2024). [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217). How to measure whether a RAG system works.

**RAG inside agents**

- Yao et al. (2023). [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629). The reason, act, observe loop.
- Asai et al. (2024). [Self-RAG](https://arxiv.org/abs/2310.11511). A model that decides when to retrieve and checks itself.
- Singh et al. (2025). [Agentic Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2501.09136).
- Anthropic (2024). [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents). Practical patterns, and when *not* to use an agent.

**What "open" means**

- Solaiman (2023). [The Gradient of Generative AI Release](https://arxiv.org/abs/2302.04844). Model release runs from fully closed to fully open.
- Liesenfeld & Dingemanse (2024). [Rethinking Open Source Generative AI: Open-Washing and the EU AI Act](https://facctconference.org/static/papers24/facct24-120.pdf). How open are "open" models?
- Open Source Initiative (2024). [The Open Source AI Definition 1.0](https://opensource.org/ai).
- U.S. NTIA (2024). [Dual-Use Foundation Models with Widely Available Model Weights](https://www.ntia.gov/issues/artificial-intelligence/open-model-weights-report). The U.S. policy view; read the executive summary.
- Groeneveld et al. (2024). [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838). What a fully open model looks like.

**Running models locally, and their energy use**

- Xu et al. (2024). [On-Device Language Models: A Comprehensive Review](https://arxiv.org/abs/2409.00088).
- Frantar et al. (2023). [GPTQ](https://arxiv.org/abs/2210.17323). How models shrink to 4 bits (technical).
- Dettmers et al. (2023). [QLoRA](https://arxiv.org/abs/2305.14314). Fine-tuning a quantized model on one GPU.
- Luccioni, Jernite & Strubell (2024). [Power Hungry Processing](https://arxiv.org/abs/2311.16863). The energy cost of AI inference; picking the right-sized model is also a climate choice.

**Planning and urban applications**

- Andrews et al. (2022). [AI in Planning: Opportunities and Challenges](https://planning-org-uploaded-media.s3.amazonaws.com/publication/download_pdf/AI-in-Planning-White-Paper-Rev.pdf). APA Foresight white paper on equity and the planner's role.
- Zhu et al. (2024). [PlanGPT](https://arxiv.org/abs/2402.19273). A planning-specific model combined with retrieval over planning documents.
- Zhou, Jeong & Chapple (2026). [An LLM-Based Chatbot System Framework for Urban Planners](https://impact.ornl.gov/en/publications/a-large-language-model-based-chatbot-system-framework-for-urban-p/). RAG over zoning codes in six cities.
- Aqib et al. (2025). [Fine-Tuning LLMs and Evaluating Retrieval Methods for Building Codes](https://arxiv.org/abs/2505.04666). Keyword search still matters for code-like text.

**Hands-on documentation and free courses**

- [Ollama documentation](https://docs.ollama.com) · [model library](https://ollama.com/library) · [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility) · [tool calling](https://ollama.com/blog/tool-support)
- [Chroma documentation](https://docs.trychroma.com) (the vector database used here)
- [LlamaIndex: RAG concepts](https://developers.llamaindex.ai/python/framework/understanding/rag/)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) · [Hugging Face Agents Course](https://huggingface.co/learn/agents-course)
- [Open WebUI](https://docs.openwebui.com): a ChatGPT-style interface for local models.

A longer, annotated version of this list, with discussion questions, is in [`READING_LIST.md`](READING_LIST.md).

---

**Files:** `config.py` (settings) · `ollama_client.py` (talks to Ollama) · `minimal_rag.py` · `ingest.py` · `rag.py` ·
`agent.py` · `app.py` (chat page) · `chat_cli.py` (terminal chat) · `check_setup.py` ·
`demo_grounding.py` and `demo_privacy.py` (optional demos: answers follow the files, and nothing leaves the laptop) · `data/` (fictional Elmwood documents)

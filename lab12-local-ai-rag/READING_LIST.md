
# Lecture 11 Reading List: Localized AI, Open Models, and RAG

**How to use this list.** ★ = required before class (about 2 hours in total). The other readings are
optional, for students who want to go deeper or are doing a related project. Each entry says why it
is on the list.

---

## 1. Start here (required)

1. ★ **Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** *NeurIPS 2020.*
   <https://arxiv.org/abs/2005.11401>
   The paper that named RAG. Read the abstract, introduction and Figure 1; skip the training details.
   *Why:* the original idea — "retrieve, then generate" — in its authors' own words.

2. ★ **Anthropic (2024). Introducing Contextual Retrieval.** Engineering blog, September 2024.
   <https://www.anthropic.com/news/contextual-retrieval>
   A readable, practical explanation of why plain chunk-and-embed retrieval fails, and how adding context
   to each chunk, keyword search (BM25) and re-ranking fix it.
   *Why:* the practical fixes when retrieval misses; no math required.

3. ★ **Kapoor, S., Bommasani, R. et al. (2024). On the Societal Impact of Open Foundation Models.** *ICML 2024.*
   <https://arxiv.org/abs/2403.07918>
   Lays out the benefits of open-weight models (customisation, local control, transparency, less
   concentration of power) and a framework for judging their risks.
   *Why:* the clearest argument for and against open weights. Read sections 1–4.

4. ★ **Sanchez, T. (2023). Planning With Artificial Intelligence.** American Planning Association, PAS Report 604.
   <https://planning.org/publications/report/9270237/>
   The planning profession's own introduction to AI, with examples such as analysing public comments in Long Beach, CA.
   *Why:* puts today's tools in a planning context. (Penn students: check the library or APA membership for access.)

---

## 2. RAG: how it works and how to make it better

5. **Gao, Y. et al. (2023, updated 2024). Retrieval-Augmented Generation for Large Language Models: A Survey.**
   <https://arxiv.org/abs/2312.10997>
   The standard overview: naive, advanced and modular RAG; retrieval, augmentation and evaluation.
   *Why:* the best single map of the field. Use it as a reference rather than reading it cover to cover.

6. **Liu, N. F. et al. (2024). Lost in the Middle: How Language Models Use Long Contexts.** *TACL.*
   <https://arxiv.org/abs/2307.03172>
   Shows that models use information at the beginning and end of a long prompt better than information in the middle.
   *Why:* explains why "just paste in more chunks" (a high top-k) can make answers worse.

7. **Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering.** *EMNLP 2020.*
   <https://arxiv.org/abs/2004.04906>
   The paper behind embedding-based retrieval (the "nearby means similar" idea).
   *Why:* for students who want to know where embeddings for search came from.

8. **Asai, A. et al. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.** *ICLR 2024.*
   <https://arxiv.org/abs/2310.11511>
   A model that decides when to retrieve and checks its own answers against the passages.
   *Why:* an early step from fixed RAG toward agentic RAG.

9. **Edge, D. et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization.** Microsoft Research.
   <https://arxiv.org/abs/2404.16130>
   Builds a knowledge graph from documents so the system can answer "summarise everything" questions that top-k retrieval cannot.
   *Why:* relevant to summarising all public comments, which top-k retrieval handles poorly.

10. **Es, S. et al. (2024). RAGAS: Automated Evaluation of Retrieval Augmented Generation.** *EACL 2024 (demo).*
    <https://arxiv.org/abs/2309.15217>
    Measures faithfulness, answer relevance and context relevance without hand-labelled answers.
    *Why:* a way to check whether a RAG system actually works.

---

## 3. RAG inside agents

11. **Yao, S. et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models.** *ICLR 2023.*
    <https://arxiv.org/abs/2210.03629>
    The reason–act–observe loop that agents use (review from Lecture 10).
    *Why:* agentic RAG is ReAct with a search tool.

12. **Singh, A. et al. (2025). Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG.**
    <https://arxiv.org/abs/2501.09136>
    Classifies agentic RAG designs (single agent, multi-agent, hierarchical, graph-based) and their uses.
    *Why:* the reference for RAG inside agents.

---

## 4. Open models: what "open" means

13. **Solaiman, I. (2023). The Gradient of Generative AI Release: Methods and Considerations.** *FAccT 2023.*
    <https://arxiv.org/abs/2302.04844>
    Release is not open vs. closed but a gradient, from fully closed to fully open.
    *Why:* the conceptual basis for "closed / open weights / open source".

14. **Liesenfeld, A. & Dingemanse, M. (2024). Rethinking Open Source Generative AI: Open-Washing and the EU AI Act.** *FAccT 2024.*
    <https://facctconference.org/static/papers24/facct24-120.pdf>
    Scores 40+ "open" models on 14 dimensions of openness and finds many are only open-weight ("open-washing").
    *Why:* teaches students to read "open" claims critically.

15. **Open Source Initiative (2024). The Open Source AI Definition 1.0.**
    <https://opensource.org/ai>
    The first formal definition of open-source AI, released October 2024; still debated because it does not require releasing training data.
    *Why:* a short primary source; good for discussion.

16. **U.S. NTIA (2024). Dual-Use Foundation Models with Widely Available Model Weights.** Report to the White House, July 2024.
    <https://www.ntia.gov/issues/artificial-intelligence/open-model-weights-report>
    The U.S. government's assessment of open-weight models; recommends monitoring rather than restricting them.
    *Why:* the policy view. Read the executive summary only.

17. **Groeneveld, D. et al. (2024). OLMo: Accelerating the Science of Language Models.** Allen Institute for AI.
    <https://arxiv.org/abs/2402.00838>
    A fully open model: weights, training data, code and evaluation are all released.
    *Why:* what "fully open" looks like in practice; compare with item 14.

### Model reports (skim one, for context)

18. **Meta AI (2024). The Llama 3 Herd of Models.** <https://arxiv.org/abs/2407.21783>
19. **OpenAI (2025). gpt-oss-120b & gpt-oss-20b Model Card.** <https://arxiv.org/abs/2508.10925>
20. **Google DeepMind (2026). Gemma 4 Technical Report.** <https://arxiv.org/abs/2607.02770>

*Why:* model reports show what makers disclose (and what they do not) about data, training, safety
testing and licences. A good exercise: compare two reports using the 14 dimensions from item 14.

---

## 5. Running models locally: efficiency and sustainability

21. **Xu, J. et al. (2024). On-Device Language Models: A Comprehensive Review.**
    <https://arxiv.org/abs/2409.00088>
    Covers model compression, efficient architectures and hardware for running language models on phones and laptops.
    *Why:* the technical background for "localized AI".

22. **Frantar, E. et al. (2023). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.** *ICLR 2023.*
    <https://arxiv.org/abs/2210.17323>
    One of the methods that made 4-bit models practical.
    *Why:* for students curious how an 8B model shrinks to about 5 GB. Optional and technical.

23. **Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.** *NeurIPS 2023.*
    <https://arxiv.org/abs/2305.14314>
    Fine-tuning a quantized model on a single GPU.
    *Why:* the route if RAG is not enough and you need to adapt a local model.

24. **Kwon, W. et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM).** *SOSP 2023.*
    <https://arxiv.org/abs/2309.06180>
    The serving engine used when a whole department shares one GPU server.
    *Why:* the "on-premises" row of the spectrum slide.

25. **Luccioni, A. S., Jernite, Y. & Strubell, E. (2024). Power Hungry Processing: Watts Driving the Cost of AI Deployment?** *ACM FAccT 2024.*
    <https://arxiv.org/abs/2311.16863>
    Measures the energy and carbon cost of AI *inference* across tasks and model sizes. Generative,
    general-purpose models use far more energy than task-specific ones.
    *Why:* the sustainability angle for this course — the right-sized model is a climate choice too.

---

## 6. Planning and urban applications

26. **Andrews, C. et al. (2022). AI in Planning: Opportunities and Challenges and How to Prepare.** APA Foresight white paper.
    <https://planning-org-uploaded-media.s3.amazonaws.com/publication/download_pdf/AI-in-Planning-White-Paper-Rev.pdf>
    Equity, ethics and the planner's role in shaping AI tools.
    *Why:* free and short; good for the responsibility discussion.

27. **Zhu, H. et al. (2024). PlanGPT: Enhancing Urban Planning with Tailored Language Model and Efficient Retrieval.**
    <https://arxiv.org/abs/2402.19273>
    A planning-specific language model combined with retrieval over planning documents.
    *Why:* the closest published system to the Elmwood demo.

28. **Zhou, X., Jeong, B. & Chapple, K. (2026). A Large Language Model-Based Chatbot System Framework for Urban Planners.** *Expert Systems with Applications*, 300, 130307.
    <https://impact.ornl.gov/en/publications/a-large-language-model-based-chatbot-system-framework-for-urban-p/>
    A chatbot that finds and interprets zoning rules across six Canadian cities, using stored and live-scraped data.
    *Why:* RAG on real zoning codes, with an evaluation against GPT-4o and Llama 3.2.

29. **Aqib, M. et al. (2025). Fine-Tuning Large Language Models and Evaluating Retrieval Methods for Improved Question Answering on Building Codes.**
    <https://arxiv.org/abs/2505.04666>
    Compares retrievers and fine-tuning on the National Building Code of Canada; keyword search (Elasticsearch) retrieved best.
    *Why:* evidence that keyword search still matters for code-like text (see hybrid search).

---

## 7. Hands-on documentation

- **Ollama documentation** — <https://docs.ollama.com> and the model library <https://ollama.com/library>
- **Ollama OpenAI compatibility** — <https://docs.ollama.com/api/openai-compatibility>
- **Chroma documentation** — <https://docs.trychroma.com>
- **Hugging Face LLM Course** (free) — <https://huggingface.co/learn/llm-course/chapter1/1>
- **Open WebUI** (a ChatGPT-style interface for local models) — <https://docs.openwebui.com>
- **LlamaIndex: RAG concepts** — <https://developers.llamaindex.ai/python/framework/understanding/rag/>

---

## Discussion questions for the readings

1. Using Liesenfeld & Dingemanse (14), how "open" is the model we used in class? Does it matter for a city agency?
2. Kapoor et al. (3) argue that open models reduce concentration of power. Does that apply to a planning department?
3. Lost in the Middle (6) and Contextual Retrieval (2) both suggest that "more context" is not always better. What does that mean for top-k?
4. Luccioni et al. (25): when is a small local model the more sustainable choice, and when is it not?
5. Zhou et al. (28) and PlanGPT (27) both evaluate their systems. What would *you* need to see before a zoning chatbot answers residents' questions?

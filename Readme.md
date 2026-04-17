# Weld Assistant RAG

An AI-powered welding knowledge assistant built with LangChain, FAISS, and Google Gemini 2.5 Flash. Ask natural-language questions about welding procedure specifications (WPS), defect criteria, and process parameters — get grounded answers retrieved directly from source documents, not hallucinated.

**Status:** Work in progress.

---

## Why this project

Welding supervisors at manufacturing plants often need quick answers from dense WPS documents and inspection logs — "what current range is approved for this procedure?", "what defects were logged last week?", "which filler is specified for 316L stainless?". Flipping through PDFs is slow. Generic chatbots hallucinate unsafe answers.

This project solves that by grounding an LLM in real welding documents using Retrieval-Augmented Generation (RAG).

---

## How it works

```
weld_data.txt  →  TextLoader  →  Chunker  →  Embeddings  →  FAISS index
                                                                 ↓
User question  →  Retriever (top-k)  →  Prompt template  →  Gemini 2.5 Flash  →  Answer
```

1. **Load** welding documents using `TextLoader`
2. **Chunk** text into 200-token segments with 50-token overlap, respecting paragraph boundaries
3. **Embed** chunks using `all-MiniLM-L6-v2` (384-dim sentence transformer, runs locally)
4. **Store** vectors in FAISS for similarity search
5. **Retrieve** the top-3 most relevant chunks for each question
6. **Prompt** Gemini 2.5 Flash with retrieved context + user question
7. **Return** a grounded answer via `StrOutputParser`

The pipeline is composed using LangChain Expression Language (LCEL) — a single declarative chain with `RunnableParallel` for parallel context/question handling and pipe operators for composition.

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| LLM | Gemini 2.5 Flash | Cost-efficient ($0.30/1M input tokens), fast, good reasoning |
| Embeddings | all-MiniLM-L6-v2 | Runs locally, no API cost, solid speed/quality tradeoff |
| Vector store | FAISS | Production-grade, Meta AI, in-memory speed |
| Framework | LangChain (LCEL) | Modern declarative chain composition |
| Language | Python 3.11 | |

---

## Setup

1. Clone the repo:
```bash
   git clone https://github.com/the-anand-sharma/Weld_Assistant_RAG_Based.git
   cd Weld_Assistant_RAG_Based
```

2. Install dependencies:
```bash
   pip install langchain langchain-community langchain-huggingface langchain-google-genai langchain-text-splitters faiss-cpu python-dotenv
```

3. Create a `.env` file in the project root with your Gemini API key:
```
   GOOGLE_API_KEY=your_key_here
```
   Get one free at [ai.google.dev](https://ai.google.dev/).

4. Run:
```bash
   python gemini_vecc.py
```

---

## Example query

**Input:** *"What are the welding parameters in WPS-2026-001?"*

**Output:** The assistant retrieves the specific WPS entry from `weld_data.txt` and returns the voltage range, current range, travel speed, and filler material — grounded in the source document, no hallucination.

---

## Roadmap

- [x] End-to-end RAG chain with Gemini + FAISS
- [x] Chunking strategy tuned for WPS documents
- [ ] Conversation memory for multi-turn queries
- [ ] Streamlit UI for non-technical users
- [ ] PDF loader for real WPS documents
- [ ] CSV loader + AI analysis of sensor readings
- [ ] Deploy on GCP Cloud Run

---

## About

Built by [Anand Sharma](https://github.com/the-anand-sharma) · [LinkedIn](https://www.linkedin.com/in/anand-sharma-573527133/)
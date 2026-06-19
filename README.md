# RAG Chatbot — Multi-Turn Conversational AI

A production-style Retrieval-Augmented Generation (RAG) chatbot with multi-turn memory, semantic retrieval, LLM-grounded answer generation, and automated evaluation — built with ChromaDB and OpenRouter.

## What This Does

Unlike a plain LLM chatbot that relies on training data, this system retrieves relevant knowledge on every turn before generating an answer — grounding responses in facts and reducing hallucination.

| Feature | Details |
|---------|---------|
| Multi-turn memory | Maintains conversation history across turns (with context window trimming) |
| Per-turn retrieval | Fresh semantic search on every user message |
| Grounded generation | LLM sees retrieved context + history before answering |
| Source attribution | Every answer cites which document it drew from |
| LLM-as-judge eval | Automated scoring of answers against expected outputs (0-10) |
| Session summary | Per-turn stats table: chunks found, similarity scores, zero-result turns |

## Architecture

```
User Message (Turn N)
        |
        v
[Embedding Model]          <- sentence-transformers (local)
        |
        v
[ChromaDB Vector Search]   <- cosine similarity -> Top-K chunks
        |
        v
[Prompt Assembly]
  system prompt
  + conversation history   <- last N turns (trimmed)
  + retrieved_context      <- fresh each turn
  + user message
        |
        v
[LLM via OpenRouter]       <- grounded answer generation
        |
        v
Answer + Source Attribution + Stats
```

## Project Structure

```
rag-chatbot/
├── rag_chatbot.py       # Full pipeline: build_kb, retrieve, llm, RAGChatbot class
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

```bash
git clone http://github.com/rctharan19/rag_chatbot.git
cd rag_chatbot

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

cp .env.example .env
# Paste your OpenRouter API key into .env
```

Get a free key at https://openrouter.ai

## Run

```bash
python rag_chatbot.py
```

Sample output:
```
[Turn 1] You: What is RAG and why is it useful?
         Chunks: 3  best_sim: 0.8821
Assistant: RAG reduces hallucination by grounding answers in retrieved facts. [Source: rag_overview]

=================================================================
SESSION SUMMARY
=================================================================
  Total turns          : 4
  Total chunks retrieved: 12
  Avg best_sim         : 0.847
  Zero-result turns    : 0

  Turn   Chunks   Best Sim   Query
  ----------------------------------------------------------------
  1      3        0.882      What is RAG and why is it useful?
  2      3        0.871      How does ChromaDB store embeddings?
```

## Key Classes and Functions

### `RAGChatbot`
The main class. Instantiate with:
```python
bot = RAGChatbot(top_k=3, min_similarity=0.30, temperature=0.2, max_history=6)
bot.chat("Your question here")
bot.session_summary()
```

### `retrieve(query, top_k, min_similarity)`
Embeds the query and returns matching chunks above the similarity threshold.

### `score_answer(question, expected, actual)`
LLM-as-judge: scores an answer against the expected output on a 0-10 scale.

### `bot.evaluate(qa_pairs)`
Runs a list of `(question, expected_answer)` pairs and prints average score.

## Customise

- **Swap the LLM**: Change `MODEL_NAME` to any model on OpenRouter (e.g. `google/gemma-2-9b-it`)
- **Add documents**: Extend `KNOWLEDGE_BASE` with your own `{source, text}` dicts
- **Persist the index**: Replace `chromadb.Client()` with `chromadb.PersistentClient(path="./chroma_db")`
- **Filter by category**: Add metadata fields and use `where=` in ChromaDB queries

## Tech Stack

| Component | Library |
|-----------|---------|
| Vector store | ChromaDB |
| Embeddings | sentence-transformers (via ChromaDB default EF) |
| LLM | OpenRouter (any OSS model) |
| Similarity | Cosine distance (HNSW) |
| Evaluation | LLM-as-judge |

## Concepts Demonstrated

- Multi-turn RAG with conversation memory
- Context window management via history trimming
- Fresh per-turn retrieval to keep context current
- Source attribution with confidence scores
- Automated LLM-based evaluation (0-10 scoring)
- Zero-result handling without hallucination


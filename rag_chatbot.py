"""
RAG Chatbot — Multi-Turn Conversational Assistant
==================================================
A production-style RAG (Retrieval-Augmented Generation) chatbot that:
  - Maintains multi-turn conversation history
  - Retrieves fresh context from a ChromaDB vector store on every turn
  - Injects history + retrieved context into each LLM prompt
  - Tracks retrieval stats across the session
  - Supports LLM-judged evaluation against expected answers
"""

import os
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your-api-key-here")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "mistralai/mistral-7b-instruct"

client_llm = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

# ── Knowledge Base ───────────────────────────────────────────────────────────

KNOWLEDGE_BASE = [
    {
        "source": "rag_overview",
        "text": (
            "RAG (Retrieval-Augmented Generation) combines a retrieval step with "
            "LLM generation. Instead of relying solely on parametric memory, the model "
            "receives relevant documents as context before generating an answer. "
            "This reduces hallucination and keeps responses grounded in facts."
        ),
    },
    {
        "source": "embeddings_intro",
        "text": (
            "Embeddings are dense vector representations of text. Similar texts have "
            "vectors that are close in high-dimensional space, measured by cosine similarity. "
            "Models like sentence-transformers convert sentences into fixed-size vectors "
            "(e.g., 384 or 768 dimensions)."
        ),
    },
    {
        "source": "chromadb_guide",
        "text": (
            "ChromaDB is an open-source vector database designed for AI applications. "
            "It stores documents alongside their embeddings and supports fast approximate "
            "nearest-neighbour search using HNSW indexing. Collections can be persisted "
            "to disk or kept in-memory for prototyping."
        ),
    },
    {
        "source": "chunking_strategy",
        "text": (
            "Chunking splits large documents into smaller pieces before indexing. "
            "Common strategies include fixed-size chunks (e.g., 512 tokens), "
            "sentence-level splitting, and recursive character splitting. "
            "Chunk size affects both retrieval precision and context window usage."
        ),
    },
    {
        "source": "openrouter_api",
        "text": (
            "OpenRouter is an API gateway that provides access to many open-source LLMs "
            "(Mistral, LLaMA, Gemma, etc.) through a single OpenAI-compatible endpoint. "
            "It supports streaming, tool use, and model switching without code changes."
        ),
    },
    {
        "source": "conversation_memory",
        "text": (
            "Multi-turn chatbots maintain conversation history as a list of "
            "{role, content} dicts. To avoid exceeding the context window, history is "
            "trimmed to the most recent N message pairs. The retrieved context is injected "
            "fresh on each turn so the model always has relevant, current information."
        ),
    },
    {
        "source": "similarity_threshold",
        "text": (
            "A similarity threshold filters out low-confidence retrievals. "
            "Chunks with cosine similarity below the threshold (e.g., 0.30) are discarded "
            "to prevent irrelevant context from confusing the LLM. "
            "If no chunks pass the threshold, the system informs the user rather than hallucinating."
        ),
    },
    {
        "source": "llm_evaluation",
        "text": (
            "LLM-as-judge evaluation uses a separate LLM call to score generated answers "
            "against expected outputs on a 0-10 scale. The judge receives the question, "
            "the expected answer, and the actual answer, then rates factual accuracy, "
            "completeness, and relevance. This enables automated quality assessment of RAG pipelines."
        ),
    },
]

# ── Vector Store Setup ───────────────────────────────────────────────────────

def build_knowledge_base(docs: list) -> chromadb.Collection:
    """Embed and index all knowledge base documents into ChromaDB."""
    chroma_client = chromadb.Client()
    ef = embedding_functions.DefaultEmbeddingFunction()

    try:
        chroma_client.delete_collection("rag_kb")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="rag_kb",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        documents=[d["text"] for d in docs],
        metadatas=[{"source": d["source"]} for d in docs],
        ids=[f"doc_{i}" for i in range(len(docs))],
    )
    print(f"✅  Indexed {len(docs)} documents into ChromaDB")
    return collection


# Initialise global collection
COLLECTION = build_knowledge_base(KNOWLEDGE_BASE)


# ── Retrieval Helper ─────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3, min_similarity: float = 0.30) -> list:
    """
    Embed the query and return top_k chunks above the similarity threshold.
    Each result dict: {text, source, similarity}
    """
    results = COLLECTION.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = round(1 - dist, 4)
        if similarity >= min_similarity:
            chunks.append({"text": doc, "source": meta["source"], "similarity": similarity})

    return chunks


# ── LLM Helper ───────────────────────────────────────────────────────────────

def llm(messages: list, temperature: float = 0.2) -> str:
    """Send a message list to the LLM and return the response text."""
    response = client_llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


# ── LLM Judge ────────────────────────────────────────────────────────────────

def score_answer(question: str, expected: str, actual: str) -> int:
    """
    Use an LLM as judge to score the actual answer vs expected on 0-10.
    Returns an integer score.
    """
    prompt = (
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Actual answer: {actual}\n\n"
        "Rate the actual answer on factual accuracy, completeness, and relevance "
        "to the expected answer. Respond with ONLY a single integer from 0 to 10."
    )
    result = llm([{"role": "user", "content": prompt}], temperature=0.0)
    try:
        return int("".join(filter(str.isdigit, result))[:2])
    except ValueError:
        return 0


# ── RAG Chatbot ───────────────────────────────────────────────────────────────

class RAGChatbot:
    """
    Multi-turn RAG chatbot with conversation memory.

    Each user message triggers:
      1. Vector retrieval from the knowledge base
      2. Context injection into a full message list
         (system + history + retrieved context + new user message)
      3. LLM generation of a grounded answer
      4. Stats tracking for session review
    """

    SYSTEM_PROMPT = (
        "You are a knowledgeable AI assistant specialising in RAG systems, "
        "embeddings, and LLM APIs. "
        "Answer questions using ONLY the retrieved context provided. "
        "If the context doesn't contain enough information, say so clearly. "
        "Cite the source name for each factual claim. "
        "Keep answers concise and accurate."
    )

    def __init__(self, top_k=3, min_similarity=0.30, temperature=0.2, max_history=6):
        self.top_k          = top_k
        self.min_similarity = min_similarity
        self.temperature    = temperature
        self.max_history    = max_history

        self.history = []
        self.stats   = []
        self.turn    = 0

    def _trim_history(self):
        """Keep only the most recent max_history message pairs."""
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

    def chat(self, user_message: str, verbose: bool = True) -> str:
        """Process one user turn. Returns the assistant's response string."""
        self.turn += 1

        # 1. Retrieve relevant chunks
        chunks = retrieve(user_message, top_k=self.top_k, min_similarity=self.min_similarity)

        # 2. Format context block
        if chunks:
            context_block = "\n\n".join(
                f"[Source: {c['source']} | sim={c['similarity']}]\n{c['text']}"
                for c in chunks
            )
            context_note = f"<retrieved_context>\n{context_block}\n</retrieved_context>\n\n"
        else:
            context_note = "<retrieved_context>No relevant documents found.</retrieved_context>\n\n"

        # 3. Build message list: system + trimmed history + new user turn with context
        self._trim_history()
        messages = (
            [{"role": "system", "content": self.SYSTEM_PROMPT}]
            + self.history
            + [{"role": "user", "content": context_note + user_message}]
        )

        # 4. Generate answer
        answer = llm(messages, temperature=self.temperature)

        # 5. Append clean history (without context block)
        self.history.append({"role": "user",      "content": user_message})
        self.history.append({"role": "assistant", "content": answer})

        # 6. Track stats
        self.stats.append({
            "turn"        : self.turn,
            "query"       : user_message,
            "chunks_found": len(chunks),
            "best_sim"    : chunks[0]["similarity"] if chunks else 0.0,
            "sources"     : [c["source"] for c in chunks],
        })

        if verbose:
            print(f"\n[Turn {self.turn}] You: {user_message}")
            print(f"         Chunks: {len(chunks)}  best_sim: {self.stats[-1]['best_sim']:.4f}")
            print(f"\nAssistant: {answer}\n")
            print("─" * 60)

        return answer

    def evaluate(self, qa_pairs: list) -> float:
        """
        Run (question, expected_answer) pairs and score each with the LLM judge.
        Returns the average score out of 10.
        """
        scores = []
        for q, expected in qa_pairs:
            answer = self.chat(q, verbose=False)
            score  = score_answer(q, expected, answer)
            scores.append(score)
            print(f"  Score {score}/10 | Q: {q[:60]}")
        avg = np.mean(scores)
        print(f"\n  --- Average: {avg:.1f}/10 ---")
        return avg

    def session_summary(self):
        """Print a summary table of retrieval stats for this session."""
        print("\n" + "=" * 65)
        print("SESSION SUMMARY")
        print("=" * 65)
        print(f"  Total turns          : {self.turn}")
        print(f"  Total chunks retrieved: {sum(s['chunks_found'] for s in self.stats)}")
        print(f"  Avg best_sim         : {np.mean([s['best_sim'] for s in self.stats]):.3f}")
        print(f"  Zero-result turns    : {sum(1 for s in self.stats if s['chunks_found'] == 0)}")
        print()
        print(f"  {'Turn':<6} {'Chunks':<8} {'Best Sim':<10} {'Query':<40}")
        print("  " + "-" * 64)
        for s in self.stats:
            print(f"  {s['turn']:<6} {s['chunks_found']:<8} {s['best_sim']:<10.3f} {s['query'][:40]}")
        print("=" * 65)


# ── Demo ─────────────────────────────────────────────────────────────────────

def main():
    print("🤖  RAG Chatbot — Multi-Turn Demo")
    print(f"    Model: {MODEL_NAME} | Top-K: 3 | Min similarity: 0.30\n")

    bot = RAGChatbot(top_k=3, min_similarity=0.30, temperature=0.2, max_history=6)

    # Sample multi-turn conversation
    demo_questions = [
        "What is RAG and why is it useful?",
        "How does ChromaDB store and search embeddings?",
        "What chunking strategy should I use for long documents?",
        "How does the similarity threshold affect retrieval quality?",
    ]

    print("── Demo Conversation ──────────────────────────────────────")
    for q in demo_questions:
        bot.chat(q)

    # Evaluation
    print("\n── Evaluation ─────────────────────────────────────────────")
    eval_pairs = [
        (
            "What is an embedding?",
            "Embeddings are dense vector representations of text where similar "
            "texts have vectors close in high-dimensional space."
        ),
        (
            "What happens when no chunks pass the similarity threshold?",
            "The system informs the user that no relevant context was found "
            "rather than hallucinating an answer."
        ),
    ]
    bot.evaluate(eval_pairs)

    # Session summary
    bot.session_summary()

    # Interactive mode
    print("\n💬  Interactive Mode  (type 'quit' to exit)")
    print("-" * 60)
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            bot.session_summary()
            break
        if user_input:
            bot.chat(user_input)


if __name__ == "__main__":
    main()

"""
RETRIEVAL + GENERATION
=======================
This is the "AG" in RAG. Given a question, it:
  1. Embeds the question with the SAME model used during ingest
     (critical — mixing embedding models gives garbage similarity scores)
  2. Finds the top-k most similar chunks in ChromaDB
  3. Builds a prompt that includes those chunks + the question
  4. Sends it to Groq's LLM and returns the answer + sources

WHY "SAME MODEL" MATTERS:
  Embeddings are only comparable to other embeddings from the same model.
  Two different models can encode the same sentence into totally different
  vector spaces — comparing across them is meaningless.

WHY GIVE THE LLM EXPLICIT INSTRUCTIONS TO SAY "I DON'T KNOW"?
  Without this, LLMs will confidently make things up when the retrieved
  chunks don't actually answer the question. Explicitly permitting
  "I don't know" reduces hallucination a lot.
"""

import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "notes"
TOP_K = 4  # how many chunks to retrieve per question

SYSTEM_PROMPT = """You are a study assistant answering questions based ONLY on the
provided notes excerpts. Rules:
- Answer using only the information in the excerpts below.
- If the excerpts don't contain enough information to answer, say so clearly —
  do not guess or use outside knowledge.
- Reference which source(s) you used when relevant.
- Keep answers focused and exam-relevant."""


def get_collection():
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def retrieve(question: str, collection, top_k: int = TOP_K):
    results = collection.query(query_texts=[question], n_results=top_k)
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    return list(zip(chunks, metadatas))


def build_prompt(question: str, retrieved: list) -> str:
    context_blocks = []
    for chunk, meta in retrieved:
        source_label = f"{meta['source']}" + (f" (page {meta['page']})" if meta.get("page") else "")
        context_blocks.append(f"[Source: {source_label}]\n{chunk}")
    context = "\n\n---\n\n".join(context_blocks)

    return f"""Notes excerpts:

{context}

---

Question: {question}"""


def answer_question(question: str, collection, groq_client: Groq) -> tuple[str, list]:
    retrieved = retrieve(question, collection)

    if not retrieved:
        return "No notes have been ingested yet. Run ingest.py first.", []

    user_prompt = build_prompt(question, retrieved)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  
    )

    answer = response.choices[0].message.content
    sources = sorted(set(f"{m['source']}" + (f" p.{m['page']}" if m.get("page") else "") for _, m in retrieved))
    return answer, sources


if __name__ == "__main__":
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    collection = get_collection()

    print(f"Loaded collection with {collection.count()} chunks. Ask a question (Ctrl+C to quit).")
    while True:
        q = input("\n> ")
        answer, sources = answer_question(q, collection, groq_client)
        print(f"\n{answer}\n\nSources: {', '.join(sources)}")

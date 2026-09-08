# Notes RAG Chatbot

A chatbot that answers questions using only your own coursework notes/PDFs,
with citations back to the source. Built to learn: embeddings, vector search,
prompt grounding, and a real chat UI.

## How it works (the whole idea in one paragraph)

Your notes get cut into chunks → each chunk is converted into a vector of
numbers that represents its *meaning* (an embedding) → those vectors are
stored in a searchable database (ChromaDB) → when you ask a question, it's
embedded the same way → the database returns the chunks whose vectors are
closest to your question's vector → those chunks get stuffed into an LLM
prompt so it answers using your actual notes instead of guessing.

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get a free Groq API key: https://console.groq.com/keys
cp .env.example .env
# then paste your key into .env

# 4. Add your notes
# Drop PDFs, .txt, or .md files into the data/ folder

# 5. Build the vector database
python ingest.py

# 6. Try it from the terminal first
python rag.py

# 7. Launch the full chat UI
streamlit run app.py
```

## Project structure

```
notes-rag/
├── data/           # your PDFs / notes go here (not committed to git)
├── chroma_db/      # generated vector database (not committed to git)
├── ingest.py       # Day 1: loads, chunks, embeds, stores your notes
├── rag.py          # Day 2: retrieval + LLM answer generation
├── app.py          # Day 3: Streamlit chat interface
├── requirements.txt
└── .env.example
```

## What to learn from each file, in order

1. **`ingest.py`** — read this first. Focus on `chunk_text()` — try changing
   `CHUNK_SIZE` and `CHUNK_OVERLAP` and re-running to see how retrieval
   quality changes.
2. **`rag.py`** — this is the retrieval-augmented generation core. Run it
   directly (`python rag.py`) to test in the terminal before touching the UI.
   Try changing `TOP_K` (how many chunks get retrieved) and `temperature`
   (how "creative" vs. grounded the LLM's answers are).
3. **`app.py`** — wraps `rag.py` in a chat UI. Look at `st.session_state` —
   this is the key Streamlit concept for anything with memory across clicks.

## Things to try once it's working

- Ask a question your notes *don't* cover — check it says "I don't know"
  instead of making something up.
- Print the retrieved chunks in `rag.py` before they're sent to the LLM —
  seeing what got retrieved (and what didn't) is the fastest way to
  understand and debug RAG quality.
- Swap `all-MiniLM-L6-v2` for a different sentence-transformers model and
  compare retrieval quality.
- Add a "show sources" expander in the UI that displays the raw retrieved
  chunks, not just the filenames.

## .gitignore (add this before pushing to GitHub)

```
venv/
.env
data/
chroma_db/
__pycache__/
```

Don't commit your notes or API key.
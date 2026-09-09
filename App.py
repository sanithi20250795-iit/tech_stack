"""
STREAMLIT UI
============
This is Day 3's work — a chat interface on top of rag.py. Run with:
    streamlit run app.py

Streamlit re-runs this whole script on every interaction, so we use
st.session_state to remember chat history and avoid reloading the
embedding model / DB connection every single time (that would be slow).
"""

import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from rag import get_collection, answer_question

load_dotenv()

st.set_page_config(page_title="Notes RAG Chatbot", page_icon="📚")
st.title("📚 Ask Your Notes")


@st.cache_resource
def load_collection():
    return get_collection()


@st.cache_resource
def load_groq_client():
    return Groq(api_key=os.environ["GROQ_API_KEY"])


collection = load_collection()
groq_client = load_groq_client()

st.caption(f"{collection.count()} chunks loaded from your notes.")

# --- Chat history lives in session_state, not a variable, ---
# --- because Streamlit re-runs this file top-to-bottom on every click ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something from your notes...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if collection.count() == 0:
            answer = "No notes ingested yet — drop files in data/ and run `python ingest.py` first."
            sources = []
        else:
            with st.spinner("Searching your notes..."):
                answer, sources = answer_question(question, collection, groq_client)

        st.markdown(answer)
        if sources:
            st.caption("Sources: " + ", ".join(sources))

    st.session_state.messages.append({"role": "assistant", "content": answer})

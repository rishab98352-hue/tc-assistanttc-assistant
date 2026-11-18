import os
import numpy as np
import faiss
import streamlit as st
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from drive_sync import fetch_and_chunk_files

st.set_page_config(page_title="T&C Field Assistant", page_icon="⚡", layout="centered")
st.title("⚡ Testing & Commissioning Assistant")
st.caption("Ask questions about SOPs, Relay Settings, Procedures, and Electrical Drawings.")

api_key = os.environ.get("GEMINI_API_KEY", "").strip()
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))
)
FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "").strip()

@st.cache_resource(show_spinner="Loading search engine...")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_active_model():
    candidates = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
    try:
        for m in client.models.list():
            name = m.name.replace("models/", "")
            if "flash" in name.lower() and not any(x in name.lower() for x in ["image", "audio", "tts", "thinking", "exp"]):
                if name not in candidates:
                    candidates.append(name)
    except Exception as e:
        return "gemini-2.5-flash", f"List warning: {e}"

    last_err = None
    for model_name in candidates:
        try:
            client.models.generate_content(model=model_name, contents="ok")
            return model_name, None
        except Exception as e:
            last_err = str(e)
    return "gemini-2.5-flash", last_err

@st.cache_resource(show_spinner="Syncing & indexing Google Drive files...")
def build_vector_index():
    chunks = fetch_and_chunk_files(FOLDER_ID, progress_callback=None, chunk_size=2000, overlap=250)
    if not chunks:
        return None, []
    
    embedder = load_embedder()
    matrix = embedder.encode(chunks, convert_to_numpy=True).astype("float32")
    
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index, chunks

with st.sidebar:
    st.header("Knowledge Base")
    if st.button("🔄 Re-sync Google Drive"):
        st.cache_resource.clear()
        st.rerun()

active_model, model_err = get_active_model()
if model_err:
    st.warning(f"API Key Notice: {model_err}")

try:
    index, chunks = build_vector_index()
except Exception as e:
    st.error(f"Index Error details: {e}")
    index, chunks = None, []

if not chunks:
    st.warning("No readable files found in your Google Drive folder. Check sharing permissions and Folder ID.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("e.g., What is the trip logic or terminal number for 86 lockout relay?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved_context = "No documents indexed yet."
            if chunks and index is not None:
                embedder = load_embedder()
                q_vec = embedder.encode([prompt], convert_to_numpy=True).astype("float32")
                _, indices = index.search(q_vec, k=min(5, len(chunks)))
                retrieved_context = "\n\n---\n\n".join([chunks[i] for i in indices[0]])

            recent_history = "\n".join(
                [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages[-6:-1]]
            )

            system_prompt = (
                "You are an intelligent, helpful Testing & Commissioning (T&C) Engineering Assistant.\n"
                "1. DOCUMENT PRIORITY: Always check the UPLOADED T&C DOCUMENTS below first. Whenever you use facts, "
                "relay settings, dates, or procedures from them, cite the exact Document Name and Page Number.\n"
                "2. MANDATORY SOURCE TRANSPARENCY RULE: If the answer (or any part of the answer) comes from your own "
                "general AI knowledge rather than the UPLOADED T&C DOCUMENTS, you MUST explicitly state at the very "
                "beginning of that section or reply:\n"
                "   '> ℹ️ **Note:** This information is based on general AI engineering knowledge and is not from your uploaded Google Drive documents.'\n"
                "3. HYBRID ANSWERS: If part of your answer comes from the Drive documents and part comes from general AI "
                "knowledge, clearly separate them and apply the note above to the general AI section.\n"
                "4. SAFETY BOUNDARY: Never invent or guess site-specific relay settings, terminal numbers, or project dates "
                "that are not in the Drive files."
            )

            full_prompt = (
                f"{system_prompt}\n\n"
                f"UPLOADED T&C DOCUMENTS:\n{retrieved_context}\n\n"
                f"RECENT CONVERSATION:\n{recent_history}\n\n"
                f"USER MESSAGE: {prompt}"
            )

            try:
                response = client.models.generate_content(
                    model=active_model,
                    contents=full_prompt
                )
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"API Error ({active_model}): {e}")

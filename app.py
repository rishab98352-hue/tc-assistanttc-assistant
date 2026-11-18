import os
import time
import numpy as np
import faiss
import streamlit as st
from google import genai
from drive_sync import fetch_and_chunk_files

st.set_page_config(page_title="T&C Field Assistant", page_icon="⚡", layout="centered")
st.title("⚡ Testing & Commissioning Assistant")
st.caption("Ask questions about SOPs, Relay Settings, Procedures, and Electrical Drawings.")

api_key = os.environ.get("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=api_key)
FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "").strip()

@st.cache_resource(show_spinner="Syncing T&C documents & drawings from Google Drive...")
def build_vector_index():
    chunks = fetch_and_chunk_files(FOLDER_ID, client)
    if not chunks:
        return None, []
    
    # Embed in batches of 50 to stay well within free-tier rate limits
    embeddings = []
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        res = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch
        )
        for emb in res.embeddings:
            embeddings.append(emb.values)
        if i + batch_size < len(chunks):
            time.sleep(2)
    
    matrix = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index, chunks

with st.sidebar:
    st.header("Knowledge Base")
    if st.button("🔄 Re-sync Google Drive"):
        st.cache_resource.clear()
        st.rerun()

try:
    index, chunks = build_vector_index()
except Exception as e:
    st.error(f"API Error details: {e}")
    st.stop()

if not chunks:
    st.warning("No readable files found in your Google Drive folder. Check sharing permissions and Folder ID.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("e.g., What is the trip logic or terminal number for 86 lockout relay?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    q_res = client.models.embed_content(
        model="gemini-embedding-001",
        contents=prompt
    )
    q_vec = np.array([q_res.embeddings[0].values], dtype="float32")
    _, indices = index.search(q_vec, k=min(6, len(chunks)))
    
    retrieved_context = "\n\n---\n\n".join([chunks[i] for i in indices[0]])

    system_prompt = (
        "You are a strict Testing & Commissioning (T&C) Field Engineering Assistant. "
        "Answer questions ONLY using the provided SOPs, manuals, and drawing extracts below. "
        "Always cite the exact Document Name and Page Number (e.g.,). "
        "Present step-by-step procedures clearly using numbered lists. "
        "SAFETY RULE: If the exact value, terminal number, or procedure is NOT explicitly present in the context, "
        "state clearly: 'This information is not found in the uploaded T&C documents.' Never guess or assume."
    )
    full_prompt = f"{system_prompt}\n\nCONTEXT:\n{retrieved_context}\n\nENGINEER'S QUESTION: {prompt}"

    with st.chat_message("assistant"):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt
        )
        st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})

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

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_active_model():
    candidates = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.8-flash"]
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

@st.cache_resource(show_spinner=False)
def build_vector_index():
    progress_bar = st.progress(0, text="Connecting to Google Drive...")
    
    def update_progress(pct, msg):
        progress_bar.progress(min(pct * 0.75, 0.75), text=msg)

    chunks

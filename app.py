import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import re
import os
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Smart Video RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    color: white;
}

[data-testid="stChatMessage"] {
    background-color: #1e293b;
    border-radius: 12px;
    padding: 10px;
}

.stButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.title("Controls")

top_k = st.sidebar.slider("Top Results", 1, 10, 8)
threshold = st.sidebar.slider("Similarity Threshold", 0.1, 1.0, 0.35)
show_chunks = st.sidebar.checkbox("Show Retrieved Chunks", True)

# -----------------------------
# LOAD DATA (SAFE VERSION)
# -----------------------------
@st.cache_resource
def load_data():
    if not os.path.exists("embeddings.joblib"):
        st.error("embeddings.joblib not found")

        st.info("""
👉 This app requires precomputed embeddings.

To fix:
1. Run locally:
   python Chunks.py
   python create_embeddings.py
   python save_embeddings.py

2. Upload embeddings.joblib to GitHub
""")
        st.stop()

    return joblib.load("embeddings.joblib")

df = load_data()

# -----------------------------
# EMBEDDING FUNCTION
# -----------------------------
def create_embedding(text_list):
    try:
        r = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": "bge-m3",
                "input": text_list
            },
            timeout=10
        )
    except:
        st.error("Ollama embedding server not running")
        return []

    if r.status_code != 200:
        st.error("Embedding error")
        return []

    return r.json().get("embeddings", [])

# -----------------------------
# LLM FUNCTION
# -----------------------------
def inference(prompt):
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=20
        )
    except:
        st.error(" Ollama LLM not running")
        return ""

    if r.status_code != 200:
        st.error("LLM error")
        return ""

    return r.json().get("response", "")

# -----------------------------
# TEXT HIGHLIGHT FUNCTION
# -----------------------------
def highlight(text, query):
    for word in query.split():
        text = re.sub(f"({word})", r"**\\1**", text, flags=re.IGNORECASE)
    return text

# -----------------------------
# TITLE
# -----------------------------
st.title("Smart Video RAG Assistant")
st.write("Ask questions from your course videos")

# -----------------------------
# EXAMPLES
# -----------------------------
examples = [
    "What is overflow property?",
    "Difference between overflow and text-overflow",
    "How to handle long text in div",
    "What is overflow x and y",
    "How to add scroll bar in css"
]

selected = st.selectbox("Try an example:", [""] + examples)

# -----------------------------
# CHAT MEMORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask something about your videos...")

if selected:
    user_input = selected

# -----------------------------
# MAIN LOGIC
# -----------------------------
if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    query_embedding = create_embedding([user_input])

    if not query_embedding:
        st.stop()

    query_embedding = query_embedding[0]

    # SIMILARITY
    matrix = np.vstack(df['embedding'])
    similarities = cosine_similarity(matrix, [query_embedding]).flatten()

    # FILTER
    filtered_indices = [i for i, score in enumerate(similarities) if score > threshold]

    # FALLBACK
    if not filtered_indices:
        sorted_indices = np.argsort(similarities)[::-1]
        top_idx = sorted_indices[:top_k]
    else:
        sorted_indices = sorted(filtered_indices, key=lambda i: similarities[i], reverse=True)
        top_idx = sorted_indices[:top_k]

    # CONFIDENCE
    avg_score = np.mean([similarities[i] for i in top_idx])
    if avg_score < threshold:
        st.warning(" Low confidence answer")

    st.write("Max similarity:", float(max(similarities)))

    # CONTEXT
    context = ""
    for idx in top_idx:
        row = df.iloc[idx]
        context += f"""
        Time: {row.get('start', 'NA')} - {row.get('end', 'NA')}
        Content: {row.get('text', '')}
        ---
        """

    # PROMPT
    prompt = f"""
You are a helpful assistant for a web development course.

Use the context below to answer the question.
If context is limited, still give a helpful answer.

Context:
{context}

Question:
{user_input}

Answer clearly and mention timestamps if possible.
"""

    with st.spinner("Generating answer..."):
        answer = inference(prompt)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# -----------------------------
# CHAT DISPLAY
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------
# SOURCES
# -----------------------------
if user_input and show_chunks:

    st.subheader("Sources Used")

    for idx in top_idx:
        row = df.iloc[idx]

        st.write(f"Time: {row.get('start', 'NA')} - {row.get('end', 'NA')}")
        st.progress(float(similarities[idx]))
        st.markdown(highlight(row.get("text", ""), user_input))
        st.write("---")
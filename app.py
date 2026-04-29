import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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
# SIDEBAR
# -----------------------------
st.sidebar.title("Controls")
top_k = st.sidebar.slider("Top Results", 1, 10, 8)
threshold = st.sidebar.slider("Similarity Threshold", 0.1, 1.0, 0.35)
show_chunks = st.sidebar.checkbox("Show Retrieved Chunks", True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_resource
def load_data():
    if not os.path.exists("embeddings.joblib"):
        st.error("embeddings.joblib not found")
        st.stop()
    return joblib.load("embeddings.joblib")

df = load_data()

# -----------------------------
# TF-IDF (CLOUD SAFE)
# -----------------------------
@st.cache_resource
def get_vectorizer(texts):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    return vectorizer

vectorizer = get_vectorizer(df["text"])

def get_query_embedding(query):
    return vectorizer.transform([query]).toarray()[0]

# -----------------------------
# HIGHLIGHT
# -----------------------------
def highlight(text, query):
    for word in query.split():
        text = re.sub(f"({word})", r"**\\1**", text, flags=re.IGNORECASE)
    return text

# -----------------------------
# UI
# -----------------------------
st.title("Smart Video RAG Assistant")
st.write("Ask questions from your course videos")

examples = [
    "What is overflow property?",
    "Difference between overflow and text-overflow",
    "How to handle long text in div",
    "What is overflow x and y",
    "How to add scroll bar in css"
]

selected = st.selectbox("Try an example:", [""] + examples)

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

    query_embedding = get_query_embedding(user_input)

    matrix = vectorizer.transform(df['text']).toarray()

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
        st.warning("Low confidence answer")

    st.write("Max similarity:", float(max(similarities)))

    # -----------------------------
    # CONTEXT → FINAL ANSWER
    # -----------------------------
    context_chunks = []
    for idx in top_idx:
        row = df.iloc[idx]
        context_chunks.append(f"""
Time: {row.get('start', 'NA')} - {row.get('end', 'NA')}
{row.get('text', '')}
""")

    answer = "\n\n".join(context_chunks[:3])

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
# 🚀 Smart Video RAG Assistant

An AI-powered system to **ask questions from video content** and get **accurate, timestamp-grounded answers** using open-source LLMs.

---

## 🔥 Overview

Built an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline on video lectures.

Instead of manually searching through long videos, users can:
- Ask natural language questions
- Get contextual answers
- Jump directly to relevant timestamps

---

## 🧠 Key Features

- 🔍 Semantic search over transcripts  
- ⏱️ Timestamp-based retrieval  
- 🤖 LLM-powered answers (Ollama)  
- 💬 Chat-style UI (Streamlit)  
- ⚙️ End-to-end pipeline (video → text → embeddings → answer)  
- 🆓 Fully open-source (no paid APIs)

---

## 🛠️ Tech Stack

**Python, Pandas, NumPy, Scikit-learn**  
**Ollama (LLM), bge-m3 (Embeddings)**
**Vector search, Cosine similarity**
**Whisper, FFmpeg, NLTK**  
**Joblib, Streamlit**

---

## 📊 Pipeline

Video → Audio → Transcript → Chunks → Embeddings → Retrieval → LLM → Answer

---

## 🎯 Dataset

Tested on:
- **CodeWithHarry – Web Development Playlist (sample video)**

Topics covered:
- CSS Overflow
- Text Overflow
- Handling long text in containers

---

## 💡 Highlights

- Real-world AI use case (education)
- Combines NLP + vector search + LLMs + UI
- Fully reproducible with open-source tools

---

## 🧪 Example Queries

- What is overflow property?
- Difference between overflow and text-overflow
- Where is ellipsis explained?

---

## 🖥️ Run Locally

```bash
python Chunks.py
python create_embeddings.py
python save_embeddings.py
streamlit run app.py
🚀 Future Work
FAISS vector DB
Multi-video support
Persistent chat memory
Deployment (Groq / OpenAI APIs)
👩‍💻 Author

Bhumika Shaw
Data Science @ IIT Madras

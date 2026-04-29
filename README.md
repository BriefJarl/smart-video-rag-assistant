# 🚀 Smart Video RAG Assistant

An AI-powered system that allows users to **ask questions directly from video content** and receive **accurate answers with timestamps**.

---

## 🔥 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline on video data.

Instead of manually searching through long tutorials, users can:

- Ask natural language questions
- Get contextual answers
- Navigate directly to relevant video timestamps

---

## 🧠 Key Features

✅ Semantic search over video transcripts  
✅ Timestamp-based answer retrieval  
✅ LLM-powered contextual responses  
✅ Chat-style interactive UI (Streamlit)  
✅ Multi-step data pipeline (video → text → embeddings → answers)  
✅ Fully open-source (no paid APIs required)

---

## 🛠️ Tech Stack

- **Python**
- **Pandas / NumPy**
- **Scikit-learn (cosine similarity)**
- **Ollama (Local LLM)**
- **bge-m3 (Embedding model)**
- **Whisper (Speech-to-text)**
- **FFmpeg (Audio extraction)**
- **NLTK (Sentence chunking)**
- **Joblib (Fast storage)**
- **Streamlit (UI)**

---

## 📊 Data Pipeline

### 1. Video Processing
- 22 videos converted into audio using FFmpeg
- Audio transcribed into text using Whisper

### 2. Preprocessing
- Converted transcripts → JSON format
- Cleaned text (removed filler words)
- Sentence-based chunking using NLTK

### 3. Embedding
- Used **bge-m3 model** via Ollama
- Stored embeddings using **joblib**

### 4. Retrieval (RAG)
- Cosine similarity search
- Top-K relevant chunks selected
- Context passed to LLM

### 5. Generation
- LLM generates final human-like answer
- Includes timestamp references

---

## 🎯 Sample Dataset

This project was tested on:

👉 A sample video from  
**CodeWithHarry – Web Development Playlist**

Topic includes:
- CSS Overflow
- Text Overflow
- Handling long text in containers

---

## 💡 What Makes This Project Strong

✔ End-to-end ML pipeline  
✔ Real-world use case (education AI)  
✔ Combines:
- NLP
- Vector search
- LLMs
- UI development  

✔ Fully reproducible using open-source tools  

---

## 🚀 How It Works


Video → Audio → Text → Chunks → Embeddings → Retrieval → LLM → Answer


---

## 🧪 Example Queries

- What is overflow property?
- Difference between overflow and text-overflow
- How to handle long text in div?
- Where is ellipsis explained in the video?

---

## 🖥️ Running Locally

```bash
# Step 1
python Chunks.py

# Step 2
python create_embeddings.py

# Step 3
python save_embeddings.py

# Step 4
streamlit run app.py
🌍 Future Improvements
FAISS vector database
Multi-video selection
Chat memory across sessions
Deployment using Groq / OpenAI APIs
Scaling to full YouTube playlists
⚠️ Note

This project uses open-source LLMs (Ollama).
Using advanced APIs like GPT-4/5 can significantly improve accuracy.

👩‍💻 Author

Bhumika Shaw
Data Science @ IIT Madras

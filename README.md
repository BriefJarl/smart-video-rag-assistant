# 🚀 Smart Video RAG Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** system that allows users to query video content using natural language and receive **accurate, context-aware answers with timestamp grounding**.

---

## 🔥 Overview

This project builds a **production-style end-to-end RAG pipeline** on video data, transforming unstructured multimedia into a **searchable knowledge system**.

Instead of manually scrubbing through long videos, users can:

- 💬 Ask questions in natural language  
- 🔍 Retrieve the most relevant video segments  
- ⏱️ Get answers grounded with timestamps  

---

## 🧠 Key Features

- 🔍 **Semantic Search** over video transcripts  
- ⏱️ **Timestamp-aware Retrieval** for precise navigation  
- 🤖 **LLM-powered Answer Generation** (LLaMA 3 via Ollama)  
- 🧩 **Context-aware Chunking Strategy**  
- 💬 **Interactive Chat Interface** (Streamlit)  
- ⚙️ **End-to-End Pipeline**  

Video → Text → Embeddings → Retrieval → Answer

- 🆓 **Fully Open Source (No Paid APIs required)**  

---

## 🛠️ Tech Stack

### 💻 Core
- Python  
- Pandas  
- NumPy  
- Scikit-learn  

### 🤖 AI / NLP
- **LLaMA 3 (via Ollama)** – Local LLM for response generation  
- **BGE-M3 Embeddings** – High-quality vector representations  
- **Whisper** – Speech-to-text transcription  

### 🔍 Retrieval System
- Vector Search (Semantic Retrieval)  
- Cosine Similarity (ranking chunks)  
- Context-preserving Chunking Strategy  

### ⚙️ Data Pipeline
- FFmpeg – Audio extraction from video  
- NLTK – Text preprocessing  
- Joblib – Embedding storage  

### 🌐 Interface
- Streamlit – Interactive UI  

---

## ⚡ System Pipeline


Video → Audio → Transcript → Chunking → Embeddings → Vector Search → LLM → Answer


---

## 📊 Dataset

**Tested on:**
- CodeWithHarry – Web Development Playlist (sample video)

**Example Topics:**
- CSS Overflow  
- Text Overflow  
- Handling long text in containers  

---

## 💡 Engineering Highlights

- 🧱 Designed a **modular RAG architecture** for video data  
- 🔍 Implemented **semantic retrieval using embeddings**  
- 🤖 Built a **context-aware QA system** (retrieval + generation)  
- ⏱️ Enabled **timestamp grounding for explainability**  
- ⚡ Optimized for **local inference using open-source LLMs**  

---

## 🧪 Example Queries

- What is overflow property?  
- Difference between overflow and text-overflow  
- Where is ellipsis explained?  

---

## 🖥️ Run Locally

```bash
# Step 1: Chunk the transcript
python Chunks.py

# Step 2: Create embeddings
python create_embeddings.py

# Step 3: Save embeddings
python save_embeddings.py

# Step 4: Run the app
streamlit run rag_query.py

🚀 Future Improvements
🔹 FAISS / Vector Database integration
🔹 Multi-video knowledge base
🔹 Conversational memory (chat history)
🔹 Cloud deployment (OpenAI / Groq APIs)
🔹 UI/UX enhancements

👩‍💻 Author
Bhumika Shaw
Data Science @ IIT Madras

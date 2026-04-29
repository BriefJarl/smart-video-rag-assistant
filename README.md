🚀 Smart Video RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) system that enables users to query video content using natural language and receive accurate, context-aware answers with timestamp grounding.

🔥 Overview

This project builds a production-style end-to-end RAG pipeline on video data, transforming unstructured multimedia content into a searchable knowledge system.

Instead of manually scrubbing through long videos, users can:

Ask questions in natural language
Retrieve the most relevant video segments
Get LLM-generated answers grounded in context + timestamps
🧠 Key Features
🔍 Semantic Search over video transcripts
⏱️ Timestamp-aware Retrieval for precise navigation
🤖 LLM-powered Answer Generation (LLaMA 3 via Ollama)
🧩 Context-aware Chunking Strategy
💬 Interactive Chat Interface (Streamlit)
⚙️ End-to-End Pipeline (Video → Text → Embeddings → Retrieval → Answer)
🆓 Fully Open-Source (No Paid APIs)
🛠️ Tech Stack
💻 Core
Python, Pandas, NumPy, Scikit-learn
🤖 AI / NLP
LLaMA 3 (via Ollama) – Local LLM for response generation
BGE-M3 Embeddings – High-quality semantic vector representations
Whisper – Speech-to-text transcription
🔍 Retrieval System
Vector Search (Semantic Retrieval)
Cosine Similarity – Ranking relevant chunks
Chunking Strategy – Context-preserving segmentation
⚙️ Data Pipeline
FFmpeg – Audio extraction from video
NLTK – Text preprocessing
Joblib – Efficient embedding storage
🌐 Interface
Streamlit – Interactive query UI
⚡ System Pipeline
Video → Audio → Transcript → Chunking → Embeddings → Vector Search → LLM → Answer
📊 Dataset

Tested on:

CodeWithHarry – Web Development Playlist (sample video)

Example topics:

CSS Overflow
Text Overflow
Handling long text in containers
💡 Engineering Highlights
Designed a modular RAG architecture for unstructured video data
Implemented semantic retrieval pipeline using vector embeddings
Built a context-aware QA system combining retrieval + generation
Enabled timestamp grounding for explainable AI responses
Optimized for local inference using open-source LLMs
🧪 Example Queries
What is overflow property?
Difference between overflow and text-overflow
Where is ellipsis explained?
🖥️ Run Locally
python Chunks.py
python create_embeddings.py
python save_embeddings.py
streamlit run app.py
🚀 Future Improvements
🔹 FAISS / Vector Database integration
🔹 Multi-video knowledge base
🔹 Conversational memory (chat history)
🔹 Cloud deployment (OpenAI / Groq / APIs)
🔹 UI/UX enhancements
👩‍💻 Author

Bhumika Shaw
Data Science @ IIT Madras

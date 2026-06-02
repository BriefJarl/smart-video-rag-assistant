import requests
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

print("Starting RAG System...")
df = joblib.load("embeddings.joblib")

print(f"Loaded {len(df)} chunks")

def create_embedding(text_list):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    if response.status_code != 200:
        print("Embedding error:", response.text)
        return []

    return response.json().get("embeddings", [])

def inference(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",  
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code != 200:
        print("LLM error:", response.text)
        return ""

    return response.json().get("response", "")
query = input("\nAsk a question: ")
print(f"\nQuestion: {query}")

query_embedding = create_embedding([query])

if not query_embedding:
    print("Failed to generate query embedding")
    exit()

query_embedding = query_embedding[0]

matrix = np.vstack(df["embedding"].values)

similarities = cosine_similarity(matrix, [query_embedding]).flatten()

threshold = 0.55

filtered_indices = [i for i, score in enumerate(similarities) if score > threshold]

if not filtered_indices:
    print("\nNo relevant answer found. Try a better question.")
    exit()

# top 5 results
top_k = 5

sorted_indices = sorted(filtered_indices, key=lambda i: similarities[i], reverse=True)

top_idx = sorted_indices[:top_k]

print("\nTop Relevant Chunks:\n")

for idx in top_idx:
    row = df.iloc[idx]

    print(f"Chunk ID : {row['chunk_id']}")
    print(f"Time     : {row.get('start', 'NA')} - {row.get('end', 'NA')}")
    print(f"Score    : {similarities[idx]:.4f}")
    print("Text     :")
    print(row['text'])
    print("\n" + "=" * 60 + "\n")
context = ""

for idx in top_idx:
    row = df.iloc[idx]
    context += f"""
Video Title: {row.get('title', 'NA')}
Video Number: {row.get('number', 'NA')}
Time: {row.get('start', 'NA')} - {row.get('end', 'NA')}
Content: {row.get('text', '')}
---
"""
prompt = f"""
You are a helpful assistant for a web development course.

Use ONLY the context below to answer.

Context:
{context}

User Question:
{query}

Instructions:
- Answer clearly and naturally
- Mention which video and timestamp to watch
- Do NOT mention 'context' or 'chunks'
- If answer not found, say: "This is not covered in the course"

Final Answer:
"""
response = inference(prompt)

print("\nFinal Answer:\n")
print(response)

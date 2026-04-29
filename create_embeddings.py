import json
import requests

print("Creating embeddings from clean chunks...")

# -----------------------------
# LOAD CLEAN CHUNKS
# -----------------------------
with open("clean_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} clean chunks")


# -----------------------------
# EMBEDDING FUNCTION
# -----------------------------
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


# -----------------------------
# GENERATE EMBEDDINGS
# -----------------------------
texts = [c["text"] for c in data]

print("Generating embeddings...")

embeddings = create_embedding(texts)

if not embeddings:
    print("Failed to generate embeddings")
    exit()


# -----------------------------
# COMBINE DATA + EMBEDDINGS
# -----------------------------
final_data = []

for i, chunk in enumerate(data):
    final_data.append({
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],
        "embedding": embeddings[i]
    })


# -----------------------------
# SAVE FILE
# -----------------------------
with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f)

print("Embeddings saved as embeddings.json")
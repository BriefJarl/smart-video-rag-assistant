import requests
import json

print("Starting embedding process...")

# Load your JSON file
with open("output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

chunks = data["chunks"] if isinstance(data, dict) else data

print(f"Total chunks: {len(chunks)}")

# Function to create embedding
def create_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "bge-m3",
            "prompt": text
        }
    )
    return response.json()["embedding"]

# Store embeddings
embeddings = []

for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i}...")

    emb = create_embedding(chunk["text"])

    embeddings.append({
        "id": i,
        "text": chunk["text"],
        "embedding": emb
    })

# Save embeddings
with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embeddings, f, indent=4)

print("Embeddings created successfully!")
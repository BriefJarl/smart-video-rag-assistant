import requests
import json
import pandas as pd

print("Starting advanced embedding process...")

# Loading json.....
with open("output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

chunks = data["chunks"] if isinstance(data, dict) else data

chunks = sorted(chunks, key=lambda x: x["start"])

print(f"Total chunks: {len(chunks)}")
def create_embedding(text_list):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    if response.status_code != 200:
        print("Error:", response.text)
        return []

    return response.json()["embeddings"]

texts = [chunk["text"] for chunk in chunks]

print("Creating embeddings in batch...")

embeddings = create_embedding(texts)

my_dicts = []

for i, chunk in enumerate(chunks):
    chunk_data = {
        "chunk_id": i,
        "start": chunk["start"],
        "end": chunk["end"],
        "text": chunk["text"],
        "embedding": embeddings[i]
    }
    my_dicts.append(chunk_data)

print("Dictionary created")

df = pd.DataFrame.from_records(my_dicts)

print("\n DataFrame Preview:")
print(df.head())


with open("final_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(my_dicts, f, indent=4)

print("Saved as final_embeddings.json")

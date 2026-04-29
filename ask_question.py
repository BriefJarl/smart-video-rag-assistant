import requests
import json

print("Ask your question")

query = input("Ask a question: ")

def create_query_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": [text]   
        }
    )

    if response.status_code != 200:
        print("Error:", response.text)
        return None

    return response.json()["embeddings"][0]  


query_embedding = create_query_embedding(query)

print("\nQuery Embedding:")
print(query_embedding)
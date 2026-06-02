import json
import nltk
from nltk.tokenize import sent_tokenize
def clean_text(text):
    fillers = ["uh", "um", "you know", "like", "okay", "so", "hmm"]

    text = text.lower()

    for f in fillers:
        text = text.replace(f, "")

    return text.strip()

def create_chunks_with_time(data, max_chunk_size=300):
    final_chunks = []
    chunk_id = 0

    for item in data["chunks"]:
        text = item["text"]
        start = item.get("start", "NA")
        end = item.get("end", "NA")

        sentences = sent_tokenize(text)

        current_chunk = ""

        for sentence in sentences:
            sentence = clean_text(sentence)

            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += " " + sentence
            else:
                final_chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip(),
                    "start": start,
                    "end": end
                })
                chunk_id += 1
                current_chunk = sentence

        if current_chunk:
            final_chunks.append({
                "chunk_id": chunk_id,
                "text": current_chunk.strip(),
                "start": start,
                "end": end
            })
            chunk_id += 1

    return final_chunks

print("Loading original chunks...")

with open("output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data['chunks'])} raw chunks")

print("Creating clean chunks with timestamps...")

clean_chunks = create_chunks_with_time(data)

print(f"Created {len(clean_chunks)} clean chunks")

with open("clean_chunks.json", "w", encoding="utf-8") as f:
    json.dump(clean_chunks, f, indent=4)

print("Clean chunks with timestamps saved")

import whisper
import json

print("Starting...")

# Load model
model = whisper.load_model("base")
print("Model loaded")
result = model.transcribe(
    audio="Audios/sample.mp3",
    task="translate",
    word_timestamps=False
)

print("Transcription done")

chunks = []

for i, segment in enumerate(result["segments"]):
    chunks.append({
        "id": i,
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"]
    })

final_output = {
    "full_text": result["text"],
    "chunks": chunks
}
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=4)

print("JSON file created successfully!")

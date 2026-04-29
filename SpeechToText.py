import whisper
import os

model = whisper.load_model("base")

AUDIO_FOLDER = "Audios"
OUTPUT_FOLDER = "Transcripts"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for file in os.listdir(AUDIO_FOLDER):
    if file.endswith(".mp3"):
        file_path = os.path.join(AUDIO_FOLDER, file)

        print(f"Processing: {file}")

        result = model.transcribe(file_path)

        text = result["text"]

        output_file = os.path.join(
            OUTPUT_FOLDER, file.replace(".mp3", ".txt")
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Saved: {output_file}")
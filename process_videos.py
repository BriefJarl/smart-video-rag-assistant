import os
import subprocess

files = os.listdir("Video_data")

for index, file in enumerate(files, start=1):
    if file.endswith(".mp4"):
        file_name = os.path.splitext(file)[0]
        
        print(index, file_name)
        
        subprocess.run([
            "ffmpeg",
            "-i", f"Video_data/{file}",
            f"Audios/{index}_{file_name}.mp3"
        ])
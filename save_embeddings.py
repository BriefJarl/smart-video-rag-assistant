import json
import pandas as pd
import joblib

print("Saving embeddings to joblib...")

with open("embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

joblib.dump(df, "embeddings.joblib")

print("Saved as embeddings.joblib")
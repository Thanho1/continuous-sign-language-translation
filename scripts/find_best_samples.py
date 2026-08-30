import pandas as pd
from difflib import SequenceMatcher

CSV_PATH = "outputs/predictions.csv"

df = pd.read_csv(CSV_PATH)

def similarity(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

df["similarity"] = df.apply(
    lambda row: similarity(row["Ground Truth"], row["Prediction"]),
    axis=1
)

df = df.sort_values("similarity", ascending=False)

print("=" * 100)
print("TOP 10 SAMPLES GAN GT NHAT")
print("=" * 100)

for i, row in df.head(10).iterrows():
    print(f"\nIndex: {i}")
    print(f"GT   : {row['Ground Truth']}")
    print(f"PRED : {row['Prediction']}")
    print(f"SIM  : {row['similarity']:.3f}")
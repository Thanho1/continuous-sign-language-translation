import pandas as pd
import re

df = pd.read_csv("outputs/predictions.csv")

def common_words(gt, pred):
    gt_words = set(re.findall(r"\b\w+\b", gt.lower()))
    pred_words = set(re.findall(r"\b\w+\b", pred.lower()))
    return gt_words & pred_words

results = []

for i, row in df.iterrows():
    common = common_words(row["Ground Truth"], row["Prediction"])

    if len(common) >= 2:
        results.append(
            (
                i,
                len(common),
                row["Ground Truth"],
                row["Prediction"],
                common,
            )
        )

results.sort(key=lambda x: x[1], reverse=True)

print("=" * 100)
print("CAC SAMPLE CO NHIEU TU CHUNG")
print("=" * 100)

for i, n, gt, pred, common in results[:20]:
    print(f"\nIndex: {i}")
    print(f"GT   : {gt}")
    print(f"PRED : {pred}")
    print(f"Common words: {sorted(common)}")
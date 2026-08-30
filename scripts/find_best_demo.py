import pandas as pd
from difflib import SequenceMatcher

PRED = "outputs/predictions.csv"
MANIFEST = "data/processed/manifest.csv"


def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


pred = pd.read_csv(PRED)
manifest = pd.read_csv(MANIFEST)

print("Predictions:", len(pred))
print("Manifest:", len(manifest))
print("\nManifest split:")
print(manifest["split"].value_counts())

# Lấy đúng 500 test
test_manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)

results = []

for i in range(min(len(pred), len(test_manifest))):
    gt = str(pred.iloc[i]["Ground Truth"])
    pred_text = str(pred.iloc[i]["Prediction"])

    sim = similarity(gt, pred_text)

    results.append({
        "index": i,
        "GT": gt,
        "PRED": pred_text,
        "similarity": sim,
        "npy_path": test_manifest.iloc[i]["npy_path"]
    })

results = sorted(
    results,
    key=lambda x: x["similarity"],
    reverse=True
)

print("\n" + "=" * 80)
print("TOP 20 SAMPLE TOT NHAT")
print("=" * 80)

for x in results[:20]:
    print(f"\nIndex: {x['index']}")
    print(f"GT   : {x['GT']}")
    print(f"PRED : {x['PRED']}")
    print(f"SIM  : {x['similarity']:.3f}")
    print(f"NPY  : {x['npy_path']}")
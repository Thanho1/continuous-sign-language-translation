"""
Chay lai dung logic loc (MIN_WORDS=3) tren split test cua bdanko dataset,
de lay lai video_id KHOP DUNG THU TU voi predictions.csv da co.

Chay: python -m src.data.build_test_video_ids
"""
import csv
import os

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"

from datasets import load_dataset

DATASET_NAME = "bdanko/how2sign-landmarks-front-raw-parquet"
N_TEST = 400  # PHAI khop dung so luong da dung luc build_manifest_bdanko.py
MIN_WORDS = 3
OUT_CSV = "data/processed/test_video_ids.csv"


def main():
    os.makedirs("data/processed", exist_ok=True)
    ds = load_dataset(DATASET_NAME, split="test", streaming=True)

    rows = []
    kept = 0
    for sample in ds:
        if kept >= N_TEST:
            break
        text = sample["sentence"]
        if len(text.split()) < MIN_WORDS:
            continue
        rows.append({"index": kept, "video_id": sample["video_id"], "sentence": text})
        kept += 1

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "video_id", "sentence"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Da luu {len(rows)} dong vao {OUT_CSV}")
    print("Dong dau tien:", rows[0])


if __name__ == "__main__":
    main()
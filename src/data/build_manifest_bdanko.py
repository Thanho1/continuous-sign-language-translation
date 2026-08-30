"""
Tai subset tu bdanko/how2sign-landmarks-front-raw-parquet, CAT BO phan face
(chi giu pose + 2 tay = 225 chieu), chuan hoa, luu .npy + manifest.csv.

CAP NHAT QUAN TRONG:
- MAX_FRAMES tang tu 150 -> 256 (percentile 50=130, 90=321 -> 150 cu cat mat qua nhieu noi dung cau dai)
- Loc bo cau qua ngan (< 3 tu) vi thuong la loi can chinh du lieu goc

Chay: python -m src.data.build_manifest_bdanko --n_train 14000 --n_val 500 --n_test 500
(dat n cao hon muc tieu vi se loc bot mot phan do MIN_WORDS)
"""
import argparse
import csv
import os

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"

import numpy as np
from datasets import load_dataset

DATASET_NAME = "bdanko/how2sign-landmarks-front-raw-parquet"
OUT_ROOT = "data/processed/keypoints_bdanko"
MANIFEST_CSV = "data/processed/manifest.csv"

MAX_FRAMES = 256   # DA TANG tu 150 -> 256, phai dong bo voi extract_keypoints.py va cac file inference
MIN_WORDS = 3       # loc bo cau qua ngan (thuong la loi can chinh du lieu goc, chi ~2.1% tong so)

# Layout da xac nhan: pose(33) + face(468) + tay trai(21) + tay phai(21) = 543
POSE_SLICE = slice(0, 33)
LEFT_HAND_SLICE = slice(501, 522)
RIGHT_HAND_SLICE = slice(522, 543)


def decode_sample(sample):
    shape = tuple(sample["shape"])
    arr = np.frombuffer(sample["features"], dtype=np.float32).reshape(shape)
    return arr


def slice_pose_hands(arr: np.ndarray) -> np.ndarray:
    pose = arr[:, POSE_SLICE, :]
    left = arr[:, LEFT_HAND_SLICE, :]
    right = arr[:, RIGHT_HAND_SLICE, :]
    combined = np.concatenate([pose, left, right], axis=1)
    return combined.reshape(combined.shape[0], -1)


def normalize(sequence: np.ndarray) -> np.ndarray:
    left_shoulder_idx = 11 * 3
    right_shoulder_idx = 12 * 3
    center = (sequence[:, left_shoulder_idx:left_shoulder_idx + 2] +
              sequence[:, right_shoulder_idx:right_shoulder_idx + 2]) / 2
    sequence = sequence.copy()
    for i in range(0, sequence.shape[1], 3):
        sequence[:, i:i + 2] -= center
    return sequence


def pad_or_truncate(sequence: np.ndarray, max_len: int) -> np.ndarray:
    T = sequence.shape[0]
    if T >= max_len:
        return sequence[:max_len]
    pad = np.zeros((max_len - T, sequence.shape[1]))
    return np.concatenate([sequence, pad], axis=0)


def process_split(split: str, n: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset(DATASET_NAME, split=split, streaming=True)

    rows = []
    skipped_short = 0
    kept = 0

    for sample in ds:
        if kept >= n:
            break

        text = sample["sentence"]
        if len(text.split()) < MIN_WORDS:
            skipped_short += 1
            continue

        arr = decode_sample(sample)
        seq = slice_pose_hands(arr)
        seq = normalize(seq)
        seq = pad_or_truncate(seq, MAX_FRAMES)

        npy_path = os.path.join(out_dir, f"{split}_{kept:04d}.npy")
        np.save(npy_path, seq.astype(np.float32))

        rows.append({"split": split, "npy_path": npy_path, "text": text})
        kept += 1
        print(f"[{split} {kept}/{n}] {sample['video_id']} -> {text[:50]}...")

    print(f"  -> Da bo qua {skipped_short} cau qua ngan (< {MIN_WORDS} tu) trong split '{split}'")
    return rows


def main(n_train: int, n_val: int, n_test: int):
    os.makedirs(os.path.dirname(MANIFEST_CSV), exist_ok=True)
    all_rows = []
    all_rows += process_split("test", n_test, OUT_ROOT)
    all_rows += process_split("validation", n_val, OUT_ROOT)
    all_rows += process_split("train", n_train, OUT_ROOT)

    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "npy_path", "text"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nHoan tat. Tong {len(all_rows)} mau. Manifest: {MANIFEST_CSV}")
    print(f"MAX_FRAMES da dung: {MAX_FRAMES} (nho dong bo gia tri nay voi extract_keypoints.py!)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=12000)
    parser.add_argument("--n_val", type=int, default=400)
    parser.add_argument("--n_test", type=int, default=400)
    args = parser.parse_args()
    main(args.n_train, args.n_val, args.n_test)
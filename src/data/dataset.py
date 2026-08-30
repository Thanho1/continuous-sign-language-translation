"""
PyTorch Dataset doc du lieu tu manifest.csv (da tao boi build_manifest_bdanko.py).
Dung chung cho ca train/val/test, loc theo cot 'split'.
"""
import csv

import numpy as np
import torch
from torch.utils.data import Dataset


class SLTDataset(Dataset):
    def __init__(self, manifest_csv: str, split: str, tokenizer, max_text_len: int = 40):
        with open(manifest_csv, "r", encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))
        self.rows = [r for r in all_rows if r["split"] == split]
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        if len(self.rows) == 0:
            raise ValueError(
                f"Khong tim thay mau nao voi split='{split}' trong {manifest_csv}. "
                f"Kiem tra lai cot 'split' co dung gia tri 'train'/'validation'/'test' khong."
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        pose_seq = np.load(row["npy_path"]).astype(np.float32)
        text = row["text"]
        return pose_seq, text

    def collate_fn(self, batch):
        pose_seqs, texts = zip(*batch)
        pose_tensor = torch.tensor(np.stack(pose_seqs))

        encoded = self.tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=self.max_text_len, return_tensors="pt",
        )
        return pose_tensor, encoded["input_ids"], encoded["attention_mask"]


if __name__ == "__main__":
    # Kiem tra nhanh - chi chay duoc SAU KHI da co manifest.csv (tu build_manifest_bdanko.py)
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ds = SLTDataset("data/processed/manifest.csv", split="train", tokenizer=tokenizer)
    print(f"So mau trong split 'train': {len(ds)}")

    pose_seq, text = ds[0]
    print(f"Shape pose_seq mau dau tien: {pose_seq.shape}")
    print(f"Text mau dau tien: {text}")
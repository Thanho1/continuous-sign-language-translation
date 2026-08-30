"""
Danh gia model tren tap test: sinh cau, tinh BLEU + WER, luu bang so sanh.

Chay: python -m src.evaluate
"""
import csv
import os

import sacrebleu
import torch
from jiwer import wer

from src.data.dataset import SLTDataset
from src.models.gloss_free_model import GlossFreeSLTModel

MANIFEST_CSV = "data/processed/manifest.csv"
CHECKPOINT_PATH = "checkpoints/best.pt"
OUTPUT_CSV = "outputs/predictions.csv"


def main():
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GlossFreeSLTModel(unfreeze_last_n_blocks=2).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"Da load checkpoint: {CHECKPOINT_PATH}")

    test_ds = SLTDataset(MANIFEST_CSV, split="test", tokenizer=model.tokenizer)
    print(f"So mau test: {len(test_ds)}")

    predictions, references = [], []
    rows = []

    for i in range(len(test_ds)):
        pose_seq, reference_text = test_ds[i]
        pose_tensor = torch.tensor(pose_seq).unsqueeze(0).to(device)  # (1, T, 225)

        generated = model.generate(pose_tensor, max_new_tokens=40)[0]

        predictions.append(generated)
        references.append(reference_text)
        rows.append({"Ground Truth": reference_text, "Prediction": generated})
        print(f"[{i+1}/{len(test_ds)}] GT: {reference_text[:50]}...")
        print(f"         PRED: {generated[:50]}...")

    # BLEU (sacrebleu can list-of-list cho references)
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    # WER (jiwer)
    error_rate = wer(references, predictions)

    print(f"\n=== KET QUA DANH GIA TREN TAP TEST ({len(test_ds)} mau) ===")
    print(f"BLEU  = {bleu.score:.2f}")
    print(f"WER   = {error_rate:.4f}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Ground Truth", "Prediction"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nBang so sanh du luu tai: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
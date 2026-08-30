"""
So sanh nhanh 3 cau hinh decoding tren MOT SUBSET NHO cua validation set
(khong dung test set - dung nguyen tac khoa hoc: tune tren val, danh gia cuoi cung tren test).

Chay: python -m src.tune_decoding
"""
import sacrebleu
import torch

from src.data.dataset import SLTDataset
from src.models.gloss_free_model import GlossFreeSLTModel

MANIFEST_CSV = "data/processed/manifest.csv"
CHECKPOINT_PATH = "checkpoints/best.pt"
N_SAMPLES = 100  # chi dung 100 mau val de so sanh nhanh, khong dung het 400

CONFIGS = {
    "Greedy (baseline)": {"num_beams": 1},
    "Beam=4": {"num_beams": 4},
    "Beam=6": {"num_beams": 6},
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GlossFreeSLTModel(unfreeze_last_n_blocks=2).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    val_ds = SLTDataset(MANIFEST_CSV, split="validation", tokenizer=model.tokenizer)
    n = min(N_SAMPLES, len(val_ds))
    print(f"Dang so sanh tren {n} mau validation (khong dung test set)\n")

    references = [val_ds[i][1] for i in range(n)]

    results = {}
    for name, cfg in CONFIGS.items():
        predictions = []
        for i in range(n):
            pose_seq, _ = val_ds[i]
            pose_tensor = torch.tensor(pose_seq).unsqueeze(0).to(device)
            pred = model.generate(pose_tensor, num_beams=cfg["num_beams"])[0]
            predictions.append(pred)

        bleu = sacrebleu.corpus_bleu(predictions, [references])
        results[name] = bleu.score
        print(f"{name}: BLEU = {bleu.score:.2f}")

    best_name = max(results, key=results.get)
    print(f"\n=> Cau hinh tot nhat tren validation: {best_name} (BLEU={results[best_name]:.2f})")
    print("Dung cau hinh nay lam mac dinh, roi chay evaluate.py TREN TEST SET 1 LAN DUY NHAT.")


if __name__ == "__main__":
    main()
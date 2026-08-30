import torch

from src.data.dataset import SLTDataset
from src.models.gloss_free_model import GlossFreeSLTModel

MANIFEST = "data/processed/manifest.csv"
CHECKPOINT = "checkpoints/best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GlossFreeSLTModel().to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

dataset = SLTDataset(
    MANIFEST,
    split="test",
    tokenizer=model.tokenizer
)

for i in range(10):
    pose_seq, text = dataset[i]

    x = torch.tensor(pose_seq, dtype=torch.float32)
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model.generate(
            x,
            max_new_tokens=40,
            num_beams=4
        )[0]

    print("=" * 70)
    print(f"Sample {i}")
    print("GT   :", text)
    print("PRED :", pred)
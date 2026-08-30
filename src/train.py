import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SLTDataset
from src.models.gloss_free_model import GlossFreeSLTModel


MANIFEST = "data/processed/manifest.csv"
CHECKPOINT_DIR = "checkpoints"

EPOCHS = 20
BATCH_SIZE = 16

LR_POSE = 5e-4
LR_GPT2 = 2e-5

WEIGHT_DECAY = 0.01
PATIENCE = 4
GRAD_CLIP = 1.0


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for pose, ids, mask in loader:
            pose = pose.to(device)
            ids = ids.to(device)
            mask = mask.to(device)

            loss = model(pose, ids, mask).loss
            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    model = GlossFreeSLTModel(
        unfreeze_last_n_blocks=2
    ).to(device)

    train_ds = SLTDataset(
        MANIFEST,
        split="train",
        tokenizer=model.tokenizer,
    )

    val_ds = SLTDataset(
        MANIFEST,
        split="validation",
        tokenizer=model.tokenizer,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_ds.collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=val_ds.collate_fn,
        pin_memory=True,
    )

    print(f"Train: {len(train_ds)}")
    print(f"Val:   {len(val_ds)}")

    pose_params = list(model.pose_encoder.parameters())

    gpt2_params = [
        p for p in model.gpt2.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": pose_params, "lr": LR_POSE},
            {"params": gpt2_params, "lr": LR_GPT2},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_loss = float("inf")
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
        )

        for pose, ids, mask in progress:
            pose = pose.to(device)
            ids = ids.to(device)
            mask = mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss = model(pose, ids, mask).loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}: "
            f"train={train_loss:.4f} "
            f"val={val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0

            torch.save(
                model.state_dict(),
                f"{CHECKPOINT_DIR}/best.pt",
            )

            print(
                f"  Best checkpoint saved: "
                f"{best_loss:.4f}"
            )
        else:
            no_improve += 1
            print(
                f"  No improvement "
                f"({no_improve}/{PATIENCE})"
            )

            if no_improve >= PATIENCE:
                print("Early stopping.")
                break

    torch.save(
        model.state_dict(),
        f"{CHECKPOINT_DIR}/last.pt",
    )

    print(f"\nBest val loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
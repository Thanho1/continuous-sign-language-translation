import torch
import torch.nn as nn

POSE_DIM = 225
DROPOUT = 0.2


class PoseEncoder(nn.Module):
    def __init__(self, pose_dim=POSE_DIM, hidden_dim=768):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(pose_dim, 256, 5, 2, 2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(DROPOUT),

            nn.Conv1d(256, 512, 5, 2, 2),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(DROPOUT),

            nn.Conv1d(512, hidden_dim, 5, 2, 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # (B, T, 225) -> (B, 225, T)
        x = x.permute(0, 2, 1)

        x = self.conv(x)

        # (B, 768, T') -> (B, T', 768)
        x = x.permute(0, 2, 1)

        return self.norm(x)
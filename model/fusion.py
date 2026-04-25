import torch
import torch.nn as nn

class AdaptiveFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weights = nn.Sequential(
            nn.Linear(dim * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
            nn.Softmax(dim=1)
        )

    def forward(self, zernike, dct, lbp, swin, vit):
        combined = torch.cat([zernike, dct, lbp, swin, vit], dim=1)
        w = self.weights(combined)

        fused = (
            w[:, 0:1] * zernike +
            w[:, 1:2] * dct +
            w[:, 2:3] * lbp +
            w[:, 3:4] * swin +
            w[:, 4:5] * vit
        )

        return fused
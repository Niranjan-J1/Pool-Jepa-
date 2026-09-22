import torch
import torch.nn as nn
import torch.nn.functional as F


class SimNorm(nn.Module):
    """Partition into simplices and apply softmax per simplex (TD-MPC2 style)."""

    def __init__(self, dim: int = 64, num_simplices: int = 8):
        super().__init__()
        self.num_simplices = num_simplices
        self.simplex_dim = dim // num_simplices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim) -> reshape to (batch, num_simplices, simplex_dim)
        shape = x.shape[:-1]
        x = x.view(*shape, self.num_simplices, self.simplex_dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shape, -1)


class VisualEncoder(nn.Module):
    """Small CNN encoder: 128x128x3 -> 64-dim latent with SimNorm."""

    def __init__(self, latent_dim: int = 64, num_simplices: int = 8):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # 128 / (2^4) = 8, so output is (batch, 256, 8, 8)
        self.flatten_dim = 256 * 8 * 8

        self.projection = nn.Linear(self.flatten_dim, latent_dim)
        self.norm = SimNorm(latent_dim, num_simplices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 3, 128, 128) float in [0, 1]
        h = self.conv(x)
        h = h.flatten(1)
        h = self.projection(h)
        h = self.norm(h)
        return h


class LatentPredictor(nn.Module):
    """MLP: (latent, action) -> predicted next latent. 3 layers, LayerNorm + Mish."""

    def __init__(self, latent_dim: int = 64, action_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, action], dim=-1)
        return self.net(x)

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


class SpatialSoftmax(nn.Module):
    """Feature maps (B, K, H, W) -> expected (x, y) per channel, (B, 2K) in [-1, 1].

    Each channel is turned into a probability map with a softmax over all pixels
    (learnable temperature), and the output is that map's centre of mass. A channel
    that fires on one small object therefore reports where that object is, instead of
    the object being averaged away by a flatten + linear projection (Levine et al., 2016).
    Output layout: [x_0, y_0, x_1, y_1, ...].

    Each map is standardized per image before the softmax. Background-subtracted frames
    are mostly zero, so raw feature maps vary by ~1e-3 and the softmax would be flat
    (every keypoint stuck at the centre). Standardized, an object is a z-score peak (~6
    at init), and keypoints track frame content from the first step.
    """

    def __init__(self, init_temperature: float = 0.5):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(init_temperature)).log())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, k, hh, ww = h.shape
        flat = h.flatten(2)
        flat = (flat - flat.mean(-1, keepdim=True)) / (flat.std(-1, keepdim=True) + 1e-6)
        attn = F.softmax(flat / self.log_temperature.exp(), dim=-1).view(b, k, hh, ww)
        ys = torch.linspace(-1.0, 1.0, hh, device=h.device, dtype=h.dtype)
        xs = torch.linspace(-1.0, 1.0, ww, device=h.device, dtype=h.dtype)
        x = (attn.sum(2) * xs).sum(-1)  # marginal over rows -> expected column
        y = (attn.sum(3) * ys).sum(-1)  # marginal over columns -> expected row
        return torch.stack([x, y], dim=-1).flatten(1)


class VisualEncoder(nn.Module):
    """Small CNN encoder: HxWx(3*k) frames -> latent.

    in_channels=3 encodes a single RGB frame; in_channels=6 encodes two stacked frames
    (previous, current) so the latent can carry velocity.

    head="linear": four stride-2 convs, flatten, linear projection to latent_dim,
        optionally SimNorm (use_simnorm). Unbounded unless SimNorm is on.
    head="spatial_softmax": three stride-2 convs + one stride-1 conv (stride 8 overall,
        e.g. 240x432 -> 30x54), a 1x1 conv to latent_dim // 2 keypoint channels, then
        SpatialSoftmax. The latent is the (x, y) of each keypoint, bounded to [-1, 1].
    """

    def __init__(self, latent_dim: int = 64, num_simplices: int = 8, in_channels: int = 3,
                 use_simnorm: bool = True, input_hw: tuple[int, int] = (128, 128),
                 head: str = "linear"):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.head = head

        if head == "linear":
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.ReLU(),
            )
            # Four stride-2 convs: 128x128 -> 8x8, 240x432 -> 15x27.
            with torch.no_grad():
                self.flatten_dim = self.conv(torch.zeros(1, in_channels, *input_hw)).numel()
            self.projection = nn.Linear(self.flatten_dim, latent_dim)
            self.norm = SimNorm(latent_dim, num_simplices) if use_simnorm else nn.Identity()
        elif head == "spatial_softmax":
            if latent_dim % 2:
                raise ValueError("spatial_softmax needs an even latent_dim (x, y per keypoint)")
            self.num_keypoints = latent_dim // 2
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, self.num_keypoints, 1),
            )
            self.spatial_softmax = SpatialSoftmax()
        else:
            raise ValueError(f"Unknown head: {head}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, H, W) float
        h = self.conv(x)
        if self.head == "spatial_softmax":
            return self.spatial_softmax(h)
        h = self.projection(h.flatten(1))
        return self.norm(h)


class LatentPredictor(nn.Module):
    """MLP: (latent, action) -> predicted next latent. 3 layers, LayerNorm + Mish.

    residual=False: outputs z_{t+1} directly (original; has to re-learn the identity).
    residual=True:  outputs z_t + f(z_t, a_t). The last layer is zero-initialized, so an
        untrained residual predictor is exactly "nothing changes" and training only has to
        learn the motion. The flag is stored as a buffer, so load_predictor() can rebuild
        the right variant from a checkpoint.
    """

    def __init__(self, latent_dim: int = 64, action_dim: int = 4, hidden_dim: int = 128,
                 residual: bool = False):
        super().__init__()
        self.residual = residual
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, latent_dim),
        )
        if residual:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            self.register_buffer("residual_flag", torch.tensor(1))

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat([z, action], dim=-1))
        return z + out if self.residual else out


def load_predictor(path: str, device="cpu", latent_dim: int = 64, action_dim: int = 4,
                   hidden_dim: int = 128) -> LatentPredictor:
    """Load a LatentPredictor checkpoint, detecting residual vs. direct from its contents."""
    state = torch.load(path, map_location=device)
    model = LatentPredictor(latent_dim, action_dim, hidden_dim, residual="residual_flag" in state)
    model.load_state_dict(state)
    return model.to(device).eval()

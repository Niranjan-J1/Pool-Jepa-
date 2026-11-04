"""Shot-outcome model: one jump from 'just before the strike' to 'every ball at rest'.

    z_rest = z_pre + f(z_pre, fire_force)

z_pre is the frozen encoder's latent of the two frames before the strike (robot already
aimed), z_rest the latent of the settled scene (the rest frame stacked with itself). Every
training example is a strike and there is no step-by-step rollout, so a strike's effect is
neither rare in the data nor diluted by compounding one-step errors.
"""
import json
import os

import torch
import torch.nn as nn


class ShotModel(nn.Module):
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Start as "nothing moves"; training learns the displacement.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_pre: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        """z_pre: (N, D); force: (N,) or (N, 1) in [0, 1]. Returns predicted z_rest (N, D)."""
        force = force.reshape(-1, 1).to(z_pre)
        return z_pre + self.net(torch.cat([z_pre, force], dim=-1))


def save_shot_model(model: ShotModel, path: str, meta: dict):
    torch.save(model.state_dict(), path)
    with open(path.replace(".pt", ".json"), "w") as fh:
        json.dump(meta, fh, indent=1)


def load_shot_model(path: str, device="cpu") -> tuple[ShotModel, dict]:
    """Returns (model, metadata). Metadata names the frozen encoder file the latents came from."""
    state = torch.load(path, map_location=device)
    hidden = state["net.0.weight"].shape[0]
    latent = state["net.0.weight"].shape[1] - 1
    model = ShotModel(latent, hidden)
    model.load_state_dict(state)
    meta_path = path.replace(".pt", ".json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return model.to(device).eval(), meta

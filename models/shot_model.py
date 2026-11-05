"""Shot-outcome model: one jump from 'just before the strike' to 'every ball at rest'.

    z_rest = z_pre + f(z_pre, fire_force)

z_pre is the frozen encoder's latent of the two frames before the strike (robot already
aimed), z_rest the latent of the settled scene (the rest frame stacked with itself). Every
training example is a strike and there is no step-by-step rollout, so a strike's effect is
neither rare in the data nor diluted by compounding one-step errors.

use_heading=True adds the robot heading as (sin, cos) input:

    z_rest = z_pre + f(z_pre, sin(theta), cos(theta), fire_force)

The strike goes along the heading, but the spatial-softmax latent encodes heading poorly
(linear R^2 ~0.27 vs 0.99 for the cue ball), so without it the model can only average over
directions and predicts almost no motion. On hardware the heading comes from the robot's
marker tracker (perception layer); in simulation from the true robot pose.
"""
import json
import os

import torch
import torch.nn as nn


class ShotModel(nn.Module):
    """MLP of `depth` hidden blocks (Linear -> LayerNorm -> Mish) of width hidden_dim, then a
    zero-initialised output layer. depth=3, hidden_dim=256 is the original model; any
    width/depth keeps the same parameter names (net.<i>.*), so load_shot_model reads both."""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, use_heading: bool = False,
                 depth: int = 3):
        super().__init__()
        self.use_heading = use_heading
        extra = 3 if use_heading else 1  # (sin, cos, force) or (force,)
        layers, width_in = [], latent_dim + extra
        for _ in range(depth):
            layers += [nn.Linear(width_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish()]
            width_in = hidden_dim
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)
        # Start as "nothing moves"; training learns the displacement.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        if use_heading:
            self.register_buffer("heading_flag", torch.tensor(1))

    def forward(self, z_pre: torch.Tensor, force: torch.Tensor,
                heading: torch.Tensor | None = None) -> torch.Tensor:
        """z_pre: (N, D); force: (N,) in [0, 1]; heading: (N,) robot heading in radians
        (required when use_heading). Returns predicted z_rest (N, D)."""
        parts = [z_pre]
        if self.use_heading:
            if heading is None:
                raise ValueError("this ShotModel needs the robot heading")
            heading = heading.reshape(-1, 1).to(z_pre)
            parts += [torch.sin(heading), torch.cos(heading)]
        parts.append(force.reshape(-1, 1).to(z_pre))
        return z_pre + self.net(torch.cat(parts, dim=-1))


def save_shot_model(model: ShotModel, path: str, meta: dict):
    torch.save(model.state_dict(), path)
    with open(path.replace(".pt", ".json"), "w") as fh:
        json.dump(meta, fh, indent=1)


def load_shot_model(path: str, device="cpu") -> tuple[ShotModel, dict]:
    """Returns (model, metadata). Metadata names the frozen encoder file the latents came from."""
    state = torch.load(path, map_location=device)
    use_heading = "heading_flag" in state
    hidden = state["net.0.weight"].shape[0]
    latent = state["net.0.weight"].shape[1] - (3 if use_heading else 1)
    # Linear weights are the 2-D net.<i>.weight tensors; all but the output layer are hidden blocks.
    n_linear = sum(1 for k, v in state.items() if k.startswith("net.") and k.endswith(".weight") and v.dim() == 2)
    model = ShotModel(latent, hidden, use_heading=use_heading, depth=n_linear - 1)
    model.load_state_dict(state)
    meta_path = path.replace(".pt", ".json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return model.to(device).eval(), meta

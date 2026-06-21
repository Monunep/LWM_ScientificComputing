from types import SimpleNamespace
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 16 * 16, hidden_size),
        )

    def forward(self, pixels, interpolate_pos_encoding=True):
        cls = self.net(pixels)
        return SimpleNamespace(last_hidden_state=cls.unsqueeze(1))


def main():
    torch.manual_seed(0)

    batch = 2
    samples = 3
    history = 3
    horizon = 5
    action_dim = 2
    hidden_dim = 32
    embed_dim = 16

    model = JEPA(
        encoder=TinyEncoder(hidden_dim),
        predictor=ARPredictor(
            num_frames=history,
            depth=2,
            heads=2,
            mlp_dim=64,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dim_head=16,
        ),
        action_encoder=Embedder(input_dim=action_dim, emb_dim=embed_dim),
        projector=MLP(hidden_dim, hidden_dim, embed_dim),
        pred_proj=MLP(hidden_dim, hidden_dim, embed_dim),
    )

    info = {
        "pixels": torch.randn(batch, history + 1, 3, 16, 16),
        "action": torch.randn(batch, history + 1, action_dim),
    }
    encoded = model.encode(info)
    assert encoded["emb"].shape == (batch, history + 1, embed_dim)
    assert encoded["act_emb"].shape == (batch, history + 1, embed_dim)

    pred = model.predict(encoded["emb"][:, :history], encoded["act_emb"][:, :history])
    assert pred.shape == (batch, history, embed_dim)

    sigreg_loss = SIGReg(knots=5, num_proj=8)(encoded["emb"].transpose(0, 1))
    assert torch.isfinite(sigreg_loss)

    rollout_info = {
        "pixels": torch.randn(batch, samples, history, 3, 16, 16),
        "goal": torch.randn(batch, samples, 1, 3, 16, 16),
        "action": torch.randn(batch, samples, history, action_dim),
    }
    actions = torch.randn(batch, samples, horizon, action_dim)
    out = model.rollout(rollout_info, actions)
    assert out["predicted_emb"].shape == (batch, samples, horizon + 1, embed_dim)

    cost_info = {
        "pixels": torch.randn(batch, samples, history, 3, 16, 16),
        "goal": torch.randn(batch, samples, 1, 3, 16, 16),
        "action": torch.randn(batch, samples, history, action_dim),
    }
    cost = model.get_cost(cost_info, actions)
    assert cost.shape == (batch, samples)
    assert torch.isfinite(cost).all()

    print("smoke test passed")


if __name__ == "__main__":
    main()

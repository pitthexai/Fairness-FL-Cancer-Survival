# MLP/mlp_models.py

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


# =========================
# MLP Model
# =========================
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[str | List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Allow "64,32" or [64, 32]
        if hidden_dims is None:
            hidden = [64, 32]
        elif isinstance(hidden_dims, str):
            hidden = [int(h.strip()) for h in hidden_dims.split(",") if h.strip()]
        else:
            hidden = list(hidden_dims)

        layers = []
        in_dim = input_dim

        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h

        # Final layer (binary classification -> single logit)
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

        # Optional: initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape: (N,) for BCEWithLogitsLoss
        logits = self.net(x)
        return logits.squeeze(1)


# =========================
# Flower Helpers
# =========================
def get_weights(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as NumPy arrays."""
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    """Load NumPy weights into model."""
    state_dict = model.state_dict()
    new_state_dict = {}

    for (key, _), w in zip(state_dict.items(), weights):
        new_state_dict[key] = torch.tensor(w)

    model.load_state_dict(new_state_dict, strict=True)
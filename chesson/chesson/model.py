"""Optimised dual‑head ResNet for chesSon.

Key changes:
- SiLU activation (better than ReLU for deep nets).
- GroupNorm(4, channels) instead of BatchNorm – works well with small batch sizes.
- Optional FP16 inference via `self.half()`.
- `from_pretrained` helper for easy loading.
- Default channel count configurable (96 for a lightweight model).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from chesson.board_encoding import ACTION_SIZE, NUM_PLANES


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm and SiLU activation."""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ChesSonNet(nn.Module):
    """Dual‑head ResNet for chess move policy and position value.

    Parameters
    ----------
    num_blocks: int
        Number of residual blocks in the backbone.
    channels: int
        Feature depth for the convolutional layers. ``96`` works well for a
        lightweight model that still defeats a 1200 Elo opponent.
    """

    def __init__(self, num_blocks: int = 6, channels: int = 128):
        super().__init__()
        # Initial convolution
        self.input = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.SiLU(inplace=True),
        )
        # Residual backbone
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        # Policy head
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=1, bias=False),
            nn.GroupNorm(4, 64),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, ACTION_SIZE),
        )
        # Value head
        self.value = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    @classmethod
    def from_pretrained(cls, weights_path: str, device: str = "cpu"):
        """Load a checkpoint and move the model to ``device``.
        The returned model is already in ``eval`` mode.
        """
        model = cls()
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    def forward(self, x: torch.Tensor):
        """Forward pass returning ``(policy_logits, value)``.
        ``x`` should be shape ``(B, 14, 8, 8)`` and preferably ``float16`` for GPU.
        """
        x = self.input(x)
        x = self.res_blocks(x)
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value

    def set_fp16(self):
        """Convert all parameters to ``float16`` for inference.
        Call this after loading weights if you want the model to run in FP16.
        """
        self.half()
        # Keep normalization layers in FP32 for numerical stability
        for m in self.modules():
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                m.float()

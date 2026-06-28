"""Word positional encoding used by the BTTR Transformer decoder."""
import torch
import torch.nn as nn


class WordPosEnc(nn.Module):
    """Sinusoidal positional encoding matching the original BTTR checkpoint."""

    def __init__(self, d_model: int = 512, max_len: int = 500, temperature: float = 10000.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float)
        dim_t = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = 1.0 / (temperature ** (dim_t / d_model))
        inv_freq = torch.einsum("i, j -> i j", position, div_term)
        pe[:, 0::2] = inv_freq.sin()
        pe[:, 1::2] = inv_freq.cos()
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.size()
        return x + self.pe[:seq_len, :][None, :, :]

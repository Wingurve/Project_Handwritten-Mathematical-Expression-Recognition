"""
DenseNet-based Image Encoder — exact match of original BTTR (Green-Wood/BTTR).

Architecture (from bttr/model/encoder.py):
  DenseNet (DenseNet-B with bottleneck)
    conv1(1→2*growth, 7×7s2) → norm1(BN) → ReLU → MaxPool(2×2)
    → DenseBlock(dense1) → Transition(trans1)
    → DenseBlock(dense2) → Transition(trans2)
    → DenseBlock(dense3) → post_norm(BN)        [NO ReLU after post_norm]

  Encoder
    DenseNet → feature_proj(Conv2d→ReLU) → LayerNorm → ImgPosEnc → flatten
"""
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ========================
# Image Position Encoding (from original BTTR pos_enc.py)
# ========================
class ImgPosEnc(nn.Module):
    """2D sinusoidal position encoding for images."""

    def __init__(
        self,
        d_model: int = 512,
        temperature: float = 10000.0,
        normalize: bool = False,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert d_model % 2 == 0
        self.half_d_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Add 2D image positional encoding.

        Parameters
        ----------
        x : torch.Tensor [b, h, w, d]
        mask: torch.Tensor [b, h, w] - bool, True=padded

        Returns
        -------
        torch.Tensor [b, h, w, d]
        """
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.half_d_model, dtype=torch.float, device=x.device)
        inv_feq = 1.0 / (self.temperature ** (dim_t / self.half_d_model))

        pos_x = torch.einsum("b h w, d -> b h w d", x_embed, inv_feq)
        pos_y = torch.einsum("b h w, d -> b h w d", y_embed, inv_feq)

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_x, pos_y), dim=3)

        return x + pos


# ========================
# DenseNet-B Components (from original BTTR)
# ========================
class _Bottleneck(nn.Module):
    """DenseNet-B bottleneck: conv1→bn1→relu→conv2→bn2→relu→concat"""

    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super().__init__()
        interChannels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(n_channels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(interChannels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        return torch.cat((x, out), 1)


class _Transition(nn.Module):
    """Transition: conv1→bn1→relu→avgpool"""

    def __init__(self, n_channels: int, n_out_channels: int, use_dropout: bool):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(n_out_channels)
        self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class DenseNet(nn.Module):
    """DenseNet-B backbone matching original BTTR DenseNet class."""

    def __init__(
        self,
        growth_rate: int = 24,
        num_layers: int = 16,
        reduction: float = 0.5,
        bottleneck: bool = True,
        use_dropout: bool = True,
    ):
        super().__init__()
        n_dense_blocks = num_layers
        n_channels = 2 * growth_rate  # 48

        self.conv1 = nn.Conv2d(1, n_channels, kernel_size=7, padding=3, stride=2, bias=False)
        self.norm1 = nn.BatchNorm2d(n_channels)

        self.dense1 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout)
        n_channels += n_dense_blocks * growth_rate  # 432
        n_out_channels = int(math.floor(n_channels * reduction))  # 216
        self.trans1 = _Transition(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense2 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout)
        n_channels += n_dense_blocks * growth_rate  # 600
        n_out_channels = int(math.floor(n_channels * reduction))  # 300
        self.trans2 = _Transition(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense3 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout)
        n_channels += n_dense_blocks * growth_rate  # 684

        self.out_channels = n_channels
        self.post_norm = nn.BatchNorm2d(self.out_channels)

    @staticmethod
    def _make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout):
        layers = []
        for _ in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(_Bottleneck(n_channels, growth_rate, use_dropout))
            else:
                layers.append(_SingleLayer(n_channels, growth_rate, use_dropout))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, x_mask):
        """Forward with mask tracking through pooling.

        Parameters
        ----------
        x : [b, 1, h, w]
        x_mask : [b, h, w] - bool, True=padded
        """
        out = self.conv1(x)              # stride 2
        out = self.norm1(out)
        out_mask = x_mask[:, 0::2, 0::2]  # downsample mask

        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out_mask = out_mask[:, 0::2, 0::2]

        out = self.dense1(out)
        out = self.trans1(out)
        out_mask = out_mask[:, 0::2, 0::2]  # avgpool 2×2 in transition

        out = self.dense2(out)
        out = self.trans2(out)
        out_mask = out_mask[:, 0::2, 0::2]

        out = self.dense3(out)
        out = self.post_norm(out)            # NO ReLU after post_norm
        return out, out_mask


# ========================
# Full Encoder (matching original BTTR Encoder)
# ========================
class DenseNetEncoder(nn.Module):
    """
    Full BTTR Encoder:
      DenseNet → feature_proj(Conv2d→ReLU) → LayerNorm → ImgPosEnc → flatten

    Checkpoint keys:
      bttr.encoder.model.*        → DenseNet
      bttr.encoder.feature_proj.* → Conv2d+ReLU
      bttr.encoder.norm.*         → LayerNorm(d_model)
    """

    def __init__(
        self,
        d_model: int = 256,
        growth_rate: int = 24,
        num_layers: int = 16,
    ):
        super().__init__()
        self.d_model = d_model

        self.model = DenseNet(growth_rate=growth_rate, num_layers=num_layers)

        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.model.out_channels, d_model, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.LayerNorm(d_model)

        # 2D image position encoding
        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)

    def forward(self, img, img_mask):
        """
        Parameters
        ----------
        img : torch.Tensor [b, 1, h', w']
        img_mask : torch.Tensor [b, h', w'] - bool, True=padded

        Returns
        -------
        feature : torch.Tensor [b, h*w, d_model]
        mask : torch.Tensor [b, h*w] - bool, True=padded
        """
        # DenseNet backbone
        feature, mask = self.model(img, img_mask)  # [b, C, h, w], [b, h, w]

        # Project channels: C → d_model
        feature = self.feature_proj(feature)  # [b, d_model, h, w]

        # Rearrange for LayerNorm and position encoding
        feature = rearrange(feature, "b d h w -> b h w d")

        # LayerNorm on channel dim
        feature = self.norm(feature)

        # 2D position encoding
        feature = self.pos_enc_2d(feature, mask)

        # Flatten spatial dims
        feature = rearrange(feature, "b h w d -> b (h w) d")
        mask = rearrange(mask, "b h w -> b (h w)")

        return feature, mask

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hermite_basis import HermiteBasisConfig
from .srf_layers import ConvBNReLU, SRFBNReLU


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, use_batchnorm=use_batchnorm, dropout=dropout),
            ConvBNReLU(out_channels, out_channels, use_batchnorm=use_batchnorm, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DoubleSRFConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        basis_path: str | Path,
        basis_config: HermiteBasisConfig,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            SRFBNReLU(in_channels, out_channels, basis_path, basis_config, use_batchnorm, dropout),
            SRFBNReLU(out_channels, out_channels, basis_path, basis_config, use_batchnorm, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, block_factory, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            up_channels = in_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            up_channels = in_channels // 2
        self.conv = block_factory(up_channels + skip_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class HermiteSRFUNet(nn.Module):
    """
    U-Net para segmentación con encoder basado en Structured Receptive Fields.

    Si encoder_block='srf', los bloques del encoder indicados en srf_stages usan
    filtros base Hermite/Gaussian derivatives + mezcla 1x1 aprendible.
    El decoder queda convolucional por defecto para mantener estabilidad.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        basis_path: str | Path = "assets/hermite_basis/hermite_order3_k7_scales_1.0_2.0.pt",
        basis_config: Optional[HermiteBasisConfig] = None,
        encoder_block: str = "srf",
        decoder_block: str = "conv",
        srf_stages: Optional[Iterable[int]] = None,
        bilinear: bool = False,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth debe ser >= 2")
        self.depth = int(depth)
        self.encoder_block = encoder_block
        self.decoder_block = decoder_block
        self.srf_stages = set(range(depth)) if srf_stages is None else set(int(s) for s in srf_stages)
        basis_config = basis_config or HermiteBasisConfig()

        channels = [base_channels * (2**i) for i in range(depth)]

        def make_block(in_ch: int, out_ch: int, block_type: str, stage: int | None = None) -> nn.Module:
            use_srf = block_type == "srf" and (stage is None or stage in self.srf_stages)
            if use_srf:
                return DoubleSRFConv(in_ch, out_ch, basis_path, basis_config, use_batchnorm, dropout)
            return DoubleConv(in_ch, out_ch, use_batchnorm, dropout)

        self.inc = make_block(in_channels, channels[0], encoder_block, stage=0)
        self.downs = nn.ModuleList()
        for i in range(1, depth):
            self.downs.append(Down(make_block(channels[i - 1], channels[i], encoder_block, stage=i)))

        self.ups = nn.ModuleList()
        rev_channels = channels[::-1]
        for i in range(depth - 1):
            in_ch = rev_channels[i]
            skip_ch = rev_channels[i + 1]
            out_ch = rev_channels[i + 1]

            def block_factory(a, b, block_type=decoder_block):
                # En decoder se recomienda conv. Si se elige srf, usa el mismo banco.
                if block_type == "srf":
                    return DoubleSRFConv(a, b, basis_path, basis_config, use_batchnorm, dropout)
                return DoubleConv(a, b, use_batchnorm, dropout)

            self.ups.append(Up(in_ch, skip_ch, out_ch, block_factory, bilinear=bilinear))

        self.outc = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        x = skips[0]
        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = skips[-1]
        for i, up in enumerate(self.ups):
            skip = skips[-2 - i]
            x = up(x, skip)
        return self.outc(x)


def build_model_from_config(cfg: dict) -> HermiteSRFUNet:
    mode = cfg["data"].get("segmentation_mode", "binary")
    out_channels = 1 if mode == "binary" else int(cfg["data"].get("num_classes", 2))
    bcfg = cfg.get("basis", {})
    basis_config = HermiteBasisConfig(
        kernel_size=int(bcfg.get("kernel_size", 7)),
        max_order=int(bcfg.get("max_order", 3)),
        scales=tuple(float(s) for s in bcfg.get("scales", [1.0, 2.0])),
        include_orders=bcfg.get("include_orders", None),
        normalize=bool(bcfg.get("normalize", True)),
        zero_mean_except_order0=bool(bcfg.get("zero_mean_except_order0", True)),
    )
    mcfg = cfg.get("model", {})
    return HermiteSRFUNet(
        in_channels=int(mcfg.get("in_channels", 3)),
        num_classes=out_channels,
        base_channels=int(mcfg.get("base_channels", 32)),
        depth=int(mcfg.get("depth", 4)),
        basis_path=bcfg.get("path", "assets/hermite_basis/hermite_basis.pt"),
        basis_config=basis_config,
        encoder_block=mcfg.get("encoder_block", "srf"),
        decoder_block=mcfg.get("decoder_block", "conv"),
        srf_stages=mcfg.get("srf_stages", None),
        bilinear=bool(mcfg.get("bilinear", False)),
        use_batchnorm=bool(mcfg.get("use_batchnorm", True)),
        dropout=float(mcfg.get("dropout", 0.0)),
    )

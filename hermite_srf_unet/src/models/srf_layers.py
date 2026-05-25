from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hermite_basis import HermiteBasisConfig, load_basis, save_basis


class HermiteBasisConv2d(nn.Module):
    """
    Capa tipo Structured Receptive Field de Jacobsen et al.

    Implementación fiel a la idea del paper:
    1) convolución fija con un banco de derivadas gaussianas/Hermite;
    2) recombinación aprendible mediante una convolución 1x1.

    Por linealidad, equivale a aprender kernels efectivos como combinación lineal
    de la base, sin construir explícitamente dichos kernels en cada forward.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        basis_path: str | Path,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = True,
        basis_config: Optional[HermiteBasisConfig] = None,
        overwrite_basis: bool = False,
    ) -> None:
        super().__init__()
        basis_path = Path(basis_path)
        if not basis_path.exists():
            if basis_config is None:
                basis_config = HermiteBasisConfig()
            save_basis(basis_path, basis_config, overwrite=overwrite_basis)

        basis, meta = load_basis(basis_path)
        if basis.ndim != 4 or basis.shape[1] != 1:
            raise ValueError("La base debe tener forma [num_basis, 1, K, K]")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_basis = int(basis.shape[0])
        self.kernel_size = int(basis.shape[-1])
        self.stride = int(stride)
        self.padding = self.kernel_size // 2 if padding is None else int(padding)
        self.meta = meta

        self.register_buffer("basis", basis.float(), persistent=True)
        self.register_buffer("basis_weight", basis.float().repeat(self.in_channels, 1, 1, 1), persistent=False)
        self.mix = nn.Conv2d(
            in_channels * self.num_basis,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if c != self.in_channels:
            raise ValueError(f"Se esperaban {self.in_channels} canales, llegaron {c}")

        # Mismo banco para cada canal de entrada: groups=in_channels.
        # weight: [C_in*num_basis, 1, K, K]
        feats = F.conv2d(
            x,
            self.basis_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=c,
        )
        return self.mix(feats)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"num_basis={self.num_basis}, kernel_size={self.kernel_size}, stride={self.stride}"
        )


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SRFBNReLU(nn.Module):
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
        layers: list[nn.Module] = [
            HermiteBasisConv2d(
                in_channels,
                out_channels,
                basis_path=basis_path,
                bias=not use_batchnorm,
                basis_config=basis_config,
            )
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

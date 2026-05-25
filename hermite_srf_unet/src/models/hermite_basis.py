from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch


@dataclass
class HermiteBasisConfig:
    kernel_size: int = 7
    max_order: int = 3
    scales: tuple[float, ...] = (1.0, 2.0)
    include_orders: Optional[list[tuple[int, int]]] = None
    normalize: bool = True
    zero_mean_except_order0: bool = True


def physicists_hermite(n: int, x: np.ndarray) -> np.ndarray:
    """Calcula H_n(x) mediante recurrencia de Hermite físico."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h0, h1 = h1, 2.0 * x * h1 - 2.0 * (k - 1) * h0
    return h1


def gaussian_2d(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(x**2 + y**2) / (2.0 * sigma**2)) / (2.0 * np.pi * sigma**2)


def gaussian_derivative_2d(a: int, b: int, sigma: float, kernel_size: int) -> np.ndarray:
    """
    Filtro de derivada gaussiana 2D asociado a polinomios de Hermite.

    Sigue la idea usada en Structured Receptive Fields: los kernels se expresan
    como combinaciones de derivadas gaussianas. Las constantes exactas no son
    críticas porque la base se normaliza y la red aprende los coeficientes de mezcla.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size debe ser impar")

    half = kernel_size // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    x, y = np.meshgrid(coords, coords)

    # Relación entre derivadas de gaussianas y polinomios de Hermite.
    hx = physicists_hermite(a, x / (sigma * np.sqrt(2.0)))
    hy = physicists_hermite(b, y / (sigma * np.sqrt(2.0)))
    g = gaussian_2d(x, y, sigma)

    filt = ((-1.0) ** (a + b)) * hx * hy * g / (sigma ** (a + b + 1e-12))
    return filt.astype(np.float32)


def order_pairs(max_order: int, include_orders: Optional[Iterable[tuple[int, int]]] = None) -> list[tuple[int, int]]:
    if include_orders is not None:
        return [(int(a), int(b)) for a, b in include_orders]
    pairs = []
    for n in range(max_order + 1):
        for a in range(n + 1):
            b = n - a
            pairs.append((a, b))
    return pairs


def make_hermite_basis(config: HermiteBasisConfig) -> tuple[torch.Tensor, list[dict]]:
    filters: list[np.ndarray] = []
    meta: list[dict] = []
    pairs = order_pairs(config.max_order, config.include_orders)

    for sigma in config.scales:
        for a, b in pairs:
            f = gaussian_derivative_2d(a, b, float(sigma), config.kernel_size)

            if config.normalize:
                if not (config.zero_mean_except_order0 and a == 0 and b == 0):
                    f = f - f.mean()
                norm = np.sqrt(np.sum(f**2))
                if norm > 0:
                    f = f / norm

            filters.append(f)
            meta.append({"a": a, "b": b, "order": a + b, "sigma": float(sigma)})

    basis = torch.from_numpy(np.stack(filters, axis=0)).float().unsqueeze(1)  # [B,1,K,K]
    return basis, meta


def save_basis(path: str | Path, config: HermiteBasisConfig, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    basis, meta = make_hermite_basis(config)
    payload = {
        "basis": basis,
        "meta": meta,
        "config": asdict(config),
    }
    torch.save(payload, path)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "config": asdict(config)}, f, indent=2)
    return path


def load_basis(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[torch.Tensor, list[dict]]:
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and "basis" in payload:
        return payload["basis"].float(), payload.get("meta", [])
    if isinstance(payload, torch.Tensor):
        return payload.float(), []
    raise ValueError(f"Formato de basis no soportado: {path}")

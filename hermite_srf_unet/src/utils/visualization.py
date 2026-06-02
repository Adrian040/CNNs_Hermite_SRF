from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
import torch


def denormalize(img: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    """
    Convierte una imagen normalizada tipo tensor C,H,W a numpy H,W,C en rango [0, 1].
    """

    img = img.detach().cpu().clone()

    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]

    img = img.clamp(0, 1).permute(1, 2, 0).numpy()

    return img


def mask_to_class_ids(mask: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Convierte máscaras codificadas como intensidades a IDs de clase.

    Ejemplos:
    - [0, 1, 2, 3, 4] se queda igual.
    - [0, 64, 128, 192, 255] se convierte a [0, 1, 2, 3, 4].
    - [0, 85, 170, 255] se convierte a [0, 1, 2, 3].
    """

    mask = np.asarray(mask)

    if mask.ndim == 3:
        # Si viene como RGB, nos quedamos con un canal.
        # Esto solo funciona si la máscara RGB realmente codifica intensidades iguales por canal.
        mask = mask[..., 0]

    unique_vals = np.unique(mask)

    # Si ya está en formato de clases, no se modifica
    if unique_vals.min() >= 0 and unique_vals.max() < num_classes:
        return mask.astype(np.uint8)

    # Si hay pocos valores únicos, hacemos mapeo directo
    if len(unique_vals) <= num_classes:
        remap = {val: idx for idx, val in enumerate(sorted(unique_vals))}
        out = np.zeros_like(mask, dtype=np.uint8)

        for val, idx in remap.items():
            out[mask == val] = idx

        return out

    # Si viene en escala 0-255 o similar, cuantizamos
    mask_float = mask.astype(np.float32)
    mask_float = mask_float - mask_float.min()

    if mask_float.max() > 0:
        mask_float = mask_float / mask_float.max()

    out = np.rint(mask_float * (num_classes - 1)).astype(np.uint8)
    out = np.clip(out, 0, num_classes - 1)

    return out


def get_class_colors(num_classes: int = 2) -> np.ndarray:
    """
    Define colores fijos para cada clase.
    La clase 0 se deja negra para fondo.
    """

    if num_classes <= 2:
        colors = np.array(
            [
                [0, 0, 0],        # clase 0: fondo
                [255, 0, 0],      # clase 1
            ],
            dtype=np.uint8,
        )
    elif num_classes == 5:
        colors = np.array(
            [
                [0, 0, 0],        # clase 0: fondo
                [255, 0, 0],      # clase 1: rojo
                [0, 255, 0],      # clase 2: verde
                [0, 0, 255],      # clase 3: azul
                [255, 255, 0],    # clase 4: amarillo
            ],
            dtype=np.uint8,
        )
    else:
        rng = np.random.default_rng(123)
        colors = np.zeros((num_classes, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]
        colors[1:] = rng.integers(
            low=40,
            high=255,
            size=(num_classes - 1, 3),
            dtype=np.uint8,
        )

    return colors


def get_discrete_cmap(num_classes: int = 2):
    """
    Devuelve cmap y norm discretos para matplotlib.
    Útil para usar con imshow(mask, cmap=cmap, norm=norm).
    """

    colors = get_class_colors(num_classes).astype(np.float32) / 255.0

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(
        boundaries=np.arange(-0.5, num_classes + 0.5, 1),
        ncolors=num_classes,
    )

    return cmap, norm


def colorize_mask(mask: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Convierte una máscara de clases a imagen RGB coloreada.

    Entrada esperada:
    - máscara con clases 0, 1, 2, ...
    - o máscara con intensidades 0, 64, 128, ...
    """

    mask = mask_to_class_ids(mask, num_classes=num_classes)
    colors = get_class_colors(num_classes)

    mask = np.clip(mask, 0, num_classes - 1)

    return colors[mask]


def save_mask_png(
    mask: np.ndarray,
    path: str | Path,
    num_classes: int = 2,
    colorized: bool = False,
) -> None:
    """
    Guarda una máscara como PNG.

    Si colorized=False:
        Guarda la máscara como IDs de clase 0, 1, 2, 3...
        Esto es mejor para evaluación posterior.

    Si colorized=True:
        Guarda una imagen RGB coloreada.
        Esto es mejor para inspección visual.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mask = mask_to_class_ids(mask, num_classes=num_classes)

    if colorized:
        arr = colorize_mask(mask, num_classes=num_classes)
        Image.fromarray(arr).save(path)
    else:
        Image.fromarray(mask.astype(np.uint8)).save(path)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.35,
    num_classes: int | None = None,
) -> np.ndarray:
    """
    Genera overlay de una máscara sobre una imagen.

    A diferencia de la versión original, esta función asigna un color distinto
    a cada clase. La clase 0 se interpreta como fondo y no se pinta.
    """

    img = image.copy()

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    if img.max() > 1:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)

    if num_classes is None:
        num_classes = int(np.max(mask)) + 1
        num_classes = max(num_classes, 2)

    mask = mask_to_class_ids(mask, num_classes=num_classes)

    color_mask = colorize_mask(mask, num_classes=num_classes).astype(np.float32) / 255.0

    out = img.copy()

    foreground = mask > 0
    out[foreground] = (
        (1 - alpha) * out[foreground]
        + alpha * color_mask[foreground]
    )

    return np.clip(out, 0, 1)


def save_training_curves(history_csv: str | Path, output_path: str | Path) -> None:
    import pandas as pd

    df = pd.read_csv(history_csv)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.set_xlabel("Época")
    ax1.set_ylabel("Dice")
    ax1.plot(df["epoch"], df["val_dice"], label="Val Dice")

    if "train_dice" in df.columns:
        ax1.plot(df["epoch"], df["train_dice"], linestyle="--", label="Train Dice")

    ax1.tick_params(axis="y")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss")
    ax2.plot(df["epoch"], df["train_loss"], label="Train Loss")
    ax2.plot(df["epoch"], df["val_loss"], linestyle="--", label="Val Loss")
    ax2.tick_params(axis="y")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
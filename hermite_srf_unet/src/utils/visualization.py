from __future__ import annotations

from pathlib import Path
import csv
import contextlib
import io

import numpy as np
from PIL import Image
import torch


def denormalize(img: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    img = img.detach().cpu().clone()
    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def _default_colors(num_classes: int) -> list[tuple[int, int, int]]:
    palette = [
        (0, 0, 0),
        (230, 25, 75),
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
    ]
    colors = [palette[i % len(palette)] for i in range(num_classes)]
    colors[0] = (0, 0, 0)
    return colors


def normalize_colors(num_classes: int, colors: list[tuple[int, int, int]] | None = None) -> np.ndarray:
    raw_colors = colors or _default_colors(num_classes)
    if len(raw_colors) < num_classes:
        raw_colors = raw_colors + _default_colors(num_classes)[len(raw_colors) :]
    arr = np.array(raw_colors[:num_classes], dtype=np.uint8)
    arr[0] = np.array([0, 0, 0], dtype=np.uint8)
    return arr


def mask_cmap(num_classes: int, colors: list[tuple[int, int, int]] | None = None):
    from matplotlib.colors import ListedColormap

    return ListedColormap(normalize_colors(num_classes, colors).astype(np.float32) / 255.0)


def colorize_mask(mask: np.ndarray, num_classes: int = 2, colors: list[tuple[int, int, int]] | None = None) -> np.ndarray:
    mask = mask.astype(np.int64)
    palette = normalize_colors(num_classes, colors)
    return palette[np.clip(mask, 0, num_classes - 1)]


def save_mask_png(
    mask: np.ndarray,
    path: str | Path,
    num_classes: int = 2,
    colors: list[tuple[int, int, int]] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colorize_mask(mask, num_classes=num_classes, colors=colors)).save(path)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.35,
    num_classes: int = 2,
    colors: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    img = image.copy()
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.max() > 1:
        img = img / 255.0
    color = colorize_mask(mask, num_classes=num_classes, colors=colors).astype(np.float32) / 255.0
    m = mask > 0
    out = img.copy()
    out[m] = (1 - alpha) * out[m] + alpha * color[m]
    return np.clip(out, 0, 1)


def class_legend_handles(
    class_names: list[str],
    colors: list[tuple[int, int, int]] | None = None,
    present_classes: list[int] | None = None,
):
    from matplotlib.patches import Patch

    palette = normalize_colors(len(class_names), colors).astype(np.float32) / 255.0
    class_ids = present_classes if present_classes is not None else list(range(len(class_names)))
    return [
        Patch(facecolor=palette[class_id], edgecolor="black", label=f"{class_id}: {class_names[class_id]}")
        for class_id in class_ids
        if 0 <= class_id < len(class_names)
    ]


def require_pyplot():
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "No se pudo importar Matplotlib. En Colab reinicia el runtime despues "
            "de instalar requirements.txt y vuelve a ejecutar el notebook."
        ) from exc

    return plt


def save_training_curves(history_csv: str | Path, output_path: str | Path) -> None:
    plt = require_pyplot()

    with open(history_csv, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    def col(name: str) -> list[float]:
        return [float(row[name]) for row in rows]

    epochs = col("epoch")
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Dice")
    ax1.plot(epochs, col("val_dice"), color="tab:blue", label="Val Dice")
    if "train_dice" in rows[0]:
        ax1.plot(epochs, col("train_dice"), color="tab:cyan", linestyle="--", label="Train Dice")
    ax1.tick_params(axis="y")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss")
    ax2.plot(epochs, col("train_loss"), color="tab:red", label="Train Loss")
    ax2.plot(epochs, col("val_loss"), color="tab:orange", linestyle="--", label="Val Loss")
    ax2.tick_params(axis="y")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

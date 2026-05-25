from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch


def denormalize(img: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    img = img.detach().cpu().clone()
    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def colorize_mask(mask: np.ndarray, num_classes: int = 2) -> np.ndarray:
    mask = mask.astype(np.int64)
    if num_classes <= 2:
        return (mask > 0).astype(np.uint8) * 255
    rng = np.random.default_rng(123)
    colors = np.zeros((num_classes, 3), dtype=np.uint8)
    colors[1:] = rng.integers(40, 255, size=(num_classes - 1, 3), dtype=np.uint8)
    return colors[np.clip(mask, 0, num_classes - 1)]


def save_mask_png(mask: np.ndarray, path: str | Path, num_classes: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if num_classes <= 2:
        arr = (mask > 0).astype(np.uint8) * 255
        Image.fromarray(arr).save(path)
    else:
        Image.fromarray(mask.astype(np.uint8)).save(path)


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    img = image.copy()
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.max() > 1:
        img = img / 255.0
    color = np.zeros_like(img)
    color[..., 0] = 1.0
    m = mask > 0
    out = img.copy()
    out[m] = (1 - alpha) * out[m] + alpha * color[m]
    return np.clip(out, 0, 1)


def save_training_curves(history_csv: str | Path, output_path: str | Path) -> None:
    import pandas as pd

    df = pd.read_csv(history_csv)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Dice")
    ax1.plot(df["epoch"], df["val_dice"], color="tab:blue", label="Val Dice")
    if "train_dice" in df.columns:
        ax1.plot(df["epoch"], df["train_dice"], color="tab:cyan", linestyle="--", label="Train Dice")
    ax1.tick_params(axis="y")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss")
    ax2.plot(df["epoch"], df["train_loss"], color="tab:red", label="Train Loss")
    ax2.plot(df["epoch"], df["val_loss"], color="tab:orange", linestyle="--", label="Val Loss")
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

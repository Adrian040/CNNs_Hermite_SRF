from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch


CLASS_COLOR_TO_ID = {
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (0, 255, 0): 2,
    (0, 0, 255): 3,
}


# ============================================================
# IMAGEN
# ============================================================

def denormalize(
    img: torch.Tensor,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """
    Convierte una imagen normalizada C,H,W a numpy H,W,C en rango [0,1].
    """

    img = img.detach().cpu().clone()

    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]

    img = img.clamp(0, 1).permute(1, 2, 0).numpy()

    return img


# ============================================================
# COLORES DE CLASES
# ============================================================

def get_class_colors(num_classes: int = 4) -> np.ndarray:
    """
    Colores fijos para tus 4 clases:

        clase 0 -> negro / fondo
        clase 1 -> rojo
        clase 2 -> verde
        clase 3 -> azul
    """

    base_colors = np.array(
        [
            [0, 0, 0],        # clase 0: fondo
            [255, 0, 0],      # clase 1: rojo
            [0, 255, 0],      # clase 2: verde
            [0, 0, 255],      # clase 3: azul
        ],
        dtype=np.uint8,
    )

    if num_classes <= 4:
        return base_colors[:num_classes]

    # Por seguridad, si algún día usas más de 4 clases
    rng = np.random.default_rng(123)
    extra = rng.integers(
        low=40,
        high=255,
        size=(num_classes - 4, 3),
        dtype=np.uint8,
    )

    return np.vstack([base_colors, extra])


def get_discrete_cmap(num_classes: int = 4):
    """
    Crea un colormap discreto para matplotlib.

    Uso:
        cmap, norm = get_discrete_cmap(4)
        plt.imshow(mask, cmap=cmap, norm=norm)
    """

    from matplotlib.colors import BoundaryNorm, ListedColormap

    colors = get_class_colors(num_classes).astype(np.float32) / 255.0

    cmap = ListedColormap(colors)

    norm = BoundaryNorm(
        boundaries=np.arange(-0.5, num_classes + 0.5, 1),
        ncolors=num_classes,
    )

    return cmap, norm


# ============================================================
# MÁSCARAS
# ============================================================

def mask_to_class_ids(mask: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """
    Asegura que una máscara esté como IDs de clase 0,1,2,3.

    Casos soportados:
    - máscara H,W ya como 0,1,2,3
    - máscara RGB con colores negro, rojo, verde, azul
    """

    mask = np.asarray(mask)

    # Caso RGB
    if mask.ndim == 3 and mask.shape[-1] == 3:
        h, w, _ = mask.shape
        out = np.zeros((h, w), dtype=np.uint8)

        known = np.zeros((h, w), dtype=bool)
        for color, class_id in CLASS_COLOR_TO_ID.items():
            color_arr = np.array(color, dtype=np.uint8)
            matches = np.all(mask == color_arr, axis=-1)
            out[matches] = class_id
            known |= matches

        if not np.all(known):
            unknown = np.unique(mask[~known].reshape(-1, 3), axis=0)
            unknown_colors = [tuple(int(x) for x in c) for c in unknown[:12]]
            suffix = "..." if len(unknown) > 12 else ""
            raise ValueError(
                "La máscara RGB contiene colores fuera de la paleta esperada: "
                f"{unknown_colors}{suffix}"
            )

        return np.clip(out, 0, num_classes - 1).astype(np.uint8)

    # Caso H,W
    mask = mask.astype(np.int64)

    invalid = (mask < 0) | (mask >= num_classes)
    if np.any(invalid):
        values = np.unique(mask[invalid])
        raise ValueError(
            f"La máscara contiene IDs fuera de 0..{num_classes - 1}: "
            f"{values[:12].tolist()}"
        )

    return mask.astype(np.uint8)


def colorize_mask(mask: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """
    Convierte una máscara de clases H,W en una imagen RGB coloreada.
    """

    mask = mask_to_class_ids(mask, num_classes=num_classes)

    colors = get_class_colors(num_classes)
    color_mask = colors[mask]

    return color_mask.astype(np.uint8)


def save_mask_png(
    mask: np.ndarray,
    path: str | Path,
    num_classes: int = 4,
    colorized: bool = False,
) -> None:
    """
    Guarda una máscara como PNG.

    colorized=False:
        Guarda la máscara como IDs 0,1,2,3.
        Esto es útil para evaluación.

    colorized=True:
        Guarda la máscara coloreada como RGB.
        Esto es útil para visualización.
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
    num_classes: int = 4,
) -> np.ndarray:
    """
    Superpone una máscara multiclase sobre la imagen.

    La clase 0 es fondo y no se pinta.
    Las clases 1,2,3 se pintan con rojo, verde y azul.
    """

    img = image.copy()

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    if img.max() > 1:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)

    mask = mask_to_class_ids(mask, num_classes=num_classes)

    color_mask = colorize_mask(mask, num_classes=num_classes).astype(np.float32) / 255.0

    out = img.copy()

    foreground = mask > 0

    out[foreground] = (
        (1 - alpha) * out[foreground]
        + alpha * color_mask[foreground]
    )

    return np.clip(out, 0, 1)


# ============================================================
# CURVAS DE ENTRENAMIENTO
# ============================================================

def save_training_curves(
    history_csv: str | Path,
    output_path: str | Path,
) -> None:
    import csv

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"No se pudieron guardar curvas de entrenamiento: matplotlib no está disponible ({exc}).")
        return

    with Path(history_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    epochs = [int(row["epoch"]) for row in rows]
    val_dice = [float(row["val_dice"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.set_xlabel("Época")
    ax1.set_ylabel("Dice")

    ax1.plot(
        epochs,
        val_dice,
        label="Val Dice",
    )

    if rows and "train_dice" in rows[0]:
        train_dice = [float(row["train_dice"]) for row in rows]
        ax1.plot(
            epochs,
            train_dice,
            linestyle="--",
            label="Train Dice",
        )

    ax1.tick_params(axis="y")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss")

    ax2.plot(
        epochs,
        train_loss,
        label="Train Loss",
    )

    ax2.plot(
        epochs,
        val_loss,
        linestyle="--",
        label="Val Loss",
    )

    ax2.tick_params(axis="y")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="center right",
    )

    ax1.grid(True, alpha=0.3)

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

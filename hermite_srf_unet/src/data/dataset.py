from __future__ import annotations

from pathlib import Path
from typing import Optional

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ============================================================
# UTILIDADES
# ============================================================

def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"No existe la carpeta: {folder}")

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]

    return sorted(files)


def find_matching_mask(image_path: Path, mask_dir: Path) -> Path:
    """
    Busca la máscara correspondiente a una imagen.

    Primero intenta coincidencia exacta:
        image.png -> mask_dir/image.png

    Si no existe, busca por stem:
        image.jpg -> mask_dir/image.png
    """

    exact = mask_dir / image_path.name

    if exact.exists():
        return exact

    candidates = [
        p for p in mask_dir.iterdir()
        if p.is_file()
        and p.stem == image_path.stem
        and p.suffix.lower() in IMG_EXTS
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No se encontró máscara para {image_path.name} en {mask_dir}"
        )

    return candidates[0]


# ============================================================
# CONVERSIÓN DE MÁSCARAS RGB A CLASES
# ============================================================

CLASS_COLOR_TO_ID = {
    (0, 0, 0): 0,        # fondo
    (255, 0, 0): 1,      # rojo
    (0, 255, 0): 2,      # verde
    (0, 0, 255): 3,      # azul
}


def _format_unique_values(values: np.ndarray, max_items: int = 12) -> str:
    vals = values[:max_items].tolist()
    suffix = "..." if len(values) > max_items else ""
    return f"{vals}{suffix}"


def mask_to_class_ids(mask: Image.Image, num_classes: int = 4) -> np.ndarray:
    """
    Convierte una máscara RGB coloreada a IDs de clase.

    Codificación esperada:

        negro -> clase 0: fondo
        rojo  -> clase 1
        verde -> clase 2
        azul  -> clase 3

    Máscara original:
        RGB, con valores como:
        (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)

    Salida:
        arreglo H,W con valores:
        0, 1, 2, 3
    """

    if num_classes != 4:
        raise ValueError(
            "Estas máscaras usan 4 clases contando fondo: "
            f"0=fondo, 1=rojo, 2=verde, 3=azul. num_classes={num_classes}."
        )

    arr = np.array(mask)

    # En modo P los valores H,W son índices de paleta, no necesariamente IDs.
    # Convertimos a RGB para respetar los colores reales de la anotación.
    if arr.ndim == 2 and mask.mode != "P":
        unique = np.unique(arr)
        if np.all((0 <= unique) & (unique < num_classes)):
            return arr.astype(np.int64)

        raise ValueError(
            "Máscara en escala de grises con valores fuera de 0..3. "
            f"Valores encontrados: {_format_unique_values(unique)}. "
            "Para multiclase usa IDs 0,1,2,3 o colores RGB exactos "
            "negro/rojo/verde/azul."
        )

    mask_rgb = np.array(mask.convert("RGB"))

    h, w, _ = mask_rgb.shape
    out = np.zeros((h, w), dtype=np.int64)

    known = np.zeros((h, w), dtype=bool)

    for color, class_id in CLASS_COLOR_TO_ID.items():
        color_arr = np.array(color, dtype=np.uint8)
        matches = np.all(mask_rgb == color_arr, axis=-1)
        out[matches] = class_id
        known |= matches

    if not np.all(known):
        unknown = np.unique(mask_rgb[~known].reshape(-1, 3), axis=0)
        unknown_colors = [tuple(int(x) for x in c) for c in unknown[:12]]
        suffix = "..." if len(unknown) > 12 else ""
        raise ValueError(
            "La máscara contiene colores que no corresponden a ninguna clase. "
            f"Colores desconocidos: {unknown_colors}{suffix}. "
            "Usa exactamente negro=(0,0,0), rojo=(255,0,0), "
            "verde=(0,255,0), azul=(0,0,255)."
        )

    return out


def rgb_mask_to_class_ids(mask: Image.Image) -> np.ndarray:
    """Alias conservado para compatibilidad con código existente."""

    return mask_to_class_ids(mask, num_classes=4)


def print_mask_rgb_info(mask: Image.Image, name: str = "") -> None:
    """
    Función de diagnóstico para revisar los colores únicos de una máscara.
    No afecta el entrenamiento.
    """

    arr = np.array(mask.convert("RGB"))
    flat = arr.reshape(-1, 3)

    colors = np.unique(flat, axis=0)

    print(f"\n{name}")
    print("Colores únicos en máscara RGB:")

    for c in colors:
        print(tuple(int(x) for x in c))


# ============================================================
# DATASET
# ============================================================

class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        image_size: tuple[int, int] = (256, 256),
        segmentation_mode: str = "multiclass",
        num_classes: int = 4,
        image_mode: str = "rgb",
        mask_threshold: int = 127,
        mean: Optional[list[float]] = None,
        std: Optional[list[float]] = None,
        augment: bool = False,
        horizontal_flip: float = 0.5,
        vertical_flip: float = 0.0,
        rotation_degrees: float = 0.0,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        self.image_paths = list_images(self.images_dir)

        if not self.image_paths:
            raise FileNotFoundError(f"No hay imágenes en {self.images_dir}")

        self.mask_paths = [
            find_matching_mask(image_path, self.masks_dir)
            for image_path in self.image_paths
        ]

        self.image_size = tuple(image_size)
        self.segmentation_mode = segmentation_mode.lower()
        self.num_classes = int(num_classes)
        self.image_mode = image_mode.lower()
        self.mask_threshold = int(mask_threshold)

        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

        self.augment = bool(augment)
        self.horizontal_flip = float(horizontal_flip)
        self.vertical_flip = float(vertical_flip)
        self.rotation_degrees = float(rotation_degrees)

        if self.segmentation_mode not in {"binary", "multiclass"}:
            raise ValueError("segmentation_mode debe ser 'binary' o 'multiclass'")

        if self.segmentation_mode == "multiclass" and self.num_classes != 4:
            raise ValueError(
                "Para estas máscaras multiclase, num_classes debe ser 4: "
                "clase 0=fondo, clase 1=rojo, clase 2=verde, clase 3=azul."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path)

        if self.image_mode == "grayscale":
            return img.convert("L")

        return img.convert("RGB")

    def _load_mask(self, path: Path) -> Image.Image:
        """
        IMPORTANTE:
        No convertir a escala de grises aquí.

        Tus máscaras están codificadas por color RGB:
            negro, rojo, verde, azul

        Si haces convert("L"), se pierde la codificación de clases.
        """

        return Image.open(path)

    def _augment_pair(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        if not self.augment:
            return image, mask

        if random.random() < self.horizontal_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if random.random() < self.vertical_flip:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        if self.rotation_degrees > 0:
            angle = random.uniform(
                -self.rotation_degrees,
                self.rotation_degrees,
            )

            image = image.rotate(
                angle,
                resample=Image.BILINEAR,
            )

            mask = mask.rotate(
                angle,
                resample=Image.NEAREST,
            )

        return image, mask

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        # En PIL el tamaño se pasa como (W, H).
        # En config normalmente usamos [H, W].
        h, w = self.image_size

        image = image.resize((w, h), Image.BILINEAR)

        # Muy importante:
        # La máscara se redimensiona con NEAREST para no crear colores intermedios.
        mask = mask.resize((w, h), Image.NEAREST)

        image, mask = self._augment_pair(image, mask)

        # -------------------------
        # Imagen
        # -------------------------
        image_np = np.array(image).astype(np.float32) / 255.0

        if image_np.ndim == 2:
            image_np = image_np[..., None]

        image_t = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()

        mean = torch.tensor(
            self.mean[: image_t.shape[0]],
            dtype=torch.float32,
        ).view(-1, 1, 1)

        std = torch.tensor(
            self.std[: image_t.shape[0]],
            dtype=torch.float32,
        ).view(-1, 1, 1)

        image_t = (image_t - mean) / std

        # -------------------------
        # Máscara
        # -------------------------
        if self.segmentation_mode == "binary":
            mask_np = np.array(mask.convert("L"))
            mask_np = (mask_np > self.mask_threshold).astype(np.float32)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0)

        elif self.segmentation_mode == "multiclass":
            mask_np = mask_to_class_ids(mask, num_classes=self.num_classes)

            # Para CrossEntropyLoss, la máscara debe ser LongTensor H,W
            mask_t = torch.from_numpy(mask_np.astype(np.int64))

        return {
            "image": image_t,
            "mask": mask_t,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "name": image_path.stem,
        }


# ============================================================
# FACTORY DESDE CONFIG
# ============================================================

def make_dataset_from_config(cfg: dict, split: str) -> SegmentationDataset:
    dcfg = cfg["data"]
    aug_cfg = dcfg.get("augment", {})

    use_aug = split == "train" and bool(aug_cfg.get("train", False))

    return SegmentationDataset(
        images_dir=dcfg[f"{split}_images"],
        masks_dir=dcfg[f"{split}_masks"],
        image_size=tuple(dcfg.get("image_size", [256, 256])),
        segmentation_mode=dcfg.get("segmentation_mode", "multiclass"),
        num_classes=int(dcfg.get("num_classes", 4)),
        image_mode=dcfg.get("image_mode", "rgb"),
        mask_threshold=int(dcfg.get("mask_threshold", 127)),
        mean=dcfg.get("normalize", {}).get("mean", [0.485, 0.456, 0.406]),
        std=dcfg.get("normalize", {}).get("std", [0.229, 0.224, 0.225]),
        augment=use_aug,
        horizontal_flip=float(aug_cfg.get("horizontal_flip", 0.5)),
        vertical_flip=float(aug_cfg.get("vertical_flip", 0.0)),
        rotation_degrees=float(aug_cfg.get("rotation_degrees", 0.0)),
    )

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import random

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    return sorted(files)


def find_matching_mask(image_path: Path, mask_dir: Path) -> Path:
    exact = mask_dir / image_path.name
    if exact.exists():
        return exact
    candidates = [p for p in mask_dir.iterdir() if p.stem == image_path.stem and p.suffix.lower() in IMG_EXTS]
    if not candidates:
        raise FileNotFoundError(f"No se encontro mascara para {image_path.name} en {mask_dir}")
    return candidates[0]


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        image_size: tuple[int, int] = (256, 256),
        segmentation_mode: str = "binary",
        num_classes: int = 2,
        image_mode: str = "rgb",
        mask_threshold: int = 127,
        mean: Optional[list[float]] = None,
        std: Optional[list[float]] = None,
        augment: bool = False,
        augmentation_backend: str = "basic",
        horizontal_flip: float = 0.5,
        vertical_flip: float = 0.0,
        rotation_degrees: float = 0.0,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_paths = list_images(self.images_dir)
        if not self.image_paths:
            raise FileNotFoundError(f"No hay imagenes en {self.images_dir}")
        self.mask_paths = [find_matching_mask(p, self.masks_dir) for p in self.image_paths]
        self.image_size = tuple(image_size)
        self.segmentation_mode = segmentation_mode.lower()
        self.num_classes = int(num_classes)
        self.image_mode = image_mode.lower()
        self.mask_threshold = int(mask_threshold)
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.augment = augment
        self.augmentation_backend = augmentation_backend.lower()
        self.horizontal_flip = float(horizontal_flip)
        self.vertical_flip = float(vertical_flip)
        self.rotation_degrees = float(rotation_degrees)
        self.albumentations_transform = (
            self._build_albumentations_transform() if self.augmentation_backend == "albumentations" else None
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path)
        if self.image_mode == "grayscale":
            return img.convert("L")
        return img.convert("RGB")

    def _load_mask(self, path: Path) -> Image.Image:
        # Para mascaras multiclass se asume que los valores de gris son IDs de clase.
        return Image.open(path).convert("L")

    def _build_albumentations_transform(self):
        try:
            import albumentations as A
        except Exception as exc:
            raise ImportError(
                "Albumentations no esta instalado. Instala requirements.txt o cambia "
                "data.augment.backend a 'basic'."
            ) from exc

        h, w = self.image_size
        if not self.augment:
            return A.Compose([A.Resize(height=h, width=w)])

        geometric = A.OneOf(
            [
                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0, p=0.1, border_mode=0),
                A.ShiftScaleRotate(scale_limit=0, rotate_limit=30, shift_limit=0, p=0.1, border_mode=0),
                A.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=0.6, border_mode=0),
                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30, shift_limit=0.1, p=0.2, border_mode=0),
            ],
            p=0.9,
        )
        quality = A.OneOf(
            [
                A.Perspective(p=0.2),
                A.GaussNoise(p=0.2),
                A.Sharpen(p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
            ],
            p=0.5,
        )
        color_transforms = []
        if self.image_mode != "grayscale":
            color_transforms = [
                A.OneOf(
                    [
                        A.CLAHE(p=0.25),
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
                        A.RandomGamma(p=0.25),
                        A.HueSaturationValue(p=0.25),
                    ],
                    p=0.3,
                )
            ]

        return A.Compose(
            [
                A.Resize(height=h, width=w),
                A.Rotate(limit=35, p=0.5, border_mode=0),
                A.OneOf(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                    ],
                    p=0.8,
                ),
                geometric,
                quality,
                *color_transforms,
            ]
        )

    def _augment_pair(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if not self.augment:
            return image, mask
        if random.random() < self.horizontal_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < self.vertical_flip:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = image.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)
        return image, mask

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        if self.albumentations_transform is not None:
            transformed = self.albumentations_transform(image=np.array(image), mask=np.array(mask))
            image_np = transformed["image"]
            mask_np = transformed["mask"]
        else:
            # PIL usa size=(W,H); config usa [H,W].
            h, w = self.image_size
            image = image.resize((w, h), Image.BILINEAR)
            mask = mask.resize((w, h), Image.NEAREST)
            image, mask = self._augment_pair(image, mask)
            image_np = np.array(image)
            mask_np = np.array(mask)

        image_np = image_np.astype(np.float32) / 255.0
        if image_np.ndim == 2:
            image_np = image_np[..., None]
        image_t = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
        mean = torch.tensor(self.mean[: image_t.shape[0]], dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(self.std[: image_t.shape[0]], dtype=torch.float32).view(-1, 1, 1)
        image_t = (image_t - mean) / std

        if self.segmentation_mode == "binary":
            mask_np = (mask_np > self.mask_threshold).astype(np.float32)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0)  # [1,H,W]
        elif self.segmentation_mode == "multiclass":
            mask_np = np.rint(mask_np).astype(np.int64)
            valid = np.isin(mask_np, np.arange(self.num_classes))
            if not valid.all():
                bad_values = np.unique(mask_np[~valid])
                raise ValueError(
                    f"La mascara {mask_path} contiene clases fuera de rango "
                    f"0..{self.num_classes - 1}: {bad_values.tolist()}"
                )
            mask_t = torch.from_numpy(mask_np)  # [H,W]
        else:
            raise ValueError("segmentation_mode debe ser 'binary' o 'multiclass'")

        return {
            "image": image_t,
            "mask": mask_t,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "name": image_path.stem,
        }


def make_dataset_from_config(cfg: dict, split: str) -> SegmentationDataset:
    dcfg = cfg["data"]
    aug_cfg = dcfg.get("augment", {})
    use_aug = split == "train" and bool(aug_cfg.get("train", False))
    return SegmentationDataset(
        images_dir=dcfg[f"{split}_images"],
        masks_dir=dcfg[f"{split}_masks"],
        image_size=tuple(dcfg.get("image_size", [256, 256])),
        segmentation_mode=dcfg.get("segmentation_mode", "binary"),
        num_classes=int(dcfg.get("num_classes", 2)),
        image_mode=dcfg.get("image_mode", "rgb"),
        mask_threshold=int(dcfg.get("mask_threshold", 127)),
        mean=dcfg.get("normalize", {}).get("mean", [0.485, 0.456, 0.406]),
        std=dcfg.get("normalize", {}).get("std", [0.229, 0.224, 0.225]),
        augment=use_aug,
        augmentation_backend=aug_cfg.get("backend", "basic"),
        horizontal_flip=float(aug_cfg.get("horizontal_flip", 0.5)),
        vertical_flip=float(aug_cfg.get("vertical_flip", 0.0)),
        rotation_degrees=float(aug_cfg.get("rotation_degrees", 0.0)),
    )

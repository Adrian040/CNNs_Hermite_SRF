# PROYECTO FINAL - APRENDIZAJE DE MÁQUINA PARA VC
# Segementación multiclase con U.Net
#
#
# Autores:
 # Oscar Eduardo Morales Toledo
 # Katya Verónica Fuentes Sánchez
 # Adrián Jesús Maldonado Oclica

from __future__ import annotations

import random
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm


# ============================================================
# RUTAS
# ============================================================

TRAIN_IMAGES = r"C:\Users\oscar\OneDrive\Documents\Maestria\2doSemestre\Aprendizaje Automatico para Vision Computacional\ProyectoFinal\CNNs_Hermite_SRF\hermite_srf_unet\data\train\images"
TRAIN_MASKS = r"C:\Users\oscar\OneDrive\Documents\Maestria\2doSemestre\Aprendizaje Automatico para Vision Computacional\ProyectoFinal\CNNs_Hermite_SRF\hermite_srf_unet\data\train\masks"

VAL_IMAGES = r"C:\Users\oscar\OneDrive\Documents\Maestria\2doSemestre\Aprendizaje Automatico para Vision Computacional\ProyectoFinal\CNNs_Hermite_SRF\hermite_srf_unet\data\val\images"
VAL_MASKS = r"C:\Users\oscar\OneDrive\Documents\Maestria\2doSemestre\Aprendizaje Automatico para Vision Computacional\ProyectoFinal\CNNs_Hermite_SRF\hermite_srf_unet\data\val\masks"

TEST_IMAGES = r"C:\Users\oscar\OneDrive\Documents\Maestria\2doSemestre\Aprendizaje Automatico para Vision Computacional\ProyectoFinal\CNNs_Hermite_SRF\hermite_srf_unet\data\test\images"
TEST_MASKS = r"C:\Users\oscar\OneDrive\Documents\Maestria\2doSemestre\Aprendizaje Automatico para Vision Computacional\ProyectoFinal\CNNs_Hermite_SRF\hermite_srf_unet\data\test\masks"


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

NUM_CLASSES = 4
IMAGE_SIZE = 512

EPOCHS = 100
BATCH_SIZE = 1
LEARNING_RATE = 7e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

OUTPUT_DIR = Path(
    r"C:\Users\oscar\OneDrive\Documents\Maestria\2doSemestre\Aprendizaje Automatico para Vision Computacional\ProyectoFinal\segmentacion_unet_normal_preprocesada_dice_resultados"
)

SAVE_EVERY = 10
VISUALIZE_EVERY = 5
NUM_DEBUG_IMAGES = 15

EARLY_STOPPING_PATIENCE = 35

# La pérdida total será:
# LOSS = CE_WEIGHT * CrossEntropy ponderada + DICE_WEIGHT * Dice Loss
CE_WEIGHT = 0.35
DICE_WEIGHT = 0.65
BACKGROUND_WEIGHT = 0.15
MAX_CLASS_WEIGHT = 12.0
USE_AMP = True
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

# ============================================================
# PREPROCESAMIENTO
# ============================================================

# Percentiles para recortar intensidades extremas.
PERCENTILE_LOW = 1.0
PERCENTILE_HIGH = 99.0

USE_ZSCORE_AFTER_PERCENTILE = False

VALID_REGION_MODE = "nonzero"  # opciones: "all", "nonzero", "negative"


# ============================================================
# CONFIGURACIÓN DEL MODELO U-NET 
# ============================================================

HERMITE_KERNEL_SIZE = 7
HERMITE_MAX_ORDER = 3
HERMITE_SCALES = (1.5,)

# En esta versión normal no se usan filtros Hermite/SRF.
# Se conservan estas variables solo para que se comparen hiperparámetros
# generales con la versión Hermite, pero no se usan para construir el modelo.
SRF_STAGES = None
BASE_CHANNELS = 24
DEPTH = 4
DROPOUT = 0.05


# ============================================================
# AUGMENTATIONS MODERADAS
# ============================================================

USE_AUGMENTATIONS = True

AUG_HORIZONTAL_FLIP_P = 0.5
AUG_VERTICAL_FLIP_P = 0.10

AUG_ROTATE_P = 0.55
AUG_ROTATE_DEGREES = 12

AUG_AFFINE_P = 0.45
AUG_TRANSLATE_PIXELS = 10
AUG_SCALE_MIN = 0.90
AUG_SCALE_MAX = 1.10
AUG_SHEAR_DEGREES = 4

AUG_GAMMA_P = 0.25
AUG_GAMMA_RANGE = (0.90, 1.10)

AUG_BLUR_P = 0.10
AUG_NOISE_P = 0.20
AUG_NOISE_STD = 0.015

AUG_CUTOUT_P = 0.0
AUG_CUTOUT_SIZE = 40


# ============================================================
# OVERSAMPLING DE IMÁGENES CON CLASES PEQUEÑAS
# ============================================================

USE_WEIGHTED_SAMPLER = True
RARE_CLASS_BOOST = 4.0

# ============================================================
# SEMILLA Y COMPATIBILIDAD PIL
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


set_seed(SEED)

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR
    RESAMPLE_NEAREST = Image.NEAREST


# ============================================================
# ARCHIVOS Y EMPAREJAMIENTO
# ============================================================

def list_files(folder: str | Path):
    folder = Path(folder)
    files = []

    for ext in IMAGE_EXTENSIONS:
        files.extend(list(folder.glob(f"*{ext}")))
        files.extend(list(folder.glob(f"*{ext.upper()}")))

    return sorted(list(set(files)))


def check_folder(path: str | Path, name: str):
    path = Path(path)

    print(f"{name}: {path}")

    if not path.exists():
        raise FileNotFoundError(f"No existe la carpeta: {path}")

    files = list_files(path)

    print(f"  archivos encontrados: {len(files)}")

    if len(files) == 0:
        raise FileNotFoundError(f"No encontré imágenes/máscaras en: {path}")

    return path


def check_all_folders():
    print("\n" + "=" * 70)
    print("REVISANDO CARPETAS")
    print("=" * 70)

    check_folder(TRAIN_IMAGES, "TRAIN_IMAGES")
    check_folder(TRAIN_MASKS, "TRAIN_MASKS")
    check_folder(VAL_IMAGES, "VAL_IMAGES")
    check_folder(VAL_MASKS, "VAL_MASKS")
    check_folder(TEST_IMAGES, "TEST_IMAGES")
    check_folder(TEST_MASKS, "TEST_MASKS")

    print("=" * 70 + "\n")


def make_pairs(images_dir: str | Path, masks_dir: str | Path):
    image_files = list_files(images_dir)
    mask_files = list_files(masks_dir)

    image_dict = {p.stem: p for p in image_files}
    mask_dict = {p.stem: p for p in mask_files}

    common_names = sorted(list(set(image_dict.keys()) & set(mask_dict.keys())))

    pairs = []

    for name in common_names:
        pairs.append(
            {
                "name": name,
                "image": image_dict[name],
                "mask": mask_dict[name],
            }
        )

    missing_masks = sorted(list(set(image_dict.keys()) - set(mask_dict.keys())))
    orphan_masks = sorted(list(set(mask_dict.keys()) - set(image_dict.keys())))

    print(f"Imágenes: {len(image_files)}")
    print(f"Máscaras: {len(mask_files)}")
    print(f"Pares válidos: {len(pairs)}")
    print(f"Imágenes sin máscara: {len(missing_masks)}")
    print(f"Máscaras sin imagen: {len(orphan_masks)}")

    if len(pairs) == 0:
        raise ValueError(
            "No se encontraron pares imagen-máscara. "
            "Revisa que tengan el mismo nombre base."
        )

    return pairs


# ============================================================
# 9. LECTURA, PREPROCESAMIENTO Y MAPEO DE MÁSCARAS
# ============================================================

def read_image_float(image_path: str | Path) -> np.ndarray:
    """
    Lee imagen monocanal preservando valores tipo float32/int16/int32 si existen.
    Si viene RGB, se promedia a monocanal.
    """
    img = Image.open(image_path)
    arr = np.array(img).astype(np.float32)

    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)

    return arr


def read_mask_raw(mask_path: str | Path) -> np.ndarray:
    mask = Image.open(mask_path)

    mask = mask.convert("L")
    return np.array(mask)


def preprocess_image_percentile(image_np: np.ndarray) -> np.ndarray:
    """
    Preprocesamiento recomendado para tus imágenes:
    1) monocanal
    2) recorte por percentiles
    3) escala a [0,1]
    4) opcional z-score usando región útil
    """
    image_np = image_np.astype(np.float32)

    finite = np.isfinite(image_np)
    if not finite.any():
        return np.zeros_like(image_np, dtype=np.float32)

    valid_values = image_np[finite]

    p_low = np.percentile(valid_values, PERCENTILE_LOW)
    p_high = np.percentile(valid_values, PERCENTILE_HIGH)

    if abs(p_high - p_low) < 1e-8:
        image_np = np.zeros_like(image_np, dtype=np.float32)
    else:
        image_np = np.clip(image_np, p_low, p_high)
        image_np = (image_np - p_low) / (p_high - p_low + 1e-8)

    image_np = np.clip(image_np, 0.0, 1.0).astype(np.float32)

    if USE_ZSCORE_AFTER_PERCENTILE:
        if VALID_REGION_MODE == "nonzero":
            valid = image_np != 0
        elif VALID_REGION_MODE == "negative":
            # Este modo aplica antes de percentiles en teoría.
            valid = image_np > 0
        else:
            valid = np.ones_like(image_np, dtype=bool)

        if valid.any():
            mu = float(image_np[valid].mean())
            sigma = float(image_np[valid].std())
        else:
            mu = float(image_np.mean())
            sigma = float(image_np.std())

        image_np = (image_np - mu) / (sigma + 1e-8)

    return image_np.astype(np.float32)


def float01_to_pil_l(image_np: np.ndarray) -> Image.Image:
    image_np = np.clip(image_np, 0.0, 1.0)
    image_u8 = (image_np * 255.0).round().astype(np.uint8)
    return Image.fromarray(image_u8, mode="L")


def find_global_mask_values(all_pairs):
    values = set()

    print("\nBuscando valores únicos globales de máscaras...")

    for item in tqdm(all_pairs, desc="Valores máscaras"):
        mask_np = read_mask_raw(item["mask"])
        unique_values = np.unique(mask_np)

        for v in unique_values:
            values.add(int(v))

    values = sorted(list(values))

    print("Valores únicos globales encontrados:")
    print(values)

    return values


def build_value_mapping(global_values, num_classes: int):
    """
    Casos:
    - Si la máscara ya tiene 0..C-1, se deja igual.
    - Si tiene valores tipo 0,85,170,255, se remapea a 0,1,2,3.
    """
    max_allowed = num_classes - 1
    already_ok = all(0 <= v <= max_allowed for v in global_values)

    if already_ok:
        mapping = {v: v for v in global_values}
        print("\nLas máscaras ya parecen estar codificadas como clases.")
        print("Mapping usado:", mapping)
        return mapping

    if len(global_values) <= num_classes:
        mapping = {v: i for i, v in enumerate(global_values)}
        print("\nLas máscaras no están en 0..C-1. Se remapearán.")
        print("Mapping usado:", mapping)
        return mapping

    raise ValueError(
        f"Hay más valores únicos en las máscaras ({len(global_values)}) "
        f"que NUM_CLASSES ({num_classes}).\n"
        f"Valores encontrados: {global_values}\n"
        "Esto puede indicar que tus máscaras son RGB/continuas o que NUM_CLASSES está mal."
    )


def apply_mask_mapping(mask_np: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(mask_np, dtype=np.int64)

    for old_value, new_value in mapping.items():
        out[mask_np == old_value] = new_value

    return out


# ============================================================
# AUGMENTATIONS SINCRONIZADAS
# ============================================================

def random_affine_pair(image: Image.Image, mask: Image.Image):
    angle = random.uniform(-AUG_ROTATE_DEGREES, AUG_ROTATE_DEGREES)

    tx = random.uniform(-AUG_TRANSLATE_PIXELS, AUG_TRANSLATE_PIXELS)
    ty = random.uniform(-AUG_TRANSLATE_PIXELS, AUG_TRANSLATE_PIXELS)

    scale = random.uniform(AUG_SCALE_MIN, AUG_SCALE_MAX)

    shear = random.uniform(-AUG_SHEAR_DEGREES, AUG_SHEAR_DEGREES)
    shear_rad = np.deg2rad(shear)

    a = 1.0 / scale
    b = np.tan(shear_rad) / scale
    d = 0.0
    e = 1.0 / scale

    c = -tx
    f = -ty

    image = image.transform(
        image.size,
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=RESAMPLE_BILINEAR,
        fillcolor=0,
    )

    mask = mask.transform(
        mask.size,
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=RESAMPLE_NEAREST,
        fillcolor=0,
    )

    image = image.rotate(
        angle,
        resample=RESAMPLE_BILINEAR,
        fillcolor=0,
    )

    mask = mask.rotate(
        angle,
        resample=RESAMPLE_NEAREST,
        fillcolor=0,
    )

    return image, mask


def apply_geometric_augmentations(image: Image.Image, mask: Image.Image):
    if random.random() < AUG_HORIZONTAL_FLIP_P:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() < AUG_VERTICAL_FLIP_P:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    if random.random() < AUG_AFFINE_P:
        image, mask = random_affine_pair(image, mask)

    elif random.random() < AUG_ROTATE_P:
        angle = random.uniform(-AUG_ROTATE_DEGREES, AUG_ROTATE_DEGREES)

        image = image.rotate(
            angle,
            resample=RESAMPLE_BILINEAR,
            fillcolor=0,
        )

        mask = mask.rotate(
            angle,
            resample=RESAMPLE_NEAREST,
            fillcolor=0,
        )

    return image, mask


def apply_intensity_augmentations_np(image_np: np.ndarray):
    image_np = image_np.astype(np.float32)
    image_np = np.clip(image_np, 0.0, 1.0)

    if random.random() < AUG_GAMMA_P:
        gamma = random.uniform(*AUG_GAMMA_RANGE)
        image_np = image_np ** gamma

    if random.random() < AUG_BLUR_P:
        pil = float01_to_pil_l(image_np)
        radius = random.uniform(0.3, 1.0)
        pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))
        image_np = np.array(pil).astype(np.float32) / 255.0

    if random.random() < AUG_NOISE_P:
        noise = np.random.normal(
            loc=0.0,
            scale=AUG_NOISE_STD,
            size=image_np.shape,
        ).astype(np.float32)
        image_np = image_np + noise

    if random.random() < AUG_CUTOUT_P:
        h, w = image_np.shape
        size = random.randint(max(10, AUG_CUTOUT_SIZE // 2), AUG_CUTOUT_SIZE)
        y = random.randint(0, max(0, h - size))
        x = random.randint(0, max(0, w - size))
        image_np[y:y + size, x:x + size] = 0.0

    return np.clip(image_np, 0.0, 1.0).astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class SegmentationDataset(Dataset):
    def __init__(
        self,
        pairs,
        value_mapping: dict[int, int],
        image_size: int = 512,
        augment: bool = False,
    ):
        self.pairs = pairs
        self.value_mapping = value_mapping
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]

        image_np = read_image_float(item["image"])
        image_np = preprocess_image_percentile(image_np)

        mask_np = read_mask_raw(item["mask"])

        image_pil = float01_to_pil_l(image_np)
        mask_pil = Image.fromarray(mask_np.astype(np.uint8), mode="L")

        image_pil = image_pil.resize(
            (self.image_size, self.image_size),
            resample=RESAMPLE_BILINEAR,
        )

        mask_pil = mask_pil.resize(
            (self.image_size, self.image_size),
            resample=RESAMPLE_NEAREST,
        )

        if self.augment and USE_AUGMENTATIONS:
            image_pil, mask_pil = apply_geometric_augmentations(
                image_pil,
                mask_pil,
            )

        image_np = np.array(image_pil).astype(np.float32) / 255.0
        mask_np = np.array(mask_pil)

        mask_np = apply_mask_mapping(mask_np, self.value_mapping)

        if self.augment and USE_AUGMENTATIONS:
            image_np = apply_intensity_augmentations_np(image_np)

        # Monocanal: [1, H, W]
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_np).long()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "name": item["name"],
        }


# ============================================================
# U-NET NORMAL MONOCANAL
# ============================================================

class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not use_batchnorm,
            )
        ]

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU(inplace=True))

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            ConvBNReLU(
                in_channels,
                out_channels,
                use_batchnorm=use_batchnorm,
                dropout=dropout,
            ),
            ConvBNReLU(
                out_channels,
                out_channels,
                use_batchnorm=use_batchnorm,
                dropout=dropout,
            ),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                in_channels,
                out_channels,
                use_batchnorm=use_batchnorm,
                dropout=dropout,
            ),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: bool = False,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
            up_channels = in_channels
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            up_channels = in_channels // 2

        self.conv = DoubleConv(
            up_channels + skip_channels,
            out_channels,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(
                x1,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetNormal(nn.Module):
    """
    U-Net normal para comparación contra Hermite/SRF U-Net.

    Mantiene el mismo flujo general de entrenamiento:
    - entrada monocanal [1,H,W]
    - preprocesamiento por percentiles
    - augmentations moderadas
    - CE ponderada + Dice Loss
    - métricas por clase

    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 24,
        depth: int = 4,
        bilinear: bool = False,
        use_batchnorm: bool = True,
        dropout: float = 0.05,
    ):
        super().__init__()

        if depth < 2:
            raise ValueError("depth debe ser >= 2.")

        self.depth = int(depth)

        channels = [
            base_channels * (2**i)
            for i in range(depth)
        ]

        self.inc = DoubleConv(
            in_channels,
            channels[0],
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

        self.downs = nn.ModuleList()
        for i in range(1, depth):
            self.downs.append(
                Down(
                    channels[i - 1],
                    channels[i],
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                )
            )

        self.ups = nn.ModuleList()
        rev_channels = channels[::-1]

        for i in range(depth - 1):
            in_ch = rev_channels[i]
            skip_ch = rev_channels[i + 1]
            out_ch = rev_channels[i + 1]

            self.ups.append(
                Up(
                    in_ch,
                    skip_ch,
                    out_ch,
                    bilinear=bilinear,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                )
            )

        self.outc = nn.Conv2d(
            channels[0],
            num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        skips = []

        x = self.inc(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = skips[-1]

        for i, up in enumerate(self.ups):
            skip = skips[-2 - i]
            x = up(x, skip)

        return self.outc(x)


# ============================================================
# PÉRDIDAS
# ============================================================

class MulticlassDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
        include_background: bool = False,
        smooth: float = 1e-6,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.include_background = bool(include_background)
        self.smooth = float(smooth)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets = targets.long()

        one_hot = F.one_hot(
            targets,
            num_classes=self.num_classes,
        )

        one_hot = one_hot.permute(0, 3, 1, 2).float()

        start = 0 if self.include_background else 1

        probs = probs[:, start:]
        one_hot = one_hot[:, start:]

        dims = (0, 2, 3)

        intersection = torch.sum(probs * one_hot, dim=dims)
        denominator = torch.sum(probs + one_hot, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        loss_per_class = 1.0 - dice

        if self.class_weights is not None:
            weights = self.class_weights[start:].to(loss_per_class.device)
            valid = weights > 0

            if valid.any():
                loss_per_class = loss_per_class[valid]
                weights = weights[valid]
                return torch.sum(loss_per_class * weights) / torch.sum(weights)

        return loss_per_class.mean()


class WeightedCEDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor,
        ce_weight: float = 0.35,
        dice_weight: float = 0.65,
    ):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
        )

        self.dice = MulticlassDiceLoss(
            num_classes=num_classes,
            class_weights=class_weights,
            include_background=False,
        )

        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets.long())
        dice_loss = self.dice(logits, targets.long())

        loss = self.ce_weight * ce_loss
        loss = loss + self.dice_weight * dice_loss

        return loss


# ============================================================
# PESOS POR CLASE Y SAMPLER
# ============================================================

@torch.no_grad()
def compute_class_counts(dataset, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)

    print("\nCalculando balance de clases en train...")

    for sample in tqdm(dataset, desc="Conteo clases"):
        mask = sample["mask"].numpy()

        values, value_counts = np.unique(
            mask,
            return_counts=True,
        )

        for v, c in zip(values, value_counts):
            v = int(v)
            if 0 <= v < num_classes:
                counts[v] += int(c)

    return counts


def make_class_weights(
    counts,
    background_weight=0.15,
    max_class_weight=12.0,
):
    counts = counts.astype(np.float64)
    weights = np.zeros_like(counts, dtype=np.float64)

    present = counts > 0

    if not present.any():
        raise ValueError("No se encontraron clases válidas.")

    median_count = np.median(counts[present])

    for c in range(len(counts)):
        if counts[c] == 0:
            weights[c] = 0.0
        else:
            weights[c] = median_count / counts[c]

    weights = np.clip(weights, 0.0, max_class_weight)

    if counts[0] > 0:
        weights[0] = background_weight

    return torch.tensor(weights, dtype=torch.float32)


def print_class_report(counts, weights):
    total = counts.sum()

    print("\n" + "=" * 70)
    print("BALANCE DE CLASES")
    print("=" * 70)

    for c in range(len(counts)):
        pct = 100.0 * counts[c] / total if total > 0 else 0.0
        print(
            f"Clase {c}: "
            f"{counts[c]:>12,d} píxeles | "
            f"{pct:>8.4f}% | "
            f"peso = {weights[c].item():.4f}"
        )

    print("=" * 70 + "\n")


def compute_sample_weights(pairs, value_mapping, num_classes):
    """
    Da más probabilidad a imágenes que contienen clases raras.
    """
    class_image_counts = np.zeros(num_classes, dtype=np.int64)
    image_classes = []

    print("\nCalculando pesos por imagen para oversampling...")

    for item in tqdm(pairs, desc="Sampler"):
        mask_np = read_mask_raw(item["mask"])
        mask_np = apply_mask_mapping(mask_np, value_mapping)
        present = sorted([int(v) for v in np.unique(mask_np) if 0 <= int(v) < num_classes])
        image_classes.append(present)

        for c in present:
            class_image_counts[c] += 1

    print("Número de imágenes donde aparece cada clase:")
    for c in range(num_classes):
        print(f"  clase {c}: {class_image_counts[c]} imágenes")

    weights = []

    for present in image_classes:
        w = 1.0

        for c in present:
            if c == 0:
                continue

            if class_image_counts[c] > 0:
                rarity = len(pairs) / class_image_counts[c]
                w += RARE_CLASS_BOOST * rarity

        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float32)

    # Normalización suave
    weights = weights / weights.mean()

    return weights


# ============================================================
# MÉTRICAS
# ============================================================

@torch.no_grad()
def compute_metrics_from_logits(logits, targets, num_classes):
    preds = torch.argmax(logits, dim=1)

    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    dice_per_class = []
    iou_per_class = []
    precision_per_class = []
    recall_per_class = []

    per_class = {}

    for c in range(1, num_classes):
        pred_c = preds_np == c
        target_c = targets_np == c

        tp = np.logical_and(pred_c, target_c).sum()
        fp = np.logical_and(pred_c, np.logical_not(target_c)).sum()
        fn = np.logical_and(np.logical_not(pred_c), target_c).sum()

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        dice_per_class.append(dice)
        iou_per_class.append(iou)
        precision_per_class.append(precision)
        recall_per_class.append(recall)

        per_class[f"dice_c{c}"] = float(dice)
        per_class[f"iou_c{c}"] = float(iou)
        per_class[f"precision_c{c}"] = float(precision)
        per_class[f"recall_c{c}"] = float(recall)

    metrics = {
        "dice": float(np.mean(dice_per_class)),
        "iou": float(np.mean(iou_per_class)),
        "precision": float(np.mean(precision_per_class)),
        "recall": float(np.mean(recall_per_class)),
    }

    metrics.update(per_class)
    return metrics


class MetricAverager:
    def __init__(self):
        self.data = {}

    def update(self, metrics, n=1):
        for key, value in metrics.items():
            if key not in self.data:
                self.data[key] = {"sum": 0.0, "count": 0}

            self.data[key]["sum"] += float(value) * n
            self.data[key]["count"] += n

    def compute(self):
        out = {}
        for key, item in self.data.items():
            out[key] = item["sum"] / max(item["count"], 1)
        return out


# ============================================================
# ENTRENAMIENTO Y VALIDACIÓN
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    num_classes,
):
    model.train()

    total_loss = 0.0
    averager = MetricAverager()

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None and device == "cuda"

        if use_amp:
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs

        batch_metrics = compute_metrics_from_logits(
            logits.detach(),
            masks.detach(),
            num_classes,
        )

        averager.update(batch_metrics, n=bs)

    metrics = averager.compute()
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    num_classes,
    desc="Val",
):
    model.eval()

    total_loss = 0.0
    averager = MetricAverager()

    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        bs = images.size(0)
        total_loss += loss.item() * bs

        batch_metrics = compute_metrics_from_logits(
            logits,
            masks,
            num_classes,
        )

        averager.update(batch_metrics, n=bs)

    metrics = averager.compute()
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


# ============================================================
# VISUALIZACIÓN
# ============================================================

def denormalize_image(image_tensor):
    """
    image_tensor: [1,H,W] ya está en [0,1] en esta versión.
    """
    image = image_tensor.detach().cpu().squeeze(0).numpy()
    image = np.clip(image, 0.0, 1.0)
    return image


def overlay_mask_gray(image, mask, num_classes, alpha=0.45):
    image_rgb = np.stack([image, image, image], axis=-1).astype(np.float32)

    colors = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.3, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    if num_classes > len(colors):
        rng = np.random.default_rng(123)
        extra = rng.random(
            (num_classes - len(colors), 3),
            dtype=np.float32,
        )
        colors = np.concatenate([colors, extra], axis=0)

    out = image_rgb.copy()

    for c in range(1, num_classes):
        region = mask == c
        if np.any(region):
            out[region] = (1 - alpha) * out[region] + alpha * colors[c]

    return np.clip(out, 0.0, 1.0)


@torch.no_grad()
def save_visualizations(
    model,
    dataset,
    device,
    output_dir,
    epoch,
    num_samples=12,
    num_classes=4,
    prefix="val",
):
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    for i, batch in enumerate(loader):
        if i >= num_samples:
            break

        image_tensor = batch["image"].to(device)
        mask_tensor = batch["mask"]

        logits = model(image_tensor)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        gt = mask_tensor[0].cpu().numpy().astype(np.uint8)
        image = denormalize_image(batch["image"][0])

        name = batch["name"][0]

        print("\n" + "-" * 60)
        print(f"{prefix} | época {epoch} | imagen: {name}")
        print("GT valores únicos:", np.unique(gt))
        print("Pred valores únicos:", np.unique(pred))

        print("Píxeles GT por clase:")
        for c in range(num_classes):
            print(f"  clase {c}: {np.sum(gt == c)}")

        print("Píxeles predichos por clase:")
        for c in range(num_classes):
            print(f"  clase {c}: {np.sum(pred == c)}")

        print("-" * 60)

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Imagen preprocesada")

        axes[1].imshow(
            gt,
            cmap="tab10",
            vmin=0,
            vmax=num_classes - 1,
        )
        axes[1].set_title("GT")

        axes[2].imshow(
            pred,
            cmap="tab10",
            vmin=0,
            vmax=num_classes - 1,
        )
        axes[2].set_title("Predicción")

        axes[3].imshow(
            overlay_mask_gray(
                image,
                gt,
                num_classes,
            )
        )
        axes[3].set_title("Overlay GT")

        axes[4].imshow(
            overlay_mask_gray(
                image,
                pred,
                num_classes,
            )
        )
        axes[4].set_title("Overlay Pred")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        save_path = output_dir / f"{prefix}_epoch_{epoch:03d}_{name}.png"
        plt.savefig(save_path, dpi=200)
        plt.close(fig)


# ============================================================
# GRÁFICAS
# ============================================================

def save_history_plots(history_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Curva de pérdida")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_dice"], label="Train Dice")
    plt.plot(history_df["epoch"], history_df["val_dice"], label="Val Dice")
    plt.xlabel("Época")
    plt.ylabel("Dice")
    plt.title("Curva de Dice")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "dice_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_iou"], label="Train IoU")
    plt.plot(history_df["epoch"], history_df["val_iou"], label="Val IoU")
    plt.xlabel("Época")
    plt.ylabel("IoU")
    plt.title("Curva de IoU")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "iou_curve.png", dpi=200)
    plt.close()

    for c in range(1, NUM_CLASSES):
        col_train = f"train_dice_c{c}"
        col_val = f"val_dice_c{c}"

        if col_train in history_df.columns and col_val in history_df.columns:
            plt.figure(figsize=(8, 5))
            plt.plot(history_df["epoch"], history_df[col_train], label=f"Train Dice clase {c}")
            plt.plot(history_df["epoch"], history_df[col_val], label=f"Val Dice clase {c}")
            plt.xlabel("Época")
            plt.ylabel("Dice")
            plt.title(f"Dice clase {c}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f"dice_clase_{c}.png", dpi=200)
            plt.close()


# ============================================================
# CHECKPOINTS
# ============================================================

def save_model_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val_dice,
    value_mapping,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_dice": best_val_dice,
        "value_mapping": value_mapping,
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "preprocessing": {
            "percentile_low": PERCENTILE_LOW,
            "percentile_high": PERCENTILE_HIGH,
            "monochannel": True,
            "use_zscore": USE_ZSCORE_AFTER_PERCENTILE,
        },
        "model": {
            "architecture": "UNetNormal",
            "base_channels": BASE_CHANNELS,
            "depth": DEPTH,
            "dropout": DROPOUT,
            "uses_hermite_srf": False,
        },
    }

    torch.save(checkpoint, path)


def load_model_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("SEGMENTACIÓN MULTICLASE CON U-NET NORMAL")
    print("MONOCANAL + PERCENTILES + AUGMENTATIONS MODERADAS + DICE")
    print("=" * 70)
    print("DEVICE:", DEVICE)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("NUM_CLASSES:", NUM_CLASSES)
    print("IMAGE_SIZE:", IMAGE_SIZE)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("PREPROCESAMIENTO: monocanal + percentiles")
    print("PERCENTILES:", PERCENTILE_LOW, PERCENTILE_HIGH)
    print("AUGMENTATIONS:", USE_AUGMENTATIONS)
    print("MODELO: U-Net normal sin Hermite/SRF")
    print("LOSS: CE ponderada + Dice Loss")
    print("=" * 70 + "\n")

    check_all_folders()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    visual_dir = OUTPUT_DIR / "visualizaciones"
    plot_dir = OUTPUT_DIR / "graficas"
    log_dir = OUTPUT_DIR / "logs"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("\nEmparejando train...")
    train_pairs = make_pairs(TRAIN_IMAGES, TRAIN_MASKS)

    print("\nEmparejando val...")
    val_pairs = make_pairs(VAL_IMAGES, VAL_MASKS)

    print("\nEmparejando test...")
    test_pairs = make_pairs(TEST_IMAGES, TEST_MASKS)

    all_pairs = train_pairs + val_pairs + test_pairs
    global_values = find_global_mask_values(all_pairs)

    value_mapping = build_value_mapping(
        global_values,
        NUM_CLASSES,
    )

    print("\nCreando datasets...")

    train_dataset = SegmentationDataset(
        train_pairs,
        value_mapping=value_mapping,
        image_size=IMAGE_SIZE,
        augment=True,
    )

    val_dataset = SegmentationDataset(
        val_pairs,
        value_mapping=value_mapping,
        image_size=IMAGE_SIZE,
        augment=False,
    )

    test_dataset = SegmentationDataset(
        test_pairs,
        value_mapping=value_mapping,
        image_size=IMAGE_SIZE,
        augment=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    class_counts = compute_class_counts(
        train_dataset,
        NUM_CLASSES,
    )

    class_weights = make_class_weights(
        class_counts,
        background_weight=BACKGROUND_WEIGHT,
        max_class_weight=MAX_CLASS_WEIGHT,
    )

    print_class_report(
        class_counts,
        class_weights,
    )

    if USE_WEIGHTED_SAMPLER:
        sample_weights = compute_sample_weights(
            train_pairs,
            value_mapping,
            NUM_CLASSES,
        )

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_shuffle = False

    else:
        train_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE == "cuda",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE == "cuda",
    )

    print("\nConstruyendo U-Net normal monocanal...")

    model = UNetNormal(
        in_channels=1,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
        depth=DEPTH,
        bilinear=False,
        use_batchnorm=True,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = WeightedCEDiceLoss(
        num_classes=NUM_CLASSES,
        class_weights=class_weights.to(DEVICE),
        ce_weight=CE_WEIGHT,
        dice_weight=DICE_WEIGHT,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.75,
        patience=12,
    )

    if USE_AMP and DEVICE == "cuda":
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    print("\nVisualización inicial antes de entrenar...")

    save_visualizations(
        model=model,
        dataset=val_dataset,
        device=DEVICE,
        output_dir=visual_dir / "epoch_000_before_training",
        epoch=0,
        num_samples=NUM_DEBUG_IMAGES,
        num_classes=NUM_CLASSES,
        prefix="val_before",
    )

    history = []
    best_val_dice = -1.0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "=" * 70)
        print(f"ÉPOCA {epoch}/{EPOCHS}")
        print("=" * 70)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=DEVICE,
            num_classes=NUM_CLASSES,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE,
            num_classes=NUM_CLASSES,
            desc="Val",
        )

        scheduler.step(val_metrics["dice"])
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_dice": val_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_iou": val_metrics["iou"],
            "train_precision": train_metrics["precision"],
            "val_precision": val_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_recall": val_metrics["recall"],
        }

        for c in range(1, NUM_CLASSES):
            row[f"train_dice_c{c}"] = train_metrics.get(f"dice_c{c}", 0.0)
            row[f"val_dice_c{c}"] = val_metrics.get(f"dice_c{c}", 0.0)
            row[f"train_iou_c{c}"] = train_metrics.get(f"iou_c{c}", 0.0)
            row[f"val_iou_c{c}"] = val_metrics.get(f"iou_c{c}", 0.0)
            row[f"train_recall_c{c}"] = train_metrics.get(f"recall_c{c}", 0.0)
            row[f"val_recall_c{c}"] = val_metrics.get(f"recall_c{c}", 0.0)
            row[f"train_precision_c{c}"] = train_metrics.get(f"precision_c{c}", 0.0)
            row[f"val_precision_c{c}"] = val_metrics.get(f"precision_c{c}", 0.0)

        history.append(row)
        history_df = pd.DataFrame(history)

        history_path = log_dir / "history.csv"
        history_df.to_csv(history_path, index=False)

        save_history_plots(
            history_df,
            plot_dir,
        )

        print(
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {row['train_loss']:.4f} | "
            f"Val Loss: {row['val_loss']:.4f} | "
            f"Train Dice: {row['train_dice']:.4f} | "
            f"Val Dice: {row['val_dice']:.4f} | "
            f"Val IoU: {row['val_iou']:.4f} | "
            f"Val Recall: {row['val_recall']:.4f}"
        )

        for c in range(1, NUM_CLASSES):
            print(
                f"  Clase {c} | "
                f"Val Dice: {row[f'val_dice_c{c}']:.4f} | "
                f"Val IoU: {row[f'val_iou_c{c}']:.4f} | "
                f"Val Recall: {row[f'val_recall_c{c}']:.4f} | "
                f"Val Precision: {row[f'val_precision_c{c}']:.4f}"
            )

        if epoch == 1 or epoch % VISUALIZE_EVERY == 0:
            save_visualizations(
                model=model,
                dataset=val_dataset,
                device=DEVICE,
                output_dir=visual_dir / f"epoch_{epoch:03d}",
                epoch=epoch,
                num_samples=NUM_DEBUG_IMAGES,
                num_classes=NUM_CLASSES,
                prefix="val",
            )

        if epoch % SAVE_EVERY == 0:
            save_model_checkpoint(
                path=checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_dice=best_val_dice,
                value_mapping=value_mapping,
            )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            no_improve = 0

            save_model_checkpoint(
                path=checkpoint_dir / "model_best_dice.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_dice=best_val_dice,
                value_mapping=value_mapping,
            )

            print(f"Nuevo mejor modelo guardado. Val Dice = {best_val_dice:.4f}")

        else:
            no_improve += 1

        if EARLY_STOPPING_PATIENCE > 0 and no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping: {EARLY_STOPPING_PATIENCE} épocas sin mejorar.")
            break

    print("\nGuardando modelo final...")

    save_model_checkpoint(
        path=checkpoint_dir / "model_final.pth",
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        best_val_dice=best_val_dice,
        value_mapping=value_mapping,
    )

    print("\n" + "=" * 70)
    print("EVALUACIÓN EN TEST CON MEJOR MODELO")
    print("=" * 70)

    best_checkpoint_path = checkpoint_dir / "model_best_dice.pth"

    checkpoint = load_model_checkpoint(
        best_checkpoint_path,
        model,
        DEVICE,
    )

    print(f"Mejor época guardada: {checkpoint['epoch']}")
    print(f"Mejor Val Dice: {checkpoint['best_val_dice']:.4f}")

    test_metrics = validate_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        desc="Test",
    )

    test_df = pd.DataFrame([test_metrics])
    test_df.to_csv(log_dir / "test_metrics.csv", index=False)

    print("\nMétricas en test:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nGuardando visualizaciones de test...")

    save_visualizations(
        model=model,
        dataset=test_dataset,
        device=DEVICE,
        output_dir=visual_dir / "test_best_model",
        epoch=checkpoint["epoch"],
        num_samples=NUM_DEBUG_IMAGES,
        num_classes=NUM_CLASSES,
        prefix="test",
    )

    print("\n" + "=" * 70)
    print("PROCESO TERMINADO")
    print("=" * 70)
    print(f"Resultados guardados en: {OUTPUT_DIR}")
    print(f"Mejor modelo: {checkpoint_dir / 'model_best_dice.pth'}")
    print(f"Historial: {log_dir / 'history.csv'}")
    print(f"Métricas test: {log_dir / 'test_metrics.csv'}")
    print(f"Visualizaciones: {visual_dir}")
    print("=" * 70)


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()

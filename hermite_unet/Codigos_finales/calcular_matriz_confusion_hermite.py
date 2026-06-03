# PROYECTO FINAL - APRENDIZAJE DE MÁQUINA PARA VC
# Matriz de confusión para modelo U-Net con Hermite/SRF
#
#
# Autores:
 # Oscar Eduardo Morales Toledo
 # Katya Verónica Fuentes Sánchez
 # Adrián Jesús Maldonado Oclica

from __future__ import annotations

import re
import random
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
# CONFIGURACIÓN
# ============================================================

CHECKPOINT_PATH = r"C:\Users\oscar\OneDrive\Documents\Maestria\2doSemestre\Aprendizaje Automatico para Vision Computacional\ProyectoFinal\segmentacion_hermite_preprocesada_dice_resultados\checkpoints\model_best_dice.pth"

# Evalúa "test" o "val"
EVAL_SPLIT = "test"

OUTPUT_DIR = Path(
    r"C:\Users\oscar\OneDrive\Documents\Maestria\2doSemestre\Aprendizaje Automatico para Vision Computacional\ProyectoFinal\segmentacion_hermite_preprocesada_dice_resultados\matriz_confusion_hermite"
)

BATCH_SIZE = 1
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

# Nombres para mostrar en la matriz.
CLASS_NAMES = [
    "Fondo",
    "Clase 1",
    "Clase 2",
    "Clase 3",
]

SEED = 42


# ============================================================
# SEMILLA Y PIL
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

    print(f"Imágenes: {len(image_files)}")
    print(f"Máscaras: {len(mask_files)}")
    print(f"Pares válidos: {len(pairs)}")

    if len(pairs) == 0:
        raise ValueError(
            "No se encontraron pares imagen-máscara. "
            "Revisa que tengan el mismo nombre base."
        )

    return pairs


# ============================================================
# PREPROCESAMIENTO Y MÁSCARAS
# ============================================================

def read_image_float(image_path: str | Path) -> np.ndarray:
    img = Image.open(image_path)
    arr = np.array(img).astype(np.float32)

    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)

    return arr


def read_mask_raw(mask_path: str | Path) -> np.ndarray:
    mask = Image.open(mask_path)
    mask = mask.convert("L")
    return np.array(mask)


def preprocess_image_percentile(
    image_np: np.ndarray,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    use_zscore: bool = False,
) -> np.ndarray:
    image_np = image_np.astype(np.float32)

    finite = np.isfinite(image_np)

    if not finite.any():
        return np.zeros_like(image_np, dtype=np.float32)

    valid_values = image_np[finite]

    p_low = np.percentile(valid_values, percentile_low)
    p_high = np.percentile(valid_values, percentile_high)

    if abs(p_high - p_low) < 1e-8:
        image_np = np.zeros_like(image_np, dtype=np.float32)
    else:
        image_np = np.clip(image_np, p_low, p_high)
        image_np = (image_np - p_low) / (p_high - p_low + 1e-8)

    image_np = np.clip(image_np, 0.0, 1.0).astype(np.float32)

    if use_zscore:
        valid = image_np != 0
        if valid.any():
            mu = float(image_np[valid].mean())
            sigma = float(image_np[valid].std())
        else:
            mu = float(image_np.mean())
            sigma = float(image_np.std())

        image_np = (image_np - mu) / (sigma + 1e-8)

    return image_np.astype(np.float32)


def apply_mask_mapping(mask_np: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(mask_np, dtype=np.int64)

    for old_value, new_value in mapping.items():
        out[mask_np == old_value] = new_value

    return out


class SegmentationDataset(Dataset):
    def __init__(
        self,
        pairs,
        value_mapping: dict[int, int],
        image_size: int = 512,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        use_zscore: bool = False,
    ):
        self.pairs = pairs
        self.value_mapping = value_mapping
        self.image_size = int(image_size)
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)
        self.use_zscore = bool(use_zscore)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]

        image_np = read_image_float(item["image"])
        image_np = preprocess_image_percentile(
            image_np,
            percentile_low=self.percentile_low,
            percentile_high=self.percentile_high,
            use_zscore=self.use_zscore,
        )

        mask_np = read_mask_raw(item["mask"])

        image_u8 = (np.clip(image_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        image_pil = Image.fromarray(image_u8, mode="L")
        mask_pil = Image.fromarray(mask_np.astype(np.uint8), mode="L")

        image_pil = image_pil.resize(
            (self.image_size, self.image_size),
            resample=RESAMPLE_BILINEAR,
        )

        mask_pil = mask_pil.resize(
            (self.image_size, self.image_size),
            resample=RESAMPLE_NEAREST,
        )

        image_np = np.array(image_pil).astype(np.float32) / 255.0
        mask_np = np.array(mask_pil)

        mask_np = apply_mask_mapping(mask_np, self.value_mapping)

        image_tensor = torch.from_numpy(image_np).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_np).long()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "name": item["name"],
        }


# ============================================================
# MODELO HERMITE/SRF U-NET
# ============================================================

@dataclass
class HermiteBasisConfig:
    kernel_size: int = 7
    max_order: int = 5
    scales: tuple[float, ...] = (1.5,)
    include_orders: tuple[int, ...] | None = None
    normalize: bool = True
    zero_mean_except_order0: bool = True


def hermite_polynomial(n: int, x: torch.Tensor) -> torch.Tensor:
    if n == 0:
        return torch.ones_like(x)

    if n == 1:
        return 2.0 * x

    h0 = torch.ones_like(x)
    h1 = 2.0 * x

    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0 = h1
        h1 = h2

    return h1


def make_hermite_2d_basis(
    config: HermiteBasisConfig,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    k = int(config.kernel_size)

    if k % 2 == 0:
        raise ValueError("kernel_size debe ser impar.")

    half = k // 2

    coords = torch.linspace(
        -half,
        half,
        steps=k,
        dtype=torch.float32,
        device=device,
    )

    basis_filters = []

    for scale in config.scales:
        scale = float(scale)

        x = coords / scale
        y = coords / scale

        gaussian_x = torch.exp(-0.5 * x**2)
        gaussian_y = torch.exp(-0.5 * y**2)

        for ny in range(config.max_order + 1):
            for nx in range(config.max_order + 1):
                total_order = nx + ny

                if total_order > config.max_order:
                    continue

                if config.include_orders is not None:
                    if total_order not in config.include_orders:
                        continue

                hx = hermite_polynomial(nx, x) * gaussian_x
                hy = hermite_polynomial(ny, y) * gaussian_y

                filt = torch.outer(hy, hx)

                if config.zero_mean_except_order0 and total_order > 0:
                    filt = filt - filt.mean()

                if config.normalize:
                    norm = torch.sqrt(torch.sum(filt**2)) + 1e-8
                    filt = filt / norm

                basis_filters.append(filt)

    if len(basis_filters) == 0:
        raise ValueError("No se generó ningún filtro Hermite.")

    basis = torch.stack(basis_filters, dim=0)
    basis = basis.unsqueeze(1)

    return basis


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


class SRFConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        basis_config: HermiteBasisConfig,
        bias: bool = False,
    ):
        super().__init__()

        basis = make_hermite_2d_basis(
            basis_config,
            device="cpu",
        )

        self.register_buffer("basis", basis)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_basis = int(basis.shape[0])
        self.kernel_size = int(basis.shape[-1])
        self.padding = self.kernel_size // 2

        self.mix = nn.Conv2d(
            in_channels * self.num_basis,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x):
        basis = self.basis.to(
            device=x.device,
            dtype=x.dtype,
        )

        weight = basis.repeat(self.in_channels, 1, 1, 1)

        y = F.conv2d(
            x,
            weight,
            bias=None,
            stride=1,
            padding=self.padding,
            groups=self.in_channels,
        )

        y = self.mix(y)

        return y


class SRFBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        basis_config: HermiteBasisConfig,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = [
            SRFConv2d(
                in_channels,
                out_channels,
                basis_config=basis_config,
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


class DoubleSRFConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        basis_config: HermiteBasisConfig,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            SRFBNReLU(
                in_channels,
                out_channels,
                basis_config=basis_config,
                use_batchnorm=use_batchnorm,
                dropout=dropout,
            ),
            SRFBNReLU(
                out_channels,
                out_channels,
                basis_config=basis_config,
                use_batchnorm=use_batchnorm,
                dropout=dropout,
            ),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = block

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        block_factory,
        bilinear: bool = False,
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

        self.conv = block_factory(
            up_channels + skip_channels,
            out_channels,
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


class HermiteSRFUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 24,
        depth: int = 4,
        basis_config: HermiteBasisConfig | None = None,
        encoder_block: str = "srf",
        decoder_block: str = "conv",
        srf_stages: list[int] | None = None,
        bilinear: bool = False,
        use_batchnorm: bool = True,
        dropout: float = 0.05,
    ):
        super().__init__()

        if depth < 2:
            raise ValueError("depth debe ser >= 2.")

        if basis_config is None:
            basis_config = HermiteBasisConfig()

        self.depth = int(depth)
        self.encoder_block = encoder_block
        self.decoder_block = decoder_block

        if srf_stages is None:
            self.srf_stages = set(range(depth))
        else:
            self.srf_stages = set(int(s) for s in srf_stages)

        channels = [
            base_channels * (2**i)
            for i in range(depth)
        ]

        def make_block(
            in_ch: int,
            out_ch: int,
            block_type: str,
            stage: int | None = None,
        ):
            use_srf = (
                block_type == "srf"
                and (
                    stage is None
                    or stage in self.srf_stages
                )
            )

            if use_srf:
                return DoubleSRFConv(
                    in_ch,
                    out_ch,
                    basis_config=basis_config,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                )

            return DoubleConv(
                in_ch,
                out_ch,
                use_batchnorm=use_batchnorm,
                dropout=dropout,
            )

        self.inc = make_block(
            in_channels,
            channels[0],
            encoder_block,
            stage=0,
        )

        self.downs = nn.ModuleList()

        for i in range(1, depth):
            self.downs.append(
                Down(
                    make_block(
                        channels[i - 1],
                        channels[i],
                        encoder_block,
                        stage=i,
                    )
                )
            )

        self.ups = nn.ModuleList()
        rev_channels = channels[::-1]

        for i in range(depth - 1):
            in_ch = rev_channels[i]
            skip_ch = rev_channels[i + 1]
            out_ch = rev_channels[i + 1]

            def block_factory(
                a,
                b,
                block_type=decoder_block,
            ):
                if block_type == "srf":
                    return DoubleSRFConv(
                        a,
                        b,
                        basis_config=basis_config,
                        use_batchnorm=use_batchnorm,
                        dropout=dropout,
                    )

                return DoubleConv(
                    a,
                    b,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                )

            self.ups.append(
                Up(
                    in_ch,
                    skip_ch,
                    out_ch,
                    block_factory,
                    bilinear=bilinear,
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
# 7. INFERENCIA DE CONFIGURACIÓN DESDE CHECKPOINT
# ============================================================

def infer_model_config_from_checkpoint(checkpoint):
    sd = checkpoint["model_state_dict"]

    num_classes = int(checkpoint.get("num_classes", sd["outc.weight"].shape[0]))
    image_size = int(checkpoint.get("image_size", 512))

    # outc.weight: [num_classes, base_channels, 1, 1]
    base_channels = int(sd["outc.weight"].shape[1])

    down_indices = []
    for key in sd.keys():
        m = re.match(r"downs\.(\d+)\.", key)
        if m:
            down_indices.append(int(m.group(1)))

    depth = max(down_indices) + 2 if down_indices else 4

    preprocessing = checkpoint.get("preprocessing", {})
    hermite = checkpoint.get("hermite", {})

    percentile_low = float(preprocessing.get("percentile_low", 1.0))
    percentile_high = float(preprocessing.get("percentile_high", 99.0))
    use_zscore = bool(preprocessing.get("use_zscore", False))

    basis_config = HermiteBasisConfig(
        kernel_size=int(hermite.get("kernel_size", 7)),
        max_order=int(hermite.get("max_order", 5)),
        scales=tuple(float(s) for s in hermite.get("scales", (1.5,))),
        include_orders=None,
        normalize=True,
        zero_mean_except_order0=True,
    )

    srf_stages = hermite.get("srf_stages", [0, 1, 2])

    value_mapping = checkpoint.get(
        "value_mapping",
        {i: i for i in range(num_classes)},
    )

    # Asegurar llaves enteras, por si se cargan como string en algún caso.
    value_mapping = {int(k): int(v) for k, v in value_mapping.items()}

    return {
        "num_classes": num_classes,
        "image_size": image_size,
        "base_channels": base_channels,
        "depth": depth,
        "percentile_low": percentile_low,
        "percentile_high": percentile_high,
        "use_zscore": use_zscore,
        "basis_config": basis_config,
        "srf_stages": srf_stages,
        "value_mapping": value_mapping,
    }


# ============================================================
# 8. MATRIZ DE CONFUSIÓN
# ============================================================

@torch.no_grad()
def compute_confusion_matrix(model, loader, num_classes, device):
    model.eval()

    conf = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
    )

    for batch in tqdm(loader, desc="Calculando matriz de confusión"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        true = masks.view(-1)
        pred = preds.view(-1)

        valid = (true >= 0) & (true < num_classes)
        true = true[valid]
        pred = pred[valid]

        indices = true * num_classes + pred
        bincount = torch.bincount(
            indices,
            minlength=num_classes * num_classes,
        )

        conf += bincount.cpu().reshape(num_classes, num_classes)

    return conf.numpy()


def normalize_confusion_by_rows(conf):
    row_sum = conf.sum(axis=1, keepdims=True)
    conf_norm = conf / np.maximum(row_sum, 1)
    return conf_norm


def save_confusion_csv(conf, conf_norm, class_names, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.DataFrame(
        conf,
        index=[f"GT_{name}" for name in class_names],
        columns=[f"Pred_{name}" for name in class_names],
    )

    df_norm = pd.DataFrame(
        conf_norm,
        index=[f"GT_{name}" for name in class_names],
        columns=[f"Pred_{name}" for name in class_names],
    )

    df_raw.to_csv(output_dir / "confusion_matrix_raw.csv", encoding="utf-8-sig")
    df_norm.to_csv(output_dir / "confusion_matrix_normalized_by_gt.csv", encoding="utf-8-sig")

    print("\nMatriz de confusión cruda:")
    print(df_raw)

    print("\nMatriz de confusión normalizada por clase real:")
    print(df_norm.round(4))


def plot_confusion_matrix(
    conf,
    class_names,
    output_path,
    title,
    normalize=False,
):
    if normalize:
        data = normalize_confusion_by_rows(conf)
        fmt = ".2f"
    else:
        data = conf
        fmt = "d"

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(data, cmap="Oranges")

    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Ground truth")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if normalize:
                text = f"{data[i, j]:.2f}"
            else:
                text = f"{int(data[i, j])}"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="black",
                fontsize=16,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close(fig)


def metrics_from_confusion(conf, class_names):
    rows = []

    for c, name in enumerate(class_names):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        tn = conf.sum() - tp - fp - fn

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        dice = 2 * tp / max(2 * tp + fp + fn, 1)

        rows.append(
            {
                "class_id": c,
                "class_name": name,
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "precision": float(precision),
                "recall": float(recall),
                "iou": float(iou),
                "dice": float(dice),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 9. MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("MATRIZ DE CONFUSIÓN - MODELO HERMITE/SRF YA ENTRENADO")
    print("=" * 70)
    print("DEVICE:", DEVICE)
    print("CHECKPOINT:", CHECKPOINT_PATH)
    print("EVAL_SPLIT:", EVAL_SPLIT)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("=" * 70 + "\n")

    checkpoint_path = Path(CHECKPOINT_PATH)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No encontré el checkpoint:\n{checkpoint_path}\n\n"
            "Cambia CHECKPOINT_PATH a la ruta correcta de model_best_dice.pth."
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    config = infer_model_config_from_checkpoint(checkpoint)

    num_classes = config["num_classes"]

    class_names = CLASS_NAMES

    if len(class_names) != num_classes:
        class_names = [f"Clase {i}" for i in range(num_classes)]

    print("Configuración inferida del checkpoint:")
    print("  época:", checkpoint.get("epoch", "NA"))
    print("  best_val_dice:", checkpoint.get("best_val_dice", "NA"))
    print("  num_classes:", num_classes)
    print("  image_size:", config["image_size"])
    print("  base_channels:", config["base_channels"])
    print("  depth:", config["depth"])
    print("  percentile_low:", config["percentile_low"])
    print("  percentile_high:", config["percentile_high"])
    print("  use_zscore:", config["use_zscore"])
    print("  hermite kernel_size:", config["basis_config"].kernel_size)
    print("  hermite max_order:", config["basis_config"].max_order)
    print("  hermite scales:", config["basis_config"].scales)
    print("  srf_stages:", config["srf_stages"])
    print("  value_mapping:", config["value_mapping"])

    if EVAL_SPLIT.lower() == "test":
        images_dir = TEST_IMAGES
        masks_dir = TEST_MASKS
    elif EVAL_SPLIT.lower() == "val":
        images_dir = VAL_IMAGES
        masks_dir = VAL_MASKS
    else:
        raise ValueError("EVAL_SPLIT debe ser 'test' o 'val'.")

    print(f"\nEmparejando {EVAL_SPLIT}...")
    pairs = make_pairs(images_dir, masks_dir)

    dataset = SegmentationDataset(
        pairs,
        value_mapping=config["value_mapping"],
        image_size=config["image_size"],
        percentile_low=config["percentile_low"],
        percentile_high=config["percentile_high"],
        use_zscore=config["use_zscore"],
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE == "cuda",
    )

    print("\nReconstruyendo modelo...")

    model = HermiteSRFUNet(
        in_channels=1,
        num_classes=num_classes,
        base_channels=config["base_channels"],
        depth=config["depth"],
        basis_config=config["basis_config"],
        encoder_block="srf",
        decoder_block="conv",
        srf_stages=config["srf_stages"],
        bilinear=False,
        use_batchnorm=True,
        dropout=0.05,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\nCalculando matriz de confusión...")

    conf = compute_confusion_matrix(
        model=model,
        loader=loader,
        num_classes=num_classes,
        device=DEVICE,
    )

    conf_norm = normalize_confusion_by_rows(conf)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_confusion_csv(
        conf=conf,
        conf_norm=conf_norm,
        class_names=class_names,
        output_dir=OUTPUT_DIR,
    )

    plot_confusion_matrix(
        conf=conf,
        class_names=class_names,
        output_path=OUTPUT_DIR / f"confusion_matrix_raw_{EVAL_SPLIT}.png",
        title=f"Matriz de confusión cruda ({EVAL_SPLIT})",
        normalize=False,
    )

    plot_confusion_matrix(
        conf=conf,
        class_names=class_names,
        output_path=OUTPUT_DIR / f"confusion_matrix_normalized_{EVAL_SPLIT}.png",
        title=f"U-Net Hermite",
        normalize=True,
    )

    metrics_df = metrics_from_confusion(
        conf=conf,
        class_names=class_names,
    )

    metrics_df.to_csv(
        OUTPUT_DIR / f"metrics_from_confusion_{EVAL_SPLIT}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nMétricas por clase derivadas de la matriz:")
    print(metrics_df.round(4))

    print("\n" + "=" * 70)
    print("LISTO")
    print("=" * 70)
    print("Archivos guardados en:")
    print(OUTPUT_DIR)
    print("\nSe generaron:")
    print("  confusion_matrix_raw.csv")
    print("  confusion_matrix_normalized_by_gt.csv")
    print(f"  confusion_matrix_raw_{EVAL_SPLIT}.png")
    print(f"  confusion_matrix_normalized_{EVAL_SPLIT}.png")
    print(f"  metrics_from_confusion_{EVAL_SPLIT}.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()

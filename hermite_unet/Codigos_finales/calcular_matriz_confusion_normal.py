# PROYECTO FINAL - APRENDIZAJE DE MÁQUINA PARA VC
# Matriz de confusión para modelo U-Net
#
#
# Autores:
 # Oscar Eduardo Morales Toledo
 # Katya Verónica Fuentes Sánchez
 # Adrián Jesús Maldonado Oclica

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# RUTAS
# ============================================================

VAL_IMAGES = r"C:\Users\oscar\OneDrive\Documents\Maestria\2doSemestre\Aprendizaje Automatico para Vision Computacional\ProyectoFinal\CNNs_Hermite_SRF\hermite_srf_unet\data\val\images"
VAL_MASKS  = r"C:\Users\oscar\OneDrive\Documents\Maestria\2doSemestre\Aprendizaje Automatico para Vision Computacional\ProyectoFinal\CNNs_Hermite_SRF\hermite_srf_unet\data\val\masks"

TEST_IMAGES = r"C:\Users\oscar\OneDrive\Documents\Maestria\2doSemestre\Aprendizaje Automatico para Vision Computacional\ProyectoFinal\CNNs_Hermite_SRF\hermite_srf_unet\data\test\images"
TEST_MASKS  = r"C:\Users\oscar\OneDrive\Documents\Maestria\2doSemestre\Aprendizaje Automatico para Vision Computacional\ProyectoFinal\CNNs_Hermite_SRF\hermite_srf_unet\data\test\masks"


# ============================================================
# CHECKPOINT Y SALIDA
# ============================================================

CHECKPOINT_PATH = r"C:\Users\oscar\OneDrive\Documents\Maestria\2doSemestre\Aprendizaje Automatico para Vision Computacional\ProyectoFinal\segmentacion_unet_normal_preprocesada_dice_resultados\checkpoints\model_best_dice.pth"

# Cambia a "val" si quieres la matriz sobre validación
EVAL_SPLIT = "test"

OUTPUT_DIR = Path(
    r"C:\Users\oscar\OneDrive\Documents\Maestria\2doSemestre\Aprendizaje Automatico para Vision Computacional\ProyectoFinal\segmentacion_unet_normal_preprocesada_dice_resultados\matriz_confusion_unet_normal"
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_WORKERS = 0
SEED = 42

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


# ============================================================
# SEMILLA
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# ============================================================
# COMPATIBILIDAD PIL
# ============================================================

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR
    RESAMPLE_NEAREST = Image.NEAREST


# ============================================================
# UTILIDADES DE ARCHIVOS
# ============================================================

def list_files(folder: str | Path):
    folder = Path(folder)

    files = []

    for ext in IMAGE_EXTENSIONS:
        files.extend(list(folder.glob(f"*{ext}")))
        files.extend(list(folder.glob(f"*{ext.upper()}")))

    return sorted(list(set(files)))


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
# PREPROCESAMIENTO DE MÁSCARAS
# ============================================================

def apply_mask_mapping(mask_np: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(mask_np, dtype=np.int64)

    for old_value, new_value in mapping.items():
        out[mask_np == old_value] = new_value

    return out


# ============================================================
# DATASET
# ============================================================

class SegmentationDatasetForEval(Dataset):
    def __init__(
        self,
        pairs,
        image_size: int,
        value_mapping: dict[int, int],
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        use_zscore: bool = False,
    ):
        self.pairs = pairs
        self.image_size = image_size
        self.value_mapping = value_mapping
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.use_zscore = use_zscore

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]

        # Imagen monocanal
        image = Image.open(item["image"])
        mask = Image.open(item["mask"]).convert("L")

        image_np = np.array(image).astype(np.float32)
        mask_np = np.array(mask)

        # Resize
        image_pil = Image.fromarray(image_np)
        mask_pil = Image.fromarray(mask_np.astype(np.uint8))

        image_pil = image_pil.resize(
            (self.image_size, self.image_size),
            resample=RESAMPLE_BILINEAR,
        )

        mask_pil = mask_pil.resize(
            (self.image_size, self.image_size),
            resample=RESAMPLE_NEAREST,
        )

        image_np = np.array(image_pil).astype(np.float32)
        mask_np = np.array(mask_pil).astype(np.uint8)

        # Preprocesamiento por percentiles
        p_low = np.percentile(image_np, self.percentile_low)
        p_high = np.percentile(image_np, self.percentile_high)

        image_np = np.clip(image_np, p_low, p_high)
        image_np = (image_np - p_low) / (p_high - p_low + 1e-8)

        if self.use_zscore:
            mu = image_np.mean()
            sigma = image_np.std()
            image_np = (image_np - mu) / (sigma + 1e-8)

        image_np = np.clip(image_np, 0.0, 1.0)

        # Mapeo de clases de máscara
        mask_np = apply_mask_mapping(mask_np, self.value_mapping)

        # Tensor monocanal
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_np).long()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "name": item["name"],
        }


# ============================================================
# ARQUITECTURA U-NET NORMAL
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
    def __init__(self, block: nn.Module):
        super().__init__()

        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            block,
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
                    DoubleConv(
                        channels[i - 1],
                        channels[i],
                        use_batchnorm=use_batchnorm,
                        dropout=dropout,
                    )
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
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
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
# MATRIZ DE CONFUSIÓN
# ============================================================

@torch.no_grad()
def compute_confusion_matrix(model, loader, num_classes, device):
    model.eval()

    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in tqdm(loader, desc="Evaluando"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        y_true = masks.cpu().numpy().reshape(-1)
        y_pred = preds.cpu().numpy().reshape(-1)

        for t, p in zip(y_true, y_pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                conf[t, p] += 1

    return conf


def normalize_confusion_matrix_by_gt(conf):
    conf = conf.astype(np.float64)
    row_sums = conf.sum(axis=1, keepdims=True)
    conf_norm = conf / (row_sums + 1e-8)
    return conf_norm


def compute_metrics_from_confusion(conf):
    num_classes = conf.shape[0]
    metrics = []

    for c in range(num_classes):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        tn = conf.sum() - tp - fp - fn

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

        metrics.append(
            {
                "class_id": c,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": precision,
                "recall": recall,
                "iou": iou,
                "dice": dice,
            }
        )

    df = pd.DataFrame(metrics)

    # Macro sin fondo
    if num_classes > 1:
        df_fg = df[df["class_id"] != 0]
        macro_row = {
            "class_id": "macro_no_bg",
            "tp": df_fg["tp"].sum(),
            "fp": df_fg["fp"].sum(),
            "fn": df_fg["fn"].sum(),
            "tn": df_fg["tn"].sum(),
            "precision": df_fg["precision"].mean(),
            "recall": df_fg["recall"].mean(),
            "iou": df_fg["iou"].mean(),
            "dice": df_fg["dice"].mean(),
        }
        df = pd.concat([df, pd.DataFrame([macro_row])], ignore_index=True)

    return df


# ============================================================
# 11. GRAFICAR MATRIZ
# ============================================================

def save_confusion_matrix_plot(conf, class_names, out_path, title):
    plt.figure(figsize=(7, 6))
    plt.imshow(conf, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if np.issubdtype(conf.dtype, np.floating) else "d"
    thresh = conf.max() / 2.0 if conf.size > 0 else 0.0

    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            plt.text(
                j,
                i,
                format(conf[i, j], fmt),
                ha="center",
                va="center",
                color="black",
                fontsize=16,
            )

    plt.ylabel("Clase real")
    plt.xlabel("Clase predicha")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("MATRIZ DE CONFUSIÓN - U-NET NORMAL YA ENTRENADA")
    print("=" * 70)
    print("DEVICE:", DEVICE)
    print("CHECKPOINT:", CHECKPOINT_PATH)
    print("EVAL_SPLIT:", EVAL_SPLIT)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("=" * 70 + "\n")

    checkpoint_path = Path(CHECKPOINT_PATH)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe el checkpoint:\n{checkpoint_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Inferencia de configuración desde checkpoint
    epoch = checkpoint.get("epoch", None)
    best_val_dice = checkpoint.get("best_val_dice", None)
    num_classes = checkpoint.get("num_classes", 4)
    image_size = checkpoint.get("image_size", 512)
    base_channels = checkpoint.get("base_channels", 24)
    depth = checkpoint.get("depth", 4)
    percentile_low = checkpoint.get("percentile_low", 1.0)
    percentile_high = checkpoint.get("percentile_high", 99.0)
    use_zscore = checkpoint.get("use_zscore", False)
    value_mapping = checkpoint.get("value_mapping", {0: 0, 1: 1, 2: 2, 3: 3})

    print("Configuración inferida del checkpoint:")
    print(f"  época: {epoch}")
    print(f"  best_val_dice: {best_val_dice}")
    print(f"  num_classes: {num_classes}")
    print(f"  image_size: {image_size}")
    print(f"  base_channels: {base_channels}")
    print(f"  depth: {depth}")
    print(f"  percentile_low: {percentile_low}")
    print(f"  percentile_high: {percentile_high}")
    print(f"  use_zscore: {use_zscore}")
    print(f"  value_mapping: {value_mapping}")

    # Selección de split
    if EVAL_SPLIT.lower() == "val":
        images_dir = VAL_IMAGES
        masks_dir = VAL_MASKS
    elif EVAL_SPLIT.lower() == "test":
        images_dir = TEST_IMAGES
        masks_dir = TEST_MASKS
    else:
        raise ValueError("EVAL_SPLIT debe ser 'val' o 'test'.")

    print(f"\nEmparejando {EVAL_SPLIT}...")
    pairs = make_pairs(images_dir, masks_dir)

    dataset = SegmentationDatasetForEval(
        pairs=pairs,
        image_size=image_size,
        value_mapping=value_mapping,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        use_zscore=use_zscore,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    print("\nReconstruyendo modelo...")

    model = UNetNormal(
        in_channels=1,
        num_classes=num_classes,
        base_channels=base_channels,
        depth=depth,
        bilinear=False,
        use_batchnorm=True,
        dropout=0.05,
    ).to(DEVICE)

    state_dict = checkpoint["model_state_dict"]

    # Por si alguna vez hubiera prefijo "module."
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)

    print("\nCalculando matriz de confusión...")
    conf = compute_confusion_matrix(
        model=model,
        loader=loader,
        num_classes=num_classes,
        device=DEVICE,
    )

    conf_norm = normalize_confusion_matrix_by_gt(conf)
    metrics_df = compute_metrics_from_confusion(conf)

    class_names = [f"Clase {i}" for i in range(num_classes)]
    class_names[0] = "Fondo"

    # Guardar CSVs
    conf_df = pd.DataFrame(conf, index=class_names, columns=class_names)
    conf_norm_df = pd.DataFrame(conf_norm, index=class_names, columns=class_names)

    raw_csv_path = OUTPUT_DIR / f"confusion_matrix_raw_{EVAL_SPLIT}.csv"
    norm_csv_path = OUTPUT_DIR / f"confusion_matrix_normalized_by_gt_{EVAL_SPLIT}.csv"
    metrics_csv_path = OUTPUT_DIR / f"metrics_from_confusion_{EVAL_SPLIT}.csv"

    conf_df.to_csv(raw_csv_path)
    conf_norm_df.to_csv(norm_csv_path)
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Guardar figuras
    raw_png_path = OUTPUT_DIR / f"confusion_matrix_raw_{EVAL_SPLIT}.png"
    norm_png_path = OUTPUT_DIR / f"confusion_matrix_normalized_{EVAL_SPLIT}.png"

    save_confusion_matrix_plot(
        conf=conf,
        class_names=class_names,
        out_path=raw_png_path,
        title=f"Matriz de confusión (raw) - {EVAL_SPLIT}",
    )

    save_confusion_matrix_plot(
        conf=conf_norm,
        class_names=class_names,
        out_path=norm_png_path,
        title=f"U-Net",
    )

    print("\nMatriz de confusión cruda:")
    print(conf_df)

    print("\nMatriz de confusión normalizada por GT:")
    print(conf_norm_df)

    print("\nMétricas por clase:")
    print(metrics_df)

    print("\n" + "=" * 70)
    print("PROCESO TERMINADO")
    print("=" * 70)
    print(f"Raw CSV: {raw_csv_path}")
    print(f"Norm CSV: {norm_csv_path}")
    print(f"Métricas CSV: {metrics_csv_path}")
    print(f"Raw PNG: {raw_png_path}")
    print(f"Norm PNG: {norm_png_path}")
    print("=" * 70)


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()
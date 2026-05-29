from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from _bootstrap import ROOT  # noqa
from src.data.dataset import IMG_EXTS, list_images
from src.models.unet_srf import build_model_from_config
from src.utils.checkpoints import load_model_weights
from src.utils.config import load_class_metadata, load_config
from src.utils.metrics import logits_to_pred
from src.utils.visualization import save_mask_png, overlay_mask


def parse_args():
    p = argparse.ArgumentParser(description="Predice todas las imágenes de una carpeta.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", default="predicted_images")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def preprocess(path: Path, cfg: dict):
    dcfg = cfg["data"]
    img = Image.open(path).convert("RGB" if dcfg.get("image_mode", "rgb") == "rgb" else "L")
    original = np.array(img.convert("RGB")) / 255.0
    h, w = dcfg.get("image_size", [256, 256])
    img_resized = img.resize((w, h), Image.BILINEAR)
    arr = np.array(img_resized).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    norm = dcfg.get("normalize", {})
    mean = torch.tensor(norm.get("mean", [0.485, 0.456, 0.406])[: x.shape[0]], dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(norm.get("std", [0.229, 0.224, 0.225])[: x.shape[0]], dtype=torch.float32).view(-1, 1, 1)
    x = ((x - mean) / std).unsqueeze(0)
    return x, original, img.size


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = args.device
    out_dir = Path(args.output_dir)
    mask_dir = out_dir / "masks"
    overlay_dir = out_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("predict", {}).get("save_overlay", True):
        overlay_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_config(cfg).to(device)
    load_model_weights(model, args.checkpoint, device=device)
    model.eval()

    mode = cfg["data"].get("segmentation_mode", "binary")
    ncls = int(cfg["data"].get("num_classes", 2))
    _, class_colors = load_class_metadata(cfg)
    threshold = float(cfg["predict"].get("threshold", cfg["train"].get("threshold", 0.5)))

    for path in tqdm(list_images(args.input_dir), desc="predict"):
        x, original, orig_size = preprocess(path, cfg)
        logits = model(x.to(device))
        pred = logits_to_pred(logits, mode=mode, threshold=threshold)[0].cpu().numpy().astype(np.uint8)
        pred_img = Image.fromarray((pred if ncls > 2 else pred * 255).astype(np.uint8))
        pred_img = pred_img.resize(orig_size, Image.NEAREST)
        pred_np = (np.array(pred_img) > 0).astype(np.uint8) if ncls <= 2 else np.array(pred_img).astype(np.uint8)
        save_mask_png(pred_np, mask_dir / f"{path.stem}_pred.png", num_classes=ncls, colors=class_colors)

        if cfg.get("predict", {}).get("save_overlay", True):
            ov = overlay_mask(original, pred_np, num_classes=ncls, colors=class_colors)
            Image.fromarray((ov * 255).astype(np.uint8)).save(overlay_dir / f"{path.stem}_overlay.png")

    print(f"Predicciones guardadas en: {out_dir}")


if __name__ == "__main__":
    main()

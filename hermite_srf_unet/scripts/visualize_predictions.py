from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from _bootstrap import ROOT  # noqa
from src.data.dataset import make_dataset_from_config
from src.models.unet_srf import build_model_from_config
from src.utils.checkpoints import load_model_weights
from src.utils.config import load_config
from src.utils.metrics import logits_to_pred
from src.utils.visualization import colorize_mask, denormalize, overlay_mask


def parse_args():
    p = argparse.ArgumentParser(description="Visualiza imagen, ground truth, predicción y overlay.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--num-samples", type=int, default=12)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def panel_with_title(array: np.ndarray, title: str, width: int = 256) -> Image.Image:
    if array.dtype != np.uint8:
        array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(array).convert("RGB")
    if img.width != width:
        height = max(1, round(img.height * width / img.width))
        img = img.resize((width, height), Image.Resampling.NEAREST)

    title_h = 28
    panel = Image.new("RGB", (img.width, img.height + title_h), "white")
    panel.paste(img, (0, title_h))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), title, fill=(0, 0, 0))
    return panel


def save_visualization_grid(path: Path, panels: list[Image.Image]) -> None:
    gap = 8
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    canvas.save(path)


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["project"].get("output_dir", "outputs/exp01")) / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = make_dataset_from_config(cfg, args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_model_from_config(cfg).to(args.device)
    load_model_weights(model, args.checkpoint, device=args.device)
    model.eval()

    mode = cfg["data"].get("segmentation_mode", "binary")
    ncls = int(cfg["data"].get("num_classes", 2))
    threshold = float(cfg["predict"].get("threshold", cfg["train"].get("threshold", 0.5)))
    norm = cfg["data"].get("normalize", {})
    mean = norm.get("mean", [0.485, 0.456, 0.406])
    std = norm.get("std", [0.229, 0.224, 0.225])

    for i, batch in enumerate(tqdm(loader, total=min(args.num_samples, len(ds)))):
        if i >= args.num_samples:
            break
        img_t = batch["image"][0]
        image = denormalize(img_t, mean, std)
        mask = batch["mask"]
        if mode == "binary":
            gt = mask[0, 0].numpy().astype(np.uint8)
        else:
            gt = mask[0].numpy().astype(np.uint8)
        logits = model(batch["image"].to(args.device))
        pred = logits_to_pred(logits, mode=mode, threshold=threshold)[0].cpu().numpy().astype(np.uint8)

        name = batch["name"][0]

        print(f"\n{name}")
        print("Modo:", mode)
        print("Número de clases:", ncls)
        print("Logits shape:", tuple(logits.shape))
        print("GT valores únicos:", np.unique(gt))
        print("Pred valores únicos:", np.unique(pred))

        if mode == "multiclass":
            pred_counts = np.bincount(
                pred.flatten(),
                minlength=ncls,
            )

            gt_counts = np.bincount(
                gt.flatten(),
                minlength=ncls,
            )

            print("Píxeles GT por clase:")
            for cls_id, count in enumerate(gt_counts.tolist()):
                print(f"  clase {cls_id}: {count}")

            print("Píxeles predichos por clase:")
            for cls_id, count in enumerate(pred_counts.tolist()):
                print(f"  clase {cls_id}: {count}")

        if mode == "multiclass":
            gt_view = colorize_mask(gt, num_classes=ncls)
            pred_view = colorize_mask(pred, num_classes=ncls)
        else:
            gt_view = np.stack([gt * 255] * 3, axis=-1).astype(np.uint8)
            pred_view = np.stack([pred * 255] * 3, axis=-1).astype(np.uint8)

        overlay = overlay_mask(image, pred, num_classes=ncls)
        panels = [
            panel_with_title(image, "Imagen"),
            panel_with_title(gt_view, "Ground Truth"),
            panel_with_title(pred_view, "Predicción"),
            panel_with_title(overlay, "Overlay"),
        ]
        save_visualization_grid(out_dir / f"{batch['name'][0]}_viz.png", panels)

    print(f"Visualizaciones guardadas en: {out_dir}")


if __name__ == "__main__":
    main()

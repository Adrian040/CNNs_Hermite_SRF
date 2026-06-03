from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from _bootstrap import ROOT  # noqa
from src.data.dataset import make_dataset_from_config
from src.models.unet_srf import build_model_from_config
from src.utils.checkpoints import load_model_weights
from src.utils.config import load_class_metadata, load_config
from src.utils.metrics import logits_to_pred
from src.utils.visualization import class_legend_handles, colorize_mask, denormalize, overlay_mask, require_pyplot


def parse_args():
    p = argparse.ArgumentParser(description="Visualiza imagen, ground truth, prediccion y overlay.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--num-samples", type=int, default=12)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    plt = require_pyplot()
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
    class_names, class_colors = load_class_metadata(cfg)
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

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        axes[0].imshow(image)
        axes[0].set_title("Imagen")
        axes[1].imshow(colorize_mask(gt, num_classes=ncls, colors=class_colors))
        axes[1].set_title("Ground Truth")
        axes[2].imshow(colorize_mask(pred, num_classes=ncls, colors=class_colors))
        axes[2].set_title("Prediccion")
        axes[3].imshow(overlay_mask(image, pred, num_classes=ncls, colors=class_colors))
        axes[3].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        present = sorted(set(np.unique(gt).tolist()) | set(np.unique(pred).tolist()) | {0})
        handles = class_legend_handles(class_names, class_colors, present_classes=present)
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), fontsize=8)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        fig.savefig(out_dir / f"{batch['name'][0]}_viz.png", dpi=200)
        plt.close(fig)

    print(f"Visualizaciones guardadas en: {out_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from _bootstrap import ROOT  # noqa
from src.data.dataset import make_dataset_from_config
from src.models.unet_srf import build_model_from_config
from src.utils.checkpoints import load_model_weights
from src.utils.config import load_config
from src.utils.metrics import logits_to_pred
from src.utils.visualization import (
    denormalize,
    overlay_mask,
    mask_to_class_ids,
    get_discrete_cmap,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualiza imagen, ground truth, predicción y overlay."
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--num-samples", type=int, default=12)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


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
    threshold = float(
        cfg["predict"].get(
            "threshold",
            cfg["train"].get("threshold", 0.5)
        )
    )

    norm_cfg = cfg["data"].get("normalize", {})
    mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
    std = norm_cfg.get("std", [0.229, 0.224, 0.225])

    cmap, mask_norm = get_discrete_cmap(ncls)

    max_samples = min(args.num_samples, len(ds))

    for i, batch in enumerate(tqdm(loader, total=max_samples)):
        if i >= args.num_samples:
            break

        image_tensor = batch["image"][0]
        image = denormalize(image_tensor, mean, std)

        mask_tensor = batch["mask"]

        if mode == "binary":
            gt = mask_tensor[0, 0].cpu().numpy()
        else:
            gt = mask_tensor[0].cpu().numpy()

        gt = mask_to_class_ids(gt, num_classes=ncls)

        logits = model(batch["image"].to(args.device))

        pred = logits_to_pred(
            logits,
            mode=mode,
            threshold=threshold,
        )[0].cpu().numpy()

        pred = mask_to_class_ids(pred, num_classes=ncls)

        name = batch["name"][0]

        print(f"\n{name}")
        print("GT valores únicos:", np.unique(gt))
        print("Pred valores únicos:", np.unique(pred))

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))

        axes[0].imshow(image)
        axes[0].set_title("Imagen")

        axes[1].imshow(gt, cmap=cmap, norm=mask_norm)
        axes[1].set_title("Ground Truth")

        axes[2].imshow(pred, cmap=cmap, norm=mask_norm)
        axes[2].set_title("Predicción")

        axes[3].imshow(
            overlay_mask(
                image,
                pred,
                alpha=0.35,
                num_classes=ncls,
            )
        )
        axes[3].set_title("Overlay predicción")

        for ax in axes:
            ax.axis("off")

        fig.tight_layout()

        save_path = out_dir / f"{name}_viz.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"\nVisualizaciones guardadas en: {out_dir}")


if __name__ == "__main__":
    main()
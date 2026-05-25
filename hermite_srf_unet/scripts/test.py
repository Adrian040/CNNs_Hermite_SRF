from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from _bootstrap import ROOT  # noqa
from src.data.dataset import make_dataset_from_config
from src.models.unet_srf import build_model_from_config
from src.utils.checkpoints import load_model_weights
from src.utils.config import load_config
from src.utils.metrics import MetricAccumulator, logits_to_pred, per_image_metrics
from src.utils.visualization import save_mask_png


def parse_args():
    p = argparse.ArgumentParser(description="Evalúa un checkpoint en test.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-preds", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = args.device
    out_dir = Path(cfg["project"].get("output_dir", "outputs/exp01")) / "test_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = make_dataset_from_config(cfg, "test")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=int(cfg["train"].get("num_workers", 2)))

    model = build_model_from_config(cfg).to(device)
    load_model_weights(model, args.checkpoint, device=device)
    model.eval()

    mode = cfg["data"].get("segmentation_mode", "binary")
    ncls = int(cfg["data"].get("num_classes", 2))
    threshold = float(cfg["predict"].get("threshold", cfg["train"].get("threshold", 0.5)))
    acc = MetricAccumulator(mode=mode, num_classes=ncls)
    rows = []

    pred_dir = out_dir / "predicted_masks"
    if args.save_preds:
        pred_dir.mkdir(exist_ok=True)

    for batch in tqdm(loader, desc="test"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        preds = logits_to_pred(logits, mode=mode, threshold=threshold)
        acc.update(preds, masks)

        pred_np = preds[0].detach().cpu().numpy().astype(np.uint8)
        if mode == "binary":
            target_np = masks[0, 0].detach().cpu().numpy().astype(np.uint8)
        else:
            target_np = masks[0].detach().cpu().numpy().astype(np.uint8)

        row = {"name": batch["name"][0]}
        row.update(per_image_metrics(pred_np, target_np, mode=mode, num_classes=ncls))
        rows.append(row)
        if args.save_preds:
            save_mask_png(pred_np, pred_dir / f"{batch['name'][0]}_pred.png", num_classes=ncls)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_image_metrics.csv", index=False)

    data_based = acc.compute()
    summary = {
        "dice_data_based": data_based["dice"],
        "iou_data_based": data_based["iou"],
        "precision_data_based": data_based["precision"],
        "recall_data_based": data_based["recall"],
        "hausdorff_image_mean": float(df["hausdorff"].mean(skipna=True)),
        "hausdorff95_image_mean": float(df["hausdorff95"].mean(skipna=True)),
        "n_images": len(df),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary_metrics.csv", index=False)

    print("Métricas finales:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Resultados guardados en: {out_dir}")


if __name__ == "__main__":
    main()

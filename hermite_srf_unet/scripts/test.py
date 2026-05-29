from __future__ import annotations

import argparse
import csv
import math
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
from src.utils.metrics import MetricAccumulator, logits_to_pred, per_image_class_metrics, per_image_metrics
from src.utils.visualization import save_mask_png


def parse_args():
    p = argparse.ArgumentParser(description="Evalua un checkpoint en test.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-preds", action="store_true")
    return p.parse_args()


def finite_values(values) -> list[float]:
    out = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(value):
            out.append(value)
    return out


def mean(values) -> float:
    vals = finite_values(values)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def std(values) -> float:
    vals = finite_values(values)
    if len(vals) < 2:
        return float("nan")
    avg = sum(vals) / len(vals)
    return float((sum((v - avg) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_per_class(class_rows: list[dict], data_rows: list[dict]) -> list[dict]:
    rows = []
    metric_names = ("dice", "iou", "precision", "recall", "hausdorff", "hausdorff95")
    for data_row in data_rows:
        class_id = int(data_row["class_id"])
        subset = [row for row in class_rows if int(row["class_id"]) == class_id]
        row = {
            "class_id": class_id,
            "class_name": data_row["class_name"],
            "n_images": int(len(subset)),
            "n_target_present": int(sum(bool(row["target_present"]) for row in subset)),
            "n_pred_present": int(sum(bool(row["pred_present"]) for row in subset)),
            "target_pixels_total": int(sum(int(row["target_pixels"]) for row in subset)),
            "pred_pixels_total": int(sum(int(row["pred_pixels"]) for row in subset)),
        }
        for metric in ("dice", "iou", "precision", "recall"):
            row[f"{metric}_data_based"] = float(data_row[metric])
        for metric in metric_names:
            row[f"{metric}_image_mean"] = mean(row[metric] for row in subset)
            row[f"{metric}_image_std"] = std(row[metric] for row in subset)
        rows.append(row)
    return rows


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
    class_names, class_colors = load_class_metadata(cfg)
    threshold = float(cfg["predict"].get("threshold", cfg["train"].get("threshold", 0.5)))
    acc = MetricAccumulator(mode=mode, num_classes=ncls)
    rows = []
    class_rows = []

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

        name = batch["name"][0]
        row = {"name": name}
        row.update(per_image_metrics(pred_np, target_np, mode=mode, num_classes=ncls, class_names=class_names))
        rows.append(row)

        for class_row in per_image_class_metrics(pred_np, target_np, mode=mode, num_classes=ncls, class_names=class_names):
            class_row = {"name": name, **class_row}
            class_rows.append(class_row)

        if args.save_preds:
            save_mask_png(pred_np, pred_dir / f"{name}_pred.png", num_classes=ncls, colors=class_colors)

    write_csv(out_dir / "per_image_metrics.csv", rows)
    write_csv(out_dir / "per_image_class_metrics.csv", class_rows)

    data_based = acc.compute()
    data_class_rows = acc.compute_per_class(class_names)
    per_class_rows = summarize_per_class(class_rows, data_class_rows)
    write_csv(out_dir / "per_class_metrics.csv", per_class_rows)

    summary = {
        "dice_data_based": data_based["dice"],
        "iou_data_based": data_based["iou"],
        "precision_data_based": data_based["precision"],
        "recall_data_based": data_based["recall"],
        "dice_image_mean": mean(row["dice"] for row in rows),
        "iou_image_mean": mean(row["iou"] for row in rows),
        "precision_image_mean": mean(row["precision"] for row in rows),
        "recall_image_mean": mean(row["recall"] for row in rows),
        "hausdorff_image_mean": mean(row["hausdorff"] for row in rows),
        "hausdorff95_image_mean": mean(row["hausdorff95"] for row in rows),
        "n_images": len(rows),
    }
    for row in per_class_rows:
        class_id = int(row["class_id"])
        for metric in ("dice", "iou", "precision", "recall"):
            summary[f"{metric}_data_based_class_{class_id}"] = row[f"{metric}_data_based"]
        for metric in ("dice", "iou", "precision", "recall", "hausdorff", "hausdorff95"):
            summary[f"{metric}_image_mean_class_{class_id}"] = row[f"{metric}_image_mean"]

    write_csv(out_dir / "summary_metrics.csv", [summary])

    print("Metricas finales:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Resultados guardados en: {out_dir}")


if __name__ == "__main__":
    main()

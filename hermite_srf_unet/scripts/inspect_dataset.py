from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from _bootstrap import ROOT  # noqa
from src.data.dataset import list_images, find_matching_mask, mask_to_class_ids


def parse_args():
    p = argparse.ArgumentParser(description="Inspecciona pares imagen-máscara y guarda un resumen CSV.")
    p.add_argument("--images-dir", default="data/all_data/images")
    p.add_argument("--masks-dir", default="data/all_data/masks")
    p.add_argument("--out", default="data/dataset_summary.csv")
    p.add_argument("--mask-threshold", type=int, default=127)
    p.add_argument("--segmentation-mode", choices=["binary", "multiclass"], default="multiclass")
    p.add_argument("--num-classes", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    images = list_images(args.images_dir)
    rows = []
    for img_path in tqdm(images):
        mask_path = find_matching_mask(img_path, Path(args.masks_dir))
        img = Image.open(img_path)
        mask_img = Image.open(mask_path)
        if args.segmentation_mode == "multiclass":
            mask = mask_to_class_ids(mask_img, num_classes=args.num_classes)
            class_counts = np.bincount(mask.flatten(), minlength=args.num_classes)
            foreground_ratio = float((mask > 0).mean())
            unique_values = ",".join(str(int(v)) for v in np.unique(mask))
        else:
            mask = np.array(mask_img.convert("L"))
            class_counts = None
            foreground_ratio = float((mask > args.mask_threshold).mean())
            unique_values = ",".join(str(int(v)) for v in np.unique(mask))

        row = {
            "image": img_path.name,
            "mask": mask_path.name,
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "mask_mode": mask_img.mode,
            "mask_unique_values": unique_values,
            "foreground_ratio": foreground_ratio,
        }

        if class_counts is not None:
            for class_id, count in enumerate(class_counts.tolist()):
                row[f"class_{class_id}_pixels"] = int(count)

        rows.append({
            **row,
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Resumen guardado en: {out}")


if __name__ == "__main__":
    main()

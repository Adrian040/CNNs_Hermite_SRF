from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from _bootstrap import ROOT  # noqa
from src.data.dataset import list_images, find_matching_mask


def parse_args():
    p = argparse.ArgumentParser(description="Inspecciona pares imagen-máscara y guarda un resumen CSV.")
    p.add_argument("--images-dir", default="data/all_data/images")
    p.add_argument("--masks-dir", default="data/all_data/masks")
    p.add_argument("--out", default="data/dataset_summary.csv")
    p.add_argument("--mask-threshold", type=int, default=127)
    return p.parse_args()


def main():
    args = parse_args()
    images = list_images(args.images_dir)
    rows = []
    for img_path in tqdm(images):
        mask_path = find_matching_mask(img_path, Path(args.masks_dir))
        img = Image.open(img_path)
        mask = np.array(Image.open(mask_path).convert("L"))
        fg = (mask > args.mask_threshold).mean()
        rows.append({
            "image": img_path.name,
            "mask": mask_path.name,
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "mask_unique_values": len(np.unique(mask)),
            "foreground_ratio_binary": float(fg),
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Resumen guardado en: {out}")


if __name__ == "__main__":
    main()

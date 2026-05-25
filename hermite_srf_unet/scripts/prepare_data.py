from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from _bootstrap import ROOT  # noqa
from src.data.dataset import IMG_EXTS, find_matching_mask, list_images


def parse_args():
    p = argparse.ArgumentParser(description="Divide data/all_data en train/val/test.")
    p.add_argument("--root", default="data")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--convert-to-png", action="store_true", help="Convierte imágenes y máscaras a PNG.")
    p.add_argument("--overwrite", action="store_true", help="Borra data/train, data/val y data/test antes de generar la división.")
    return p.parse_args()


def copy_or_convert(src: Path, dst: Path, convert_to_png: bool, is_mask: bool = False) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if convert_to_png:
        dst = dst.with_suffix(".png")
        img = Image.open(src)
        if is_mask:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        img.save(dst)
    else:
        shutil.copy2(src, dst)
    return dst


def main():
    args = parse_args()
    root = Path(args.root)
    images_dir = root / "all_data" / "images"
    masks_dir = root / "all_data" / "masks"

    ratios = [args.train, args.val, args.test]
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Los porcentajes train+val+test deben sumar 1.0")

    for split in ["train", "val", "test"]:
        split_dir = root / split
        if split_dir.exists() and args.overwrite:
            shutil.rmtree(split_dir)
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "masks").mkdir(parents=True, exist_ok=True)

    images = list_images(images_dir)
    if not images:
        raise FileNotFoundError(f"No se encontraron imágenes en {images_dir}")

    pairs = [(img, find_matching_mask(img, masks_dir)) for img in images]
    random.seed(args.seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(round(n * args.train))
    n_val = int(round(n * args.val))
    if n_train + n_val > n:
        n_val = n - n_train
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    for split, split_pairs in splits.items():
        print(f"{split}: {len(split_pairs)} pares")
        for img, mask in tqdm(split_pairs):
            img_dst = root / split / "images" / img.name
            mask_dst = root / split / "masks" / mask.name
            out_img = copy_or_convert(img, img_dst, args.convert_to_png, is_mask=False)
            out_mask = copy_or_convert(mask, mask_dst, args.convert_to_png, is_mask=True)
            # Si se convierte a PNG, fuerza que máscara e imagen conserven el mismo stem.
            if args.convert_to_png and out_mask.stem != out_img.stem:
                fixed = out_mask.with_name(out_img.stem + ".png")
                out_mask.rename(fixed)

    print("División terminada.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Segmenta imágenes originales y máscaras de lesiones usando máscaras de pulmón.

Estructura esperada:
all_data/
  images/       -> imágenes originales .tif
  lung_mask/    -> máscaras de pulmón .tif con valores 0, 1, 2
  masks/        -> máscaras de lesiones .tif con una o varias clases

Salida:
all_data/
  images_lung_seg/
  masks_lung_seg/
  lung_segmentation_log.csv

Uso:
python segmentar_por_lung_mask.py --base_dir all_data

También puedes pasar rutas explícitas:
python segmentar_por_lung_mask.py \
  --images_dir all_data/images \
  --lung_masks_dir all_data/lung_mask \
  --lesion_masks_dir all_data/masks \
  --out_images_dir all_data/images_lung_seg \
  --out_masks_dir all_data/masks_lung_seg
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile as tiff


def unique_values_summary(arr, max_items=50):
    """Resume valores únicos sin modificar la máscara."""
    arr = np.asarray(arr)

    # Caso típico: máscara de etiquetas 2D/3D con enteros.
    if not (arr.ndim >= 3 and arr.shape[-1] in (3, 4)):
        vals = np.unique(arr)
        vals_list = vals[:max_items].tolist()
        suffix = "" if len(vals) <= max_items else f" ... (+{len(vals) - max_items} más)"
        return vals_list, int(len(vals)), suffix

    # Caso alternativo: máscara RGB/RGBA. Se reportan colores únicos.
    flat = arr.reshape(-1, arr.shape[-1])
    vals = np.unique(flat, axis=0)
    vals_list = [tuple(map(int, v)) for v in vals[:max_items]]
    suffix = "" if len(vals) <= max_items else f" ... (+{len(vals) - max_items} más)"
    return vals_list, int(len(vals)), suffix


def broadcast_lung_mask(mask_bin, target_shape, target_name="array"):
    """
    Ajusta la máscara binaria de pulmón para aplicarla sobre imágenes o máscaras
    2D, 3D con canales al final, 3D con canales al inicio, o volúmenes.
    """
    mask_bin = np.asarray(mask_bin, dtype=bool)

    if mask_bin.shape == target_shape:
        return mask_bin

    # Canales al final: H,W,C o Z,H,W,C
    if len(target_shape) == mask_bin.ndim + 1 and target_shape[:mask_bin.ndim] == mask_bin.shape:
        return mask_bin[..., None]

    # Canales al inicio: C,H,W o C,Z,H,W
    if len(target_shape) == mask_bin.ndim + 1 and target_shape[-mask_bin.ndim:] == mask_bin.shape:
        return mask_bin[None, ...]

    # Intento general de broadcasting agregando ejes al final.
    tmp = mask_bin
    while tmp.ndim < len(target_shape):
        tmp = tmp[..., None]

    try:
        np.broadcast_to(tmp, target_shape)
        return tmp
    except ValueError:
        pass

    raise ValueError(
        f"No se puede aplicar la lung_mask con shape {mask_bin.shape} "
        f"sobre {target_name} con shape {target_shape}."
    )


def apply_lung_segmentation(arr, lung_mask_bin, target_name):
    """
    Aplica la máscara de pulmón sin cambiar el dtype ni los valores internos.
    Fuera del pulmón se asigna 0.
    """
    arr = np.asarray(arr)
    lung_broadcast = broadcast_lung_mask(lung_mask_bin, arr.shape, target_name)
    out = np.where(lung_broadcast, arr, 0)
    return out.astype(arr.dtype, copy=False)


def find_matching_file(folder, stem):
    """Busca archivo .tif/.tiff con el mismo stem."""
    candidates = list(folder.glob(stem + ".tif")) + list(folder.glob(stem + ".tiff"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        return None
    raise RuntimeError(f"Hay más de un archivo candidato para {stem} en {folder}: {candidates}")


def main():
    parser = argparse.ArgumentParser(
        description="Segmenta imágenes y máscaras de lesiones usando máscaras de pulmón .tif."
    )

    parser.add_argument("--base_dir", type=str, default="all_data")

    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--lung_masks_dir", type=str, default=None)
    parser.add_argument("--lesion_masks_dir", type=str, default=None)

    parser.add_argument("--out_images_dir", type=str, default=None)
    parser.add_argument("--out_masks_dir", type=str, default=None)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compression", type=str, default=None, help="Ejemplo: zlib. Por defecto no comprime.")
    parser.add_argument("--max_unique_print", type=int, default=50)

    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    images_dir = Path(args.images_dir) if args.images_dir else base_dir / "images"
    lung_masks_dir = Path(args.lung_masks_dir) if args.lung_masks_dir else base_dir / "lung_mask"
    lesion_masks_dir = Path(args.lesion_masks_dir) if args.lesion_masks_dir else base_dir / "masks"

    out_images_dir = Path(args.out_images_dir) if args.out_images_dir else base_dir / "images_lung_seg"
    out_masks_dir = Path(args.out_masks_dir) if args.out_masks_dir else base_dir / "masks_lung_seg"

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(images_dir.glob("*.tif")) + list(images_dir.glob("*.tiff")))

    if len(image_files) == 0:
        raise RuntimeError(f"No encontré archivos .tif/.tiff en {images_dir}")

    log_rows = []
    skipped = 0
    processed = 0

    print(f"Imágenes encontradas: {len(image_files)}")
    print(f"Salida imágenes: {out_images_dir}")
    print(f"Salida máscaras:  {out_masks_dir}")
    print("-" * 80)

    for img_path in image_files:
        stem = img_path.stem

        lung_path = find_matching_file(lung_masks_dir, stem)
        lesion_path = find_matching_file(lesion_masks_dir, stem)

        if lung_path is None or lesion_path is None:
            print(f"[SKIP] {stem}: falta lung_mask o mask de lesión.")
            skipped += 1
            continue

        out_img_path = out_images_dir / img_path.name
        out_mask_path = out_masks_dir / lesion_path.name

        if not args.overwrite and (out_img_path.exists() or out_mask_path.exists()):
            print(f"[SKIP] {stem}: salida ya existe. Usa --overwrite para reemplazar.")
            skipped += 1
            continue

        img = tiff.imread(img_path)
        lung_mask = tiff.imread(lung_path)
        lesion_mask = tiff.imread(lesion_path)

        lung_unique = np.unique(lung_mask)
        lung_mask_bin = lung_mask > 0

        lesion_unique_before, n_unique_before, suffix_before = unique_values_summary(
            lesion_mask, max_items=args.max_unique_print
        )

        img_seg = apply_lung_segmentation(img, lung_mask_bin, target_name=f"imagen {img_path.name}")
        lesion_seg = apply_lung_segmentation(
            lesion_mask, lung_mask_bin, target_name=f"máscara {lesion_path.name}"
        )

        lesion_unique_after, n_unique_after, suffix_after = unique_values_summary(
            lesion_seg, max_items=args.max_unique_print
        )

        # Guardado conservando dtype y valores. Fuera de pulmón queda en 0.
        tiff.imwrite(out_img_path, img_seg, compression=args.compression)
        tiff.imwrite(out_mask_path, lesion_seg, compression=args.compression)

        lung_pixels = int(np.count_nonzero(lung_mask_bin))
        total_pixels = int(lung_mask_bin.size)
        lung_percent = 100.0 * lung_pixels / total_pixels

        print(f"[OK] {stem}")
        print(f"  lung_mask únicos: {lung_unique.tolist()}")
        print(f"  lesión antes: {lesion_unique_before}{suffix_before}")
        print(f"  lesión después: {lesion_unique_after}{suffix_after}")
        print(f"  región pulmonar: {lung_pixels}/{total_pixels} píxeles ({lung_percent:.2f}%)")

        log_rows.append({
            "file": img_path.name,
            "image_shape": str(tuple(img.shape)),
            "image_dtype": str(img.dtype),
            "lung_mask_shape": str(tuple(lung_mask.shape)),
            "lung_mask_dtype": str(lung_mask.dtype),
            "lung_mask_unique": str(lung_unique.tolist()),
            "lesion_mask_shape": str(tuple(lesion_mask.shape)),
            "lesion_mask_dtype": str(lesion_mask.dtype),
            "lesion_unique_before": str(lesion_unique_before) + suffix_before,
            "n_lesion_unique_before": n_unique_before,
            "lesion_unique_after": str(lesion_unique_after) + suffix_after,
            "n_lesion_unique_after": n_unique_after,
            "lung_pixels": lung_pixels,
            "total_pixels": total_pixels,
            "lung_percent": round(lung_percent, 4),
            "out_image": str(out_img_path),
            "out_mask": str(out_mask_path),
        })

        processed += 1

    log_path = base_dir / "lung_segmentation_log.csv"
    if log_rows:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

    print("-" * 80)
    print(f"Procesadas: {processed}")
    print(f"Omitidas:   {skipped}")
    if log_rows:
        print(f"Log guardado en: {log_path}")


if __name__ == "__main__":
    main()

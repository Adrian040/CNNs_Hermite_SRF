from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from _bootstrap import ROOT  # noqa
from src.models.hermite_basis import HermiteBasisConfig, make_hermite_basis, save_basis


def parse_args():
    p = argparse.ArgumentParser(description="Genera y guarda banco de filtros Hermite/Gaussian derivatives.")
    p.add_argument("--out", default="assets/hermite_basis/hermite_order3_k7_scales_1.0_2.0.pt")
    p.add_argument("--kernel-size", type=int, default=7)
    p.add_argument("--max-order", type=int, default=3)
    p.add_argument("--scales", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--preview", action="store_true", help="Guarda una imagen PNG con los filtros.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = HermiteBasisConfig(kernel_size=args.kernel_size, max_order=args.max_order, scales=tuple(args.scales))
    path = save_basis(args.out, cfg, overwrite=args.overwrite)
    basis, meta = make_hermite_basis(cfg)
    print(f"Base guardada en: {path}")
    print(f"Filtros: {basis.shape[0]}, shape: {tuple(basis.shape)}")

    if args.preview:
        cols = min(6, basis.shape[0])
        rows = int((basis.shape[0] + cols - 1) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.ravel()
        for i in range(len(axes)):
            axes[i].axis("off")
            if i < basis.shape[0]:
                axes[i].imshow(basis[i, 0].numpy(), cmap="gray")
                m = meta[i]
                axes[i].set_title(f"a={m['a']}, b={m['b']}\nσ={m['sigma']}", fontsize=8)
        fig.tight_layout()
        preview_path = Path(args.out).with_suffix(".png")
        fig.savefig(preview_path, dpi=200)
        print(f"Preview guardado en: {preview_path}")


if __name__ == "__main__":
    main()

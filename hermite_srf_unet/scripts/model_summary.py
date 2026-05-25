from __future__ import annotations

import argparse
import torch

from _bootstrap import ROOT  # noqa
from src.models.unet_srf import build_model_from_config
from src.utils.config import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Muestra número de parámetros del modelo.")
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model = build_model_from_config(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Parámetros totales: {total:,}")
    print(f"Parámetros entrenables: {trainable:,}")


if __name__ == "__main__":
    main()

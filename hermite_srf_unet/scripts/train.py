from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from _bootstrap import ROOT  # noqa
from src.data.dataset import make_dataset_from_config
from src.models.unet_srf import build_model_from_config
from src.models.hermite_basis import HermiteBasisConfig, save_basis
from src.utils.checkpoints import save_checkpoint
from src.utils.config import load_config
from src.utils.losses import make_loss
from src.utils.metrics import MetricAccumulator, logits_to_pred
from src.utils.seed import set_seed
from src.utils.visualization import save_training_curves


def parse_args():
    p = argparse.ArgumentParser(description="Entrena U-Net con encoder SRF-Hermite.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def make_optimizer(cfg, model):
    name = cfg["train"].get("optimizer", "adamw").lower()
    lr = float(cfg["train"].get("lr", 3e-4))
    wd = float(cfg["train"].get("weight_decay", 1e-5))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def write_history_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_one_epoch(model, loader, criterion, optimizer, device, cfg, scaler=None):
    model.train()
    mode = cfg["data"].get("segmentation_mode", "binary")
    ncls = int(cfg["data"].get("num_classes", 2))
    threshold = float(cfg["train"].get("threshold", 0.5))
    acc = MetricAccumulator(mode=mode, num_classes=ncls)
    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=(scaler is not None and device.startswith("cuda"))):
            logits = model(images)
            loss = criterion(logits, masks)

        if scaler is not None and device.startswith("cuda"):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        preds = logits_to_pred(logits.detach(), mode=mode, threshold=threshold)
        acc.update(preds, masks.detach())

    metrics = acc.compute()
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg):
    model.eval()
    mode = cfg["data"].get("segmentation_mode", "binary")
    ncls = int(cfg["data"].get("num_classes", 2))
    threshold = float(cfg["train"].get("threshold", 0.5))
    acc = MetricAccumulator(mode=mode, num_classes=ncls)
    total_loss = 0.0

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += float(loss.item()) * images.size(0)
        preds = logits_to_pred(logits, mode=mode, threshold=threshold)
        acc.update(preds, masks)

    metrics = acc.compute()
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def ensure_basis(cfg):
    b = cfg.get("basis", {})
    path = b.get("path", "assets/hermite_basis/hermite_basis.pt")
    basis_cfg = HermiteBasisConfig(
        kernel_size=int(b.get("kernel_size", 7)),
        max_order=int(b.get("max_order", 3)),
        scales=tuple(float(s) for s in b.get("scales", [1.0, 2.0])),
        include_orders=b.get("include_orders", None),
        normalize=bool(b.get("normalize", True)),
        zero_mean_except_order0=bool(b.get("zero_mean_except_order0", True)),
    )
    save_basis(path, basis_cfg, overwrite=bool(b.get("overwrite", False)))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["project"].get("seed", 42)))
    ensure_basis(cfg)

    out_dir = Path(cfg["project"].get("output_dir", "outputs/exp01"))
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    fig_dir = out_dir / "figures"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    train_ds = make_dataset_from_config(cfg, "train")
    val_ds = make_dataset_from_config(cfg, "val")
    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"].get("batch_size", 8)), shuffle=True, num_workers=int(cfg["train"].get("num_workers", 2)), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["train"].get("batch_size", 8)), shuffle=False, num_workers=int(cfg["train"].get("num_workers", 2)), pin_memory=True)

    model = build_model_from_config(cfg).to(device)
    criterion = make_loss(cfg)
    optimizer = make_optimizer(cfg, model)

    sch_cfg = cfg["train"].get("scheduler", {})
    scheduler = None
    if sch_cfg.get("type", "none") == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=float(sch_cfg.get("factor", 0.5)), patience=int(sch_cfg.get("patience", 6)))

    start_epoch = 1
    best_dice = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_dice = float(ckpt.get("metrics", {}).get("val_dice", -1.0))

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("amp", True)) and device.startswith("cuda"))
    history = []
    epochs = int(cfg["train"].get("epochs", 80))
    save_every = int(cfg["train"].get("save_every", 10))
    patience = int(cfg["train"].get("early_stopping_patience", 0))
    no_improve = 0

    for epoch in range(start_epoch, epochs + 1):
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device, cfg, scaler=scaler)
        val_m = evaluate(model, val_loader, criterion, device, cfg)
        if scheduler is not None:
            scheduler.step(val_m["dice"])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_m["loss"],
            "val_loss": val_m["loss"],
            "train_dice": train_m["dice"],
            "val_dice": val_m["dice"],
            "train_iou": train_m["iou"],
            "val_iou": val_m["iou"],
            "train_precision": train_m["precision"],
            "val_precision": val_m["precision"],
            "train_recall": train_m["recall"],
            "val_recall": val_m["recall"],
        }
        history.append(row)
        hist_path = log_dir / "history.csv"
        write_history_csv(hist_path, history)
        try:
            save_training_curves(hist_path, fig_dir / "training_curves.png")
        except Exception as exc:
            print(f"No se pudo generar training_curves.png: {exc}")

        print(f"Epoch {epoch:03d}/{epochs} | train loss {row['train_loss']:.4f} dice {row['train_dice']:.4f} | val loss {row['val_loss']:.4f} dice {row['val_dice']:.4f}")

        if epoch % save_every == 0:
            save_checkpoint(ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pth", model, optimizer, scheduler, epoch, {"val_dice": val_m["dice"]}, cfg)

        if val_m["dice"] > best_dice:
            best_dice = val_m["dice"]
            no_improve = 0
            save_checkpoint(ckpt_dir / "model_best_dice.pth", model, optimizer, scheduler, epoch, {"val_dice": best_dice}, cfg)
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            print(f"Early stopping: {patience} épocas sin mejorar Dice.")
            break

    save_checkpoint(ckpt_dir / "model_final.pth", model, optimizer, scheduler, epoch, {"val_dice": val_m["dice"]}, cfg)
    print(f"Entrenamiento terminado. Mejor Val Dice: {best_dice:.4f}")
    print(f"Salidas en: {out_dir}")


if __name__ == "__main__":
    main()

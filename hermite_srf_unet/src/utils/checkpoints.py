from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, epoch: int = 0, metrics: dict | None = None, cfg: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
        "config": cfg or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def load_model_weights(model, checkpoint_path: str | Path, device="cpu") -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    return ckpt if isinstance(ckpt, dict) else {}

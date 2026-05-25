from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def logits_to_pred(logits: torch.Tensor, mode: str, threshold: float = 0.5) -> torch.Tensor:
    if mode == "binary":
        return (torch.sigmoid(logits) > threshold).long().squeeze(1)
    return torch.argmax(logits, dim=1).long()


class MetricAccumulator:
    def __init__(self, mode: str = "binary", num_classes: int = 2, include_background: bool = False):
        self.mode = mode
        self.num_classes = num_classes
        self.include_background = include_background
        self.reset()

    def reset(self) -> None:
        n = 1 if self.mode == "binary" else self.num_classes
        self.tp = np.zeros(n, dtype=np.float64)
        self.fp = np.zeros(n, dtype=np.float64)
        self.fn = np.zeros(n, dtype=np.float64)
        self.tn = np.zeros(n, dtype=np.float64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        p = preds.detach().cpu().numpy()
        t = targets.detach().cpu().numpy()
        if self.mode == "binary":
            if t.ndim == 4:
                t = t[:, 0]
            p = p.astype(bool)
            t = t.astype(bool)
            self.tp[0] += np.logical_and(p, t).sum()
            self.fp[0] += np.logical_and(p, ~t).sum()
            self.fn[0] += np.logical_and(~p, t).sum()
            self.tn[0] += np.logical_and(~p, ~t).sum()
        else:
            if t.ndim == 4:
                t = t[:, 0]
            for c in range(self.num_classes):
                pc = p == c
                tc = t == c
                self.tp[c] += np.logical_and(pc, tc).sum()
                self.fp[c] += np.logical_and(pc, ~tc).sum()
                self.fn[c] += np.logical_and(~pc, tc).sum()
                self.tn[c] += np.logical_and(~pc, ~tc).sum()

    def compute(self) -> dict[str, float]:
        eps = 1e-8
        dice = (2 * self.tp + eps) / (2 * self.tp + self.fp + self.fn + eps)
        iou = (self.tp + eps) / (self.tp + self.fp + self.fn + eps)
        precision = (self.tp + eps) / (self.tp + self.fp + eps)
        recall = (self.tp + eps) / (self.tp + self.fn + eps)

        if self.mode == "binary":
            return {
                "dice": float(dice[0]),
                "iou": float(iou[0]),
                "precision": float(precision[0]),
                "recall": float(recall[0]),
            }

        start = 0 if self.include_background else 1
        return {
            "dice": float(np.nanmean(dice[start:])),
            "iou": float(np.nanmean(iou[start:])),
            "precision": float(np.nanmean(precision[start:])),
            "recall": float(np.nanmean(recall[start:])),
        }


def surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return np.logical_xor(mask, binary_erosion(mask))


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("nan")

    sp = surface(pred)
    st = surface(target)
    if not sp.any() or not st.any():
        return float("nan")

    dt_target = distance_transform_edt(~st)
    dt_pred = distance_transform_edt(~sp)
    d1 = dt_target[sp].max() if sp.any() else np.nan
    d2 = dt_pred[st].max() if st.any() else np.nan
    return float(max(d1, d2))


def hausdorff95(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("nan")
    sp = surface(pred)
    st = surface(target)
    dt_target = distance_transform_edt(~st)
    dt_pred = distance_transform_edt(~sp)
    dists = np.concatenate([dt_target[sp], dt_pred[st]])
    if dists.size == 0:
        return float("nan")
    return float(np.percentile(dists, 95))


def per_image_metrics(pred: np.ndarray, target: np.ndarray, mode: str, num_classes: int = 2, include_background: bool = False) -> dict[str, float]:
    eps = 1e-8
    if mode == "binary":
        p = pred.astype(bool)
        t = target.astype(bool)
        tp = np.logical_and(p, t).sum()
        fp = np.logical_and(p, ~t).sum()
        fn = np.logical_and(~p, t).sum()
        return {
            "dice": float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
            "iou": float((tp + eps) / (tp + fp + fn + eps)),
            "precision": float((tp + eps) / (tp + fp + eps)),
            "recall": float((tp + eps) / (tp + fn + eps)),
            "hausdorff": hausdorff_distance(p, t),
            "hausdorff95": hausdorff95(p, t),
        }

    vals = []
    start = 0 if include_background else 1
    for c in range(start, num_classes):
        pc = pred == c
        tc = target == c
        tp = np.logical_and(pc, tc).sum()
        fp = np.logical_and(pc, ~tc).sum()
        fn = np.logical_and(~pc, tc).sum()
        vals.append({
            "dice": float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
            "iou": float((tp + eps) / (tp + fp + fn + eps)),
            "precision": float((tp + eps) / (tp + fp + eps)),
            "recall": float((tp + eps) / (tp + fn + eps)),
            "hausdorff": hausdorff_distance(pc, tc),
            "hausdorff95": hausdorff95(pc, tc),
        })
    return {k: float(np.nanmean([v[k] for v in vals])) for k in vals[0].keys()}

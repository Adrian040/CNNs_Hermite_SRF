from __future__ import annotations

import contextlib
import io

import numpy as np
import torch

try:
    with contextlib.redirect_stderr(io.StringIO()):
        from scipy.ndimage import binary_erosion, distance_transform_edt
except Exception:
    binary_erosion = None
    distance_transform_edt = None


def logits_to_pred(logits: torch.Tensor, mode: str, threshold: float = 0.5) -> torch.Tensor:
    if mode == "binary":
        return (torch.sigmoid(logits) > threshold).long().squeeze(1)
    return torch.argmax(logits, dim=1).long()


class MetricAccumulator:
    def __init__(self, mode: str = "binary", num_classes: int = 2, include_background: bool = False):
        self.mode = mode
        self.num_classes = max(2, int(num_classes)) if mode == "binary" else int(num_classes)
        self.include_background = include_background
        self.reset()

    def reset(self) -> None:
        n = self.num_classes
        self.tp = np.zeros(n, dtype=np.float64)
        self.fp = np.zeros(n, dtype=np.float64)
        self.fn = np.zeros(n, dtype=np.float64)
        self.tn = np.zeros(n, dtype=np.float64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        p = preds.detach().cpu().numpy()
        t = targets.detach().cpu().numpy()
        if t.ndim == 4:
            t = t[:, 0]
        if self.mode == "binary":
            p = p.astype(np.uint8)
            t = t.astype(np.uint8)
        for c in range(self.num_classes):
            pc = p == c
            tc = t == c
            self.tp[c] += np.logical_and(pc, tc).sum()
            self.fp[c] += np.logical_and(pc, ~tc).sum()
            self.fn[c] += np.logical_and(~pc, tc).sum()
            self.tn[c] += np.logical_and(~pc, ~tc).sum()

    def compute(self) -> dict[str, float]:
        rows = self.compute_per_class()
        start = 0 if self.include_background else 1
        selected = rows[start:]
        return {
            "dice": _nanmean_or([row["dice"] for row in selected], default=0.0),
            "iou": _nanmean_or([row["iou"] for row in selected], default=0.0),
            "precision": _nanmean_or([row["precision"] for row in selected], default=0.0),
            "recall": _nanmean_or([row["recall"] for row in selected], default=0.0),
        }

    def compute_per_class(self, class_names: list[str] | None = None) -> list[dict[str, float | int | str]]:
        rows = []
        for c in range(self.num_classes):
            tp = self.tp[c]
            fp = self.fp[c]
            fn = self.fn[c]
            tn = self.tn[c]
            rows.append({
                "class_id": c,
                "class_name": class_names[c] if class_names and c < len(class_names) else f"Clase {c}",
                "dice": _safe_divide(2 * tp, 2 * tp + fp + fn),
                "iou": _safe_divide(tp, tp + fp + fn),
                "precision": _safe_divide(tp, tp + fp),
                "recall": _safe_divide(tp, tp + fn),
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
                "tn": float(tn),
                "target_pixels": int(tp + fn),
                "pred_pixels": int(tp + fp),
            })
        return rows


def _safe_divide(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return float(num / den)


def _nanmean_or(values, default: float = float("nan")) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float(default)
    return float(arr.mean())


def surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    if binary_erosion is not None:
        erosion = binary_erosion(mask)
    else:
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        erosion = np.ones_like(mask, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                erosion &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return np.logical_xor(mask, erosion)


def _surface_distances(source_surface: np.ndarray, target_surface: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
    source = np.argwhere(source_surface)
    target = np.argwhere(target_surface)
    if source.size == 0 or target.size == 0:
        return np.array([], dtype=np.float64)
    distances = []
    target = target.astype(np.float64)
    for start in range(0, len(source), chunk_size):
        chunk = source[start : start + chunk_size].astype(np.float64)
        sq = ((chunk[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
        distances.append(np.sqrt(sq.min(axis=1)))
    return np.concatenate(distances)


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

    if distance_transform_edt is not None:
        dt_target = distance_transform_edt(~st)
        dt_pred = distance_transform_edt(~sp)
        d1 = dt_target[sp].max() if sp.any() else np.nan
        d2 = dt_pred[st].max() if st.any() else np.nan
    else:
        d1 = _surface_distances(sp, st).max()
        d2 = _surface_distances(st, sp).max()
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
    if distance_transform_edt is not None:
        dt_target = distance_transform_edt(~st)
        dt_pred = distance_transform_edt(~sp)
        dists = np.concatenate([dt_target[sp], dt_pred[st]])
    else:
        dists = np.concatenate([_surface_distances(sp, st), _surface_distances(st, sp)])
    if dists.size == 0:
        return float("nan")
    return float(np.percentile(dists, 95))


def _single_class_metrics(pred_class: np.ndarray, target_class: np.ndarray) -> dict[str, float | int | bool]:
    p = pred_class.astype(bool)
    t = target_class.astype(bool)
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    target_present = bool(t.any())
    pred_present = bool(p.any())
    return {
        "dice": _safe_divide(2 * tp, 2 * tp + fp + fn),
        "iou": _safe_divide(tp, tp + fp + fn),
        "precision": _safe_divide(tp, tp + fp),
        "recall": _safe_divide(tp, tp + fn),
        "hausdorff": hausdorff_distance(p, t) if (pred_present or target_present) else float("nan"),
        "hausdorff95": hausdorff95(p, t) if (pred_present or target_present) else float("nan"),
        "target_present": target_present,
        "pred_present": pred_present,
        "target_pixels": int(t.sum()),
        "pred_pixels": int(p.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def per_image_class_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mode: str,
    num_classes: int = 2,
    class_names: list[str] | None = None,
) -> list[dict[str, float | int | str | bool]]:
    ncls = max(2, int(num_classes)) if mode == "binary" else int(num_classes)
    rows = []
    for c in range(ncls):
        row = {
            "class_id": c,
            "class_name": class_names[c] if class_names and c < len(class_names) else f"Clase {c}",
        }
        row.update(_single_class_metrics(pred == c, target == c))
        rows.append(row)
    return rows


def per_image_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mode: str,
    num_classes: int = 2,
    include_background: bool = False,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    rows = per_image_class_metrics(pred, target, mode=mode, num_classes=num_classes, class_names=class_names)
    start = 0 if include_background else 1
    selected = [row for row in rows if int(row["class_id"]) >= start]
    out = {
        "dice": _nanmean_or([row["dice"] for row in selected], default=0.0),
        "iou": _nanmean_or([row["iou"] for row in selected], default=0.0),
        "precision": _nanmean_or([row["precision"] for row in selected], default=0.0),
        "recall": _nanmean_or([row["recall"] for row in selected], default=0.0),
        "hausdorff": _nanmean_or([row["hausdorff"] for row in selected]),
        "hausdorff95": _nanmean_or([row["hausdorff95"] for row in selected]),
    }
    for row in rows:
        c = int(row["class_id"])
        for key in ("dice", "iou", "precision", "recall", "hausdorff", "hausdorff95"):
            out[f"{key}_class_{c}"] = float(row[key])
        out[f"target_present_class_{c}"] = bool(row["target_present"])
        out[f"pred_present_class_{c}"] = bool(row["pred_present"])
        out[f"target_pixels_class_{c}"] = int(row["target_pixels"])
        out[f"pred_pixels_class_{c}"] = int(row["pred_pixels"])
    return out

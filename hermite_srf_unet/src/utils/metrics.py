from __future__ import annotations

import numpy as np
import torch
from importlib.metadata import PackageNotFoundError, version

_NDIMAGE_FUNCS = None
_NDIMAGE_ERROR = None


def _version_tuple(package: str) -> tuple[int, ...]:
    try:
        raw = version(package)
    except PackageNotFoundError:
        return ()
    parts = []
    for part in raw.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _get_ndimage_funcs():
    global _NDIMAGE_FUNCS, _NDIMAGE_ERROR
    if _NDIMAGE_FUNCS is not None:
        return _NDIMAGE_FUNCS
    if _NDIMAGE_ERROR is not None:
        return None
    numpy_v = _version_tuple("numpy")
    scipy_v = _version_tuple("scipy")
    if numpy_v >= (2,) and scipy_v and scipy_v < (1, 13):
        _NDIMAGE_ERROR = RuntimeError(
            f"SciPy {scipy_v} no es compatible con NumPy {numpy_v} en este entorno."
        )
        return None
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except Exception as exc:
        _NDIMAGE_ERROR = exc
        return None
    _NDIMAGE_FUNCS = (binary_erosion, distance_transform_edt)
    return _NDIMAGE_FUNCS


def logits_to_pred(logits: torch.Tensor, mode: str, threshold: float = 0.5) -> torch.Tensor:
    if mode == "binary":
        return (torch.sigmoid(logits) > threshold).long().squeeze(1)
    return torch.argmax(logits, dim=1).long()


def _segmentation_scores(tp, fp, fn):
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)

    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn

    dice = np.divide(2 * tp, dice_den, out=np.full_like(tp, np.nan), where=dice_den > 0)
    iou = np.divide(tp, iou_den, out=np.full_like(tp, np.nan), where=iou_den > 0)
    precision = np.divide(tp, precision_den, out=np.full_like(tp, np.nan), where=precision_den > 0)
    recall = np.divide(tp, recall_den, out=np.full_like(tp, np.nan), where=recall_den > 0)

    # Si hay clase real pero no se predijo ningún píxel, la precisión no es perfecta.
    precision[(precision_den == 0) & (recall_den > 0)] = 0.0

    return dice, iou, precision, recall


def _nanmean_or_nan(values: np.ndarray) -> float:
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


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
            self._validate_multiclass_values(p, t)
            for c in range(self.num_classes):
                pc = p == c
                tc = t == c
                self.tp[c] += np.logical_and(pc, tc).sum()
                self.fp[c] += np.logical_and(pc, ~tc).sum()
                self.fn[c] += np.logical_and(~pc, tc).sum()
                self.tn[c] += np.logical_and(~pc, ~tc).sum()

    def _validate_multiclass_values(self, preds: np.ndarray, targets: np.ndarray) -> None:
        invalid_pred = (preds < 0) | (preds >= self.num_classes)
        invalid_target = (targets < 0) | (targets >= self.num_classes)
        if np.any(invalid_pred) or np.any(invalid_target):
            details = []
            if np.any(invalid_pred):
                details.append(f"pred={np.unique(preds[invalid_pred])[:12].tolist()}")
            if np.any(invalid_target):
                details.append(f"target={np.unique(targets[invalid_target])[:12].tolist()}")
            raise ValueError(
                f"Valores multiclase fuera de 0..{self.num_classes - 1}: "
                + ", ".join(details)
            )

    def compute(self) -> dict[str, float]:
        dice, iou, precision, recall = _segmentation_scores(self.tp, self.fp, self.fn)

        if self.mode == "binary":
            return {
                "dice": float(dice[0]),
                "iou": float(iou[0]),
                "precision": float(precision[0]),
                "recall": float(recall[0]),
            }

        start = 0 if self.include_background else 1
        return {
            "dice": _nanmean_or_nan(dice[start:]),
            "iou": _nanmean_or_nan(iou[start:]),
            "precision": _nanmean_or_nan(precision[start:]),
            "recall": _nanmean_or_nan(recall[start:]),
        }


def surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    funcs = _get_ndimage_funcs()
    if funcs is None:
        return np.zeros_like(mask, dtype=bool)
    binary_erosion, _ = funcs
    return np.logical_xor(mask, binary_erosion(mask))


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("nan")
    funcs = _get_ndimage_funcs()
    if funcs is None:
        return float("nan")
    _, distance_transform_edt = funcs

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
    funcs = _get_ndimage_funcs()
    if funcs is None:
        return float("nan")
    _, distance_transform_edt = funcs
    sp = surface(pred)
    st = surface(target)
    dt_target = distance_transform_edt(~st)
    dt_pred = distance_transform_edt(~sp)
    dists = np.concatenate([dt_target[sp], dt_pred[st]])
    if dists.size == 0:
        return float("nan")
    return float(np.percentile(dists, 95))


def per_image_metrics(pred: np.ndarray, target: np.ndarray, mode: str, num_classes: int = 2, include_background: bool = False) -> dict[str, float]:
    if mode == "binary":
        p = pred.astype(bool)
        t = target.astype(bool)
        tp = np.logical_and(p, t).sum()
        fp = np.logical_and(p, ~t).sum()
        fn = np.logical_and(~p, t).sum()
        dice, iou, precision, recall = _segmentation_scores([tp], [fp], [fn])
        return {
            "dice": float(dice[0]),
            "iou": float(iou[0]),
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "hausdorff": hausdorff_distance(p, t),
            "hausdorff95": hausdorff95(p, t),
        }

    invalid_pred = (pred < 0) | (pred >= num_classes)
    invalid_target = (target < 0) | (target >= num_classes)
    if np.any(invalid_pred) or np.any(invalid_target):
        details = []
        if np.any(invalid_pred):
            details.append(f"pred={np.unique(pred[invalid_pred])[:12].tolist()}")
        if np.any(invalid_target):
            details.append(f"target={np.unique(target[invalid_target])[:12].tolist()}")
        raise ValueError(
            f"Valores multiclase fuera de 0..{num_classes - 1}: "
            + ", ".join(details)
        )

    vals = []
    start = 0 if include_background else 1
    for c in range(start, num_classes):
        pc = pred == c
        tc = target == c
        tp = np.logical_and(pc, tc).sum()
        fp = np.logical_and(pc, ~tc).sum()
        fn = np.logical_and(~pc, tc).sum()
        dice, iou, precision, recall = _segmentation_scores([tp], [fp], [fn])
        both_empty = not pc.any() and not tc.any()
        vals.append({
            "dice": float(dice[0]),
            "iou": float(iou[0]),
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "hausdorff": float("nan") if both_empty else hausdorff_distance(pc, tc),
            "hausdorff95": float("nan") if both_empty else hausdorff95(pc, tc),
        })
    return {k: _nanmean_or_nan(np.asarray([v[k] for v in vals], dtype=np.float64)) for k in vals[0].keys()}

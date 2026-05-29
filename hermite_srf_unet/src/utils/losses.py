from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = (1, 2, 3)
        inter = torch.sum(probs * targets, dim=dims)
        denom = torch.sum(probs + targets, dim=dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class MulticlassDiceLoss(nn.Module):
    def __init__(self, num_classes: int, include_background: bool = False, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        start = 0 if self.include_background else 1
        probs = probs[:, start:]
        one_hot = one_hot[:, start:]
        dims = (0, 2, 3)
        inter = torch.sum(probs * one_hot, dim=dims)
        denom = torch.sum(probs + one_hot, dim=dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


def make_loss(cfg: dict) -> nn.Module:
    mode = cfg["data"].get("segmentation_mode", "binary")
    loss_name = cfg["train"].get("loss", "bce_dice")
    dice_weight = float(cfg["train"].get("dice_weight", 0.5))
    class_weights = cfg["train"].get("class_weights", None)

    if mode == "binary":
        pos_weight = cfg["train"].get("pos_weight", None)
        if pos_weight is None and isinstance(class_weights, list) and len(class_weights) >= 2 and float(class_weights[0]) > 0:
            pos_weight = float(class_weights[1]) / float(class_weights[0])
        pos_weight_t = torch.tensor([float(pos_weight)], dtype=torch.float32) if pos_weight is not None else None
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
        dice = BinaryDiceLoss()
        if loss_name == "bce":
            return bce
        if loss_name == "dice":
            return dice
        return CombinedLoss(bce, dice, dice_weight=dice_weight)

    num_classes = int(cfg["data"].get("num_classes", 2))
    weight = torch.tensor(class_weights, dtype=torch.float32) if isinstance(class_weights, list) else None
    ce = nn.CrossEntropyLoss(weight=weight)
    dice = MulticlassDiceLoss(num_classes=num_classes, include_background=False)
    if loss_name == "ce":
        return ce
    if loss_name == "dice":
        return dice
    if loss_name not in {"ce_dice", "cross_entropy_dice", "bce_dice"}:
        raise ValueError(f"Loss multiclase no soportada: {loss_name}")
    return CombinedLoss(ce, dice, dice_weight=dice_weight)


class CombinedLoss(nn.Module):
    def __init__(self, main_loss: nn.Module, dice_loss: nn.Module, dice_weight: float = 0.5):
        super().__init__()
        self.main_loss = main_loss
        self.dice_loss = dice_loss
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.dice_weight) * self.main_loss(logits, targets) + self.dice_weight * self.dice_loss(logits, targets)

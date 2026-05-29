from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_class_metadata(cfg: dict) -> tuple[list[str], list[tuple[int, int, int]]]:
    dcfg = cfg.get("data", {})
    num_classes = int(dcfg.get("num_classes", 2))
    metadata_path = dcfg.get("class_labels_path")
    raw = {}
    if metadata_path:
        path = Path(metadata_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

    raw_names = raw.get("classes", {})
    raw_colors = raw.get("colors", {})
    default_colors = [
        (0, 0, 0),
        (230, 25, 75),
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
    ]

    names: list[str] = []
    colors: list[tuple[int, int, int]] = []
    for class_id in range(num_classes):
        names.append(str(raw_names.get(class_id, raw_names.get(str(class_id), f"Clase {class_id}"))))
        color = raw_colors.get(class_id, raw_colors.get(str(class_id)))
        if color is None:
            color = default_colors[class_id % len(default_colors)]
        colors.append(tuple(int(max(0, min(255, channel))) for channel in color[:3]))
    colors[0] = (0, 0, 0)
    return names, colors

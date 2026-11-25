from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from basketball_ai_tracker.vision.correction_exporter import CLASS_NAMES
from basketball_ai_tracker.vision.yolo_detector import normalize_yolo_class_name

EBARD_YOLO_MODEL_URL = (
    "https://huggingface.co/GabrieleGiudici/E-BARD-detection-models/"
    "resolve/main/BODD_yolov8n_0001.pt"
)
EBARD_MODEL_CARD_URL = "https://huggingface.co/GabrieleGiudici/E-BARD-detection-models"
EBARD_LICENSE = "cc-by-4.0"


def build_model_compatibility_report(class_names: Iterable[str]) -> dict:
    mapped: dict[str, list[str]] = {label: [] for label in CLASS_NAMES}
    unmapped: list[str] = []

    for raw_name in class_names:
        name = str(raw_name)
        label = normalize_yolo_class_name(name)
        if label is None:
            unmapped.append(name)
        else:
            mapped[label.value].append(name)

    missing_targets = [label for label in CLASS_NAMES if not mapped[label]]
    return {
        "target_classes": CLASS_NAMES,
        "mapped_classes": mapped,
        "unmapped_classes": unmapped,
        "missing_target_classes": missing_targets,
        "compatible": not missing_targets,
    }


def install_yolo_weights(source: Path, target: Path, *, overwrite: bool = False) -> Path:
    source = source.expanduser()
    target = target.expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Weights file not found: {source}")
    if source.suffix.lower() != ".pt":
        raise ValueError("Expected an Ultralytics/PyTorch .pt weights file.")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Target already exists: {target}. Use overwrite=True to replace it.")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def download_file(url: str, target: Path, *, overwrite: bool = False) -> Path:
    target = target.expanduser()
    if target.exists() and not overwrite:
        return target

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Install requests first: pip install requests") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_target = target.with_suffix(target.suffix + ".download")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temporary_target.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    temporary_target.replace(target)
    return target


def write_external_model_attribution(target_dir: Path, entries: list[dict]) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    attribution_path = target_dir / "ATTRIBUTION.md"
    lines = ["# External Model Attribution", ""]
    for entry in entries:
        lines.extend(
            [
                f"## {entry['name']}",
                "",
                f"- Source: {entry['source']}",
                f"- License: {entry['license']}",
                f"- Local file: `{entry['local_file']}`",
                "",
            ]
        )
    attribution_path.write_text("\n".join(lines), encoding="utf-8")
    return attribution_path


def class_names_from_ultralytics_weights(weights_path: Path) -> list[str]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install training dependencies first: pip install -e '.[train]'") from exc

    model = YOLO(str(weights_path.expanduser()))
    names = getattr(model, "names", {}) or {}
    if isinstance(names, dict):
        return [str(name) for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
    return [str(name) for name in names]

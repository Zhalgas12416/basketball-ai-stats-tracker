from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_CLASS_MAP = {
    "person": 0,
    "sports ball": 1,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap YOLO labels from a pretrained model.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets" / "basketball")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: pip install -e '.[train]'") from exc

    model = YOLO(args.model)
    total_labels = 0
    for split in ["train", "val"]:
        image_dir = args.dataset / "images" / split
        label_dir = args.dataset / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(_iter_images(image_dir)):
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists() and not args.overwrite:
                continue
            labels = _predict_labels(model, image_path, args.confidence, args.imgsz)
            label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            total_labels += len(labels)

    print(f"Wrote {total_labels} pseudo-labels.")
    print("Review and correct these labels before serious training, especially for the ball.")


def _iter_images(directory: Path):
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        yield from directory.glob(suffix)


def _predict_labels(model, image_path: Path, confidence: float, imgsz: int) -> list[str]:
    results = model.predict(str(image_path), conf=confidence, imgsz=imgsz, verbose=False)
    if not results:
        return []
    result = results[0]
    labels: list[str] = []
    names = result.names
    boxes = result.boxes
    if boxes is None:
        return labels

    for cls_tensor, xywhn_tensor in zip(boxes.cls, boxes.xywhn):
        source_name = names[int(cls_tensor)]
        target_class = DEFAULT_CLASS_MAP.get(source_name)
        if target_class is None:
            continue
        x_center, y_center, width, height = [float(value) for value in xywhn_tensor]
        labels.append(f"{target_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return labels


if __name__ == "__main__":
    main()

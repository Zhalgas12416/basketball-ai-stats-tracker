from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_NAMES = {
    0: "ball",
    1: "rim",
    2: "player",
    3: "referee",
}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.x_center - self.width / 2,
            self.y_center - self.height / 2,
            self.x_center + self.width / 2,
            self.y_center + self.height / 2,
        )


@dataclass
class ExportedHardFrame:
    source_image: str
    output_image: str
    ball_labels: int
    missed_balls: int
    predicted_balls: int
    best_ball_iou: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine labeled YOLO frames where a checkpoint misses ground-truth ball boxes."
    )
    parser.add_argument("--data", type=Path, required=True, help="Source YOLO data.yaml.")
    parser.add_argument("--model", type=Path, required=True, help="YOLO checkpoint to evaluate.")
    parser.add_argument("--output", type=Path, required=True, help="Output YOLO dataset directory.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-images", type=int, default=400, help="Maximum hard frames to export.")
    parser.add_argument("--conf", type=float, default=0.2, help="Prediction confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.25, help="IoU needed to count a ball as found.")
    parser.add_argument("--imgsz", type=int, default=704)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking.")
    parser.add_argument("--include-val", action="store_true", help="Copy/symlink source val split into output.")
    parser.add_argument("--include-test", action="store_true", help="Copy/symlink source test split into output.")
    parser.add_argument("--summary", type=Path, help="Optional JSON summary path.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: pip install -e '.[train]'") from exc

    dataset = _read_data_yaml(args.data)
    source_root = dataset["path"]
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    exported = _mine_split(
        model=model,
        source_root=source_root,
        split_path=dataset[args.split],
        output_root=output,
        output_split="train",
        max_images=args.max_images,
        conf=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        copy_images=args.copy_images,
    )

    if args.include_val and "val" in dataset:
        _copy_full_split(source_root, dataset["val"], output, "val", copy_images=args.copy_images)
    if args.include_test and "test" in dataset:
        _copy_full_split(source_root, dataset["test"], output, "test", copy_images=args.copy_images)

    _write_data_yaml(output, dataset.get("names", DEFAULT_NAMES))

    summary = {
        "source_data": str(args.data),
        "source_split": args.split,
        "model": str(args.model),
        "output": str(output),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "exported_count": len(exported),
        "exported": [asdict(item) for item in exported],
    }
    summary_path = args.summary or output / "false_negative_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Exported {len(exported)} hard ball frames to {output}")
    print(f"Wrote summary to {summary_path}")


def _read_data_yaml(data_yaml: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YOLO data.yaml files.") from exc

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    source_root = Path(payload["path"]).expanduser()
    if not source_root.is_absolute():
        source_root = (data_yaml.parent / source_root).resolve()
    names = payload.get("names", DEFAULT_NAMES)
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}
    else:
        names = {int(key): value for key, value in names.items()}
    return {
        "path": source_root.resolve(),
        "train": payload["train"],
        "val": payload.get("val"),
        "test": payload.get("test"),
        "names": names,
    }


def _mine_split(
    *,
    model,
    source_root: Path,
    split_path: str,
    output_root: Path,
    output_split: str,
    max_images: int,
    conf: float,
    iou_threshold: float,
    imgsz: int,
    device: str,
    copy_images: bool,
) -> list[ExportedHardFrame]:
    image_dir = source_root / split_path
    label_dir = source_root / split_path.replace("/images", "/labels")
    output_image_dir = output_root / output_split / "images"
    output_label_dir = output_root / output_split / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    exported: list[ExportedHardFrame] = []
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = _read_yolo_boxes(label_path)
        ball_labels = [box for box in labels if box.class_id == 0]
        if not ball_labels:
            continue

        result = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )[0]
        predicted_balls = _prediction_ball_boxes(result)
        missed_balls, best_iou = _count_missed_balls(ball_labels, predicted_balls, iou_threshold)
        if missed_balls == 0:
            continue

        output_image_path = output_image_dir / image_path.name
        output_label_path = output_label_dir / f"{image_path.stem}.txt"
        _symlink_or_copy(image_path, output_image_path, copy_images=copy_images)
        output_label_path.write_text(_format_yolo_labels(labels), encoding="utf-8")
        exported.append(
            ExportedHardFrame(
                source_image=str(image_path),
                output_image=str(output_image_path),
                ball_labels=len(ball_labels),
                missed_balls=missed_balls,
                predicted_balls=len(predicted_balls),
                best_ball_iou=best_iou,
            )
        )
        if len(exported) >= max_images:
            break

    return exported


def _copy_full_split(
    source_root: Path,
    split_path: str,
    output_root: Path,
    output_split: str,
    *,
    copy_images: bool,
) -> None:
    image_dir = source_root / split_path
    label_dir = source_root / split_path.replace("/images", "/labels")
    output_image_dir = output_root / output_split / "images"
    output_label_dir = output_root / output_split / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
        label_path = label_dir / f"{image_path.stem}.txt"
        _symlink_or_copy(image_path, output_image_dir / image_path.name, copy_images=copy_images)
        _symlink_or_copy(label_path, output_label_dir / label_path.name, copy_images=copy_images)


def _read_yolo_boxes(label_path: Path) -> list[YoloBox]:
    if not label_path.exists():
        return []
    boxes: list[YoloBox] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        values = [float(value) for value in parts[1:5]]
        if any(value < 0 or value > 1 for value in values):
            continue
        boxes.append(YoloBox(class_id, values[0], values[1], values[2], values[3]))
    return boxes


def _prediction_ball_boxes(result) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    if result.boxes is None or len(result.boxes) == 0:
        return boxes
    image_h, image_w = result.orig_shape
    xywhn = result.boxes.xywhn.cpu().tolist()
    class_ids = result.boxes.cls.cpu().tolist()
    for class_id, box in zip(class_ids, xywhn):
        if int(class_id) != 0:
            continue
        boxes.append(YoloBox(0, float(box[0]), float(box[1]), float(box[2]), float(box[3])))
    return boxes


def _count_missed_balls(
    ball_labels: list[YoloBox],
    predicted_balls: list[YoloBox],
    iou_threshold: float,
) -> tuple[int, float]:
    missed = 0
    best_seen_iou = 0.0
    for label in ball_labels:
        best_iou = max((_iou(label, prediction) for prediction in predicted_balls), default=0.0)
        best_seen_iou = max(best_seen_iou, best_iou)
        if best_iou < iou_threshold:
            missed += 1
    return missed, best_seen_iou


def _iou(first: YoloBox, second: YoloBox) -> float:
    ax1, ay1, ax2, ay2 = first.xyxy()
    bx1, by1, bx2, by2 = second.xyxy()
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else intersection / union


def _format_yolo_labels(boxes: list[YoloBox]) -> str:
    lines = [
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
        for box in boxes
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _symlink_or_copy(source: Path, target: Path, *, copy_images: bool) -> None:
    if target.exists() or target.is_symlink():
        return
    if copy_images:
        shutil.copy2(source, target)
        return
    try:
        relative_source = os.path.relpath(source, start=target.parent)
        target.symlink_to(relative_source)
    except OSError:
        shutil.copy2(source, target)


def _write_data_yaml(output: Path, names: dict[int, str]) -> None:
    names_yaml = "\n".join(f"  {class_id}: {class_name}" for class_id, class_name in sorted(names.items()))
    payload = f"""path: {output}
train: train/images
val: val/images
test: test/images

names:
{names_yaml}
"""
    (output / "data.yaml").write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()

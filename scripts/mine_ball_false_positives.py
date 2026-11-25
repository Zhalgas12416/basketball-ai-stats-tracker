from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


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
class ExportedFalsePositiveFrame:
    source_image: str
    output_image: str
    ball_labels: int
    predicted_balls: int
    false_positive_balls: int
    max_false_positive_confidence: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine YOLO frames where a checkpoint predicts ball boxes that do not match labels."
    )
    parser.add_argument("--data", type=Path, required=True, help="Source YOLO data.yaml.")
    parser.add_argument("--model", type=Path, required=True, help="YOLO checkpoint to evaluate.")
    parser.add_argument("--output", type=Path, required=True, help="Output YOLO dataset directory.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-images", type=int, default=400)
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--iou", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking.")
    parser.add_argument("--include-val", action="store_true")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: pip install -e '.[train]'") from exc

    dataset = _read_data_yaml(args.data)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    exported = _mine_split(
        model=model,
        source_root=dataset["path"],
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

    if args.include_val and dataset.get("val"):
        _copy_full_split(dataset["path"], dataset["val"], output, "val", copy_images=args.copy_images)
    if args.include_test and dataset.get("test"):
        _copy_full_split(dataset["path"], dataset["test"], output, "test", copy_images=args.copy_images)

    _write_data_yaml(output, dataset["names"])
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
    summary_path = args.summary or output / "false_positive_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Exported {len(exported)} false-positive ball frames to {output}")
    print(f"Wrote summary to {summary_path}")


def _read_data_yaml(data_yaml: Path) -> dict:
    import yaml

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    source_root = Path(payload["path"]).expanduser()
    if not source_root.is_absolute():
        source_root = (data_yaml.parent / source_root).resolve()
    names = payload.get("names", {})
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}
    else:
        names = {int(index): name for index, name in names.items()}
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
) -> list[ExportedFalsePositiveFrame]:
    image_dir = source_root / split_path
    label_dir = source_root / split_path.replace("/images", "/labels")
    output_image_dir = output_root / output_split / "images"
    output_label_dir = output_root / output_split / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    exported: list[ExportedFalsePositiveFrame] = []
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = _read_yolo_boxes(label_path)
        ball_labels = [box.xyxy() for box in labels if box.class_id == 0]
        result = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )[0]
        predicted_balls = _prediction_ball_boxes(result)
        false_positive_confidences = [
            confidence
            for confidence, box in predicted_balls
            if not ball_labels or max(_iou(box, label) for label in ball_labels) < iou_threshold
        ]
        if not false_positive_confidences:
            continue

        output_image_path = output_image_dir / image_path.name
        output_label_path = output_label_dir / f"{image_path.stem}.txt"
        _symlink_or_copy(image_path, output_image_path, copy_files=copy_images)
        output_label_path.write_text(_format_yolo_labels(labels), encoding="utf-8")
        exported.append(
            ExportedFalsePositiveFrame(
                source_image=str(image_path),
                output_image=str(output_image_path),
                ball_labels=len(ball_labels),
                predicted_balls=len(predicted_balls),
                false_positive_balls=len(false_positive_confidences),
                max_false_positive_confidence=max(false_positive_confidences),
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
        _symlink_or_copy(image_path, output_image_dir / image_path.name, copy_files=copy_images)
        _symlink_or_copy(label_path, output_label_dir / label_path.name, copy_files=copy_images)


def _read_yolo_boxes(label_path: Path) -> list[YoloBox]:
    if not label_path.exists():
        return []
    boxes: list[YoloBox] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        values = [float(value) for value in parts[1:5]]
        if any(value < 0 or value > 1 for value in values):
            continue
        boxes.append(YoloBox(class_id, values[0], values[1], values[2], values[3]))
    return boxes


def _prediction_ball_boxes(result) -> list[tuple[float, tuple[float, float, float, float]]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes: list[tuple[float, tuple[float, float, float, float]]] = []
    for class_id, confidence, box in zip(
        result.boxes.cls.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.xywhn.cpu().tolist(),
    ):
        if int(class_id) != 0:
            continue
        x_center, y_center, width, height = [float(value) for value in box]
        boxes.append(
            (
                float(confidence),
                (
                    x_center - width / 2,
                    y_center - height / 2,
                    x_center + width / 2,
                    y_center + height / 2,
                ),
            )
        )
    return boxes


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second
    intersection_x1 = max(first_x1, second_x1)
    intersection_y1 = max(first_y1, second_y1)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)
    intersection = max(0.0, intersection_x2 - intersection_x1) * max(0.0, intersection_y2 - intersection_y1)
    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _format_yolo_labels(labels: list[YoloBox]) -> str:
    return "".join(
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}\n"
        for box in labels
    )


def _symlink_or_copy(source: Path, target: Path, *, copy_files: bool) -> None:
    if target.exists() or target.is_symlink():
        return
    if copy_files:
        shutil.copy2(source, target)
        return
    try:
        target.symlink_to(os.path.relpath(source, start=target.parent))
    except OSError:
        shutil.copy2(source, target)


def _write_data_yaml(output: Path, names: dict[int, str]) -> None:
    names_yaml = "\n".join(f"  {class_id}: {class_name}" for class_id, class_name in sorted(names.items()))
    (output / "data.yaml").write_text(
        f"""path: {output}
train: train/images
val: val/images
test: test/images

names:
{names_yaml}
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

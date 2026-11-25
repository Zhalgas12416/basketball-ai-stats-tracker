from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class PixelBox:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a YOLO training dataset with tight ball-centered crops from a source YOLO dataset."
    )
    parser.add_argument("--data", type=Path, required=True, help="Source YOLO data.yaml.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ball-class-id", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=192, help="Minimum square crop side in pixels.")
    parser.add_argument(
        "--crop-scales",
        default="3,4,5",
        help="Comma-separated crop-side multiples of ball size. One crop is made per ball per scale.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.20,
        help="Minimum visible fraction of a non-ball box required to keep it in a crop.",
    )
    parser.add_argument("--train-limit", type=int, default=0, help="Optional max source train images to process.")
    parser.add_argument("--copy-val-test", action="store_true", help="Copy/symlink full-frame val/test splits.")
    parser.add_argument("--copy-files", action="store_true", help="Copy instead of symlinking preserved images.")
    parser.add_argument("--clean-output", action="store_true", help="Delete output before building.")
    args = parser.parse_args()

    dataset = _read_data_yaml(args.data)
    output = args.output.expanduser().resolve()
    if args.clean_output and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    crop_scales = _parse_crop_scales(args.crop_scales)
    train_count = _write_train_crops(
        source_root=dataset["path"],
        split_path=dataset["train"],
        output=output,
        ball_class_id=args.ball_class_id,
        crop_size=args.crop_size,
        crop_scales=crop_scales,
        min_visibility=args.min_visibility,
        train_limit=args.train_limit or None,
    )
    val_count = 0
    test_count = 0
    if args.copy_val_test and dataset.get("val"):
        val_count = _copy_split(dataset["path"], dataset["val"], output, "val", copy_files=args.copy_files)
    if args.copy_val_test and dataset.get("test"):
        test_count = _copy_split(dataset["path"], dataset["test"], output, "test", copy_files=args.copy_files)
    if not args.copy_val_test:
        val_count = _copy_split(dataset["path"], dataset["train"], output, "val", limit=200, copy_files=args.copy_files)

    _write_data_yaml(output, dataset["names"])
    _write_readme(
        output=output,
        source_data=args.data,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        crop_size=args.crop_size,
        crop_scales=crop_scales,
        min_visibility=args.min_visibility,
    )
    print(f"Built ball crop dataset at {output}")
    print(f"  train_crops: {train_count}")
    print(f"  val: {val_count}")
    print(f"  test: {test_count}")


def _read_data_yaml(data_yaml: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YOLO data.yaml files.") from exc

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
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


def _parse_crop_scales(raw_value: str) -> list[float]:
    crop_scales = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not crop_scales:
        raise ValueError("--crop-scales must include at least one positive number")
    if any(value <= 0 for value in crop_scales):
        raise ValueError("--crop-scales values must be positive")
    return crop_scales


def _write_train_crops(
    *,
    source_root: Path,
    split_path: str,
    output: Path,
    ball_class_id: int,
    crop_size: int,
    crop_scales: list[float],
    min_visibility: float,
    train_limit: int | None,
) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to create ball-centered crops.") from exc

    image_dir = source_root / split_path
    label_dir = source_root / split_path.replace("/images", "/labels")
    output_image_dir = output / "train" / "images"
    output_label_dir = output / "train" / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    processed = 0
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
        if train_limit is not None and processed >= train_limit:
            break
        processed += 1
        label_path = label_dir / f"{image_path.stem}.txt"
        boxes = _read_yolo_boxes(label_path)
        ball_boxes = [box for box in boxes if box.class_id == ball_class_id]
        if not ball_boxes:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image_h, image_w = image.shape[:2]
        pixel_boxes = [_to_pixel_box(box, image_w, image_h) for box in boxes]

        for ball_index, ball_box in enumerate(ball_boxes, start=1):
            for scale_index, crop_scale in enumerate(crop_scales, start=1):
                crop_x1, crop_y1, crop_x2, crop_y2 = _make_crop_bounds(
                    ball_box,
                    image_w,
                    image_h,
                    crop_size,
                    crop_scale,
                )
                crop_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
                if crop_image.size == 0:
                    continue
                crop_h, crop_w = crop_image.shape[:2]
                crop_labels = _boxes_to_crop(
                    boxes=pixel_boxes,
                    crop_x1=crop_x1,
                    crop_y1=crop_y1,
                    crop_x2=crop_x2,
                    crop_y2=crop_y2,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    min_visibility=min_visibility,
                )
                if not any(box.class_id == ball_class_id for box in crop_labels):
                    continue

                stem = f"{image_path.stem}_tightball_b{ball_index:02d}_s{scale_index:02d}"
                cv2.imwrite(
                    str(output_image_dir / f"{stem}.jpg"),
                    crop_image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                )
                (output_label_dir / f"{stem}.txt").write_text(_format_yolo_labels(crop_labels), encoding="utf-8")
                written += 1
    return written


def _copy_split(
    source_root: Path,
    split_path: str,
    output: Path,
    output_split: str,
    *,
    limit: int | None = None,
    copy_files: bool,
) -> int:
    image_dir = source_root / split_path
    label_dir = source_root / split_path.replace("/images", "/labels")
    output_image_dir = output / output_split / "images"
    output_label_dir = output / output_split / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
        if limit is not None and count >= limit:
            break
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        _symlink_or_copy(image_path, output_image_dir / image_path.name, copy_files=copy_files)
        _symlink_or_copy(label_path, output_label_dir / label_path.name, copy_files=copy_files)
        count += 1
    return count


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
        if any(value < 0.0 or value > 1.0 for value in values):
            continue
        if values[2] <= 0.0 or values[3] <= 0.0:
            continue
        boxes.append(YoloBox(class_id, values[0], values[1], values[2], values[3]))
    return boxes


def _format_yolo_labels(boxes: list[YoloBox]) -> str:
    lines = [
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
        for box in boxes
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _to_pixel_box(box: YoloBox, image_w: int, image_h: int) -> PixelBox:
    return PixelBox(
        class_id=box.class_id,
        x1=(box.x_center - box.width / 2) * image_w,
        y1=(box.y_center - box.height / 2) * image_h,
        x2=(box.x_center + box.width / 2) * image_w,
        y2=(box.y_center + box.height / 2) * image_h,
    )


def _boxes_to_crop(
    *,
    boxes: list[PixelBox],
    crop_x1: int,
    crop_y1: int,
    crop_x2: int,
    crop_y2: int,
    crop_w: int,
    crop_h: int,
    min_visibility: float,
) -> list[YoloBox]:
    crop_labels: list[YoloBox] = []
    for box in boxes:
        original_area = max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)
        if original_area <= 0:
            continue

        x1 = min(max(box.x1, crop_x1), crop_x2) - crop_x1
        y1 = min(max(box.y1, crop_y1), crop_y2) - crop_y1
        x2 = min(max(box.x2, crop_x1), crop_x2) - crop_x1
        y2 = min(max(box.y2, crop_y1), crop_y2) - crop_y1
        visible_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if visible_area <= 0:
            continue
        if box.class_id != 0 and visible_area / original_area < min_visibility:
            continue

        crop_labels.append(
            YoloBox(
                class_id=box.class_id,
                x_center=((x1 + x2) / 2) / crop_w,
                y_center=((y1 + y2) / 2) / crop_h,
                width=(x2 - x1) / crop_w,
                height=(y2 - y1) / crop_h,
            )
        )
    return crop_labels


def _make_crop_bounds(
    box: YoloBox,
    image_w: int,
    image_h: int,
    crop_size: int,
    crop_scale: float,
) -> tuple[int, int, int, int]:
    box_w = box.width * image_w
    box_h = box.height * image_h
    side = int(round(max(crop_size, box_w * crop_scale, box_h * crop_scale)))
    side = max(16, min(side, image_w, image_h))
    center_x = box.x_center * image_w
    center_y = box.y_center * image_h
    x1 = int(round(center_x - side / 2))
    y1 = int(round(center_y - side / 2))
    x1 = min(max(0, x1), max(0, image_w - side))
    y1 = min(max(0, y1), max(0, image_h - side))
    return x1, y1, min(image_w, x1 + side), min(image_h, y1 + side)


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


def _write_readme(
    *,
    output: Path,
    source_data: Path,
    train_count: int,
    val_count: int,
    test_count: int,
    crop_size: int,
    crop_scales: list[float],
    min_visibility: float,
) -> None:
    readme = f"""# YOLO Tight Ball Crop Dataset

Created with `scripts/build_yolo_ball_crop_dataset.py`.

- source_data: `{source_data}`
- train_crops: {train_count}
- val: {val_count}
- test: {test_count}
- crop_size: {crop_size}
- crop_scales: {", ".join(str(value) for value in crop_scales)}
- min_visibility: {min_visibility}
- output: `{output}`
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()

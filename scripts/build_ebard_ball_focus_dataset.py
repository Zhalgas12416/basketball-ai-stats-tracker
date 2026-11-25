from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an E-BARD ball-only YOLO dataset with optional ball-centered training crops."
    )
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "datasets" / "ebard-yolo-original")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets" / "ebard-ball-focus")
    parser.add_argument("--ball-class-id", type=int, default=0, help="Source class id for basketball in E-BARD.")
    parser.add_argument("--crop-size", type=int, default=320, help="Minimum square crop size in pixels.")
    parser.add_argument("--crop-scale", type=float, default=10.0, help="Crop side as a multiple of ball size.")
    parser.add_argument(
        "--crop-scales",
        default=None,
        help="Comma-separated crop-side multiples of ball size. Overrides --crop-scale when provided.",
    )
    parser.add_argument("--no-crops", action="store_true", help="Only create full-frame ball-only labels.")
    parser.add_argument(
        "--skip-empty-train",
        action="store_true",
        help="Skip training frames without a ball label. Validation/test frames are always preserved.",
    )
    parser.add_argument("--clean-output", action="store_true", help="Delete the output dataset before building.")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.clean_output and output.exists():
        shutil.rmtree(output)

    if not source.exists():
        raise FileNotFoundError(f"E-BARD source dataset not found: {source}")

    crop_scales = _parse_crop_scales(args.crop_scales, default=args.crop_scale)

    crop_count = 0
    full_frame_count = 0
    ball_count = 0
    positive_frame_count = 0
    for split in ["train", "val", "test"]:
        split_result = _build_split(
            source=source,
            output=output,
            split=split,
            ball_class_id=args.ball_class_id,
            crop_size=args.crop_size,
            crop_scales=crop_scales,
            include_crops=(split == "train" and not args.no_crops),
            skip_empty=(split == "train" and args.skip_empty_train),
        )
        full_frame_count += split_result["full_frames"]
        ball_count += split_result["balls"]
        positive_frame_count += split_result["positive_frames"]
        crop_count += split_result["crops"]
        print(
            f"{split}: full_frames={split_result['full_frames']} "
            f"positive_frames={split_result['positive_frames']} balls={split_result['balls']} "
            f"crops={split_result['crops']}"
        )

    _write_data_yaml(output)
    _write_readme(output, source, crop_size=args.crop_size, crop_scales=crop_scales)
    print("\nBuilt ball-focused dataset")
    print(f"  output: {output}")
    print(f"  full_frames: {full_frame_count}")
    print(f"  positive_frames: {positive_frame_count}")
    print(f"  balls: {ball_count}")
    print(f"  training_crops: {crop_count}")


def _build_split(
    *,
    source: Path,
    output: Path,
    split: str,
    ball_class_id: int,
    crop_size: int,
    crop_scales: list[float],
    include_crops: bool,
    skip_empty: bool,
) -> dict[str, int]:
    image_dir = source / split / "images"
    label_dir = source / split / "labels"
    if not image_dir.exists() or not label_dir.exists():
        return {"full_frames": 0, "positive_frames": 0, "balls": 0, "crops": 0}

    output_image_dir = output / split / "images"
    output_label_dir = output / split / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    full_frames = 0
    positive_frames = 0
    ball_count = 0
    crop_count = 0
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
        source_label_path = label_dir / f"{image_path.stem}.txt"
        source_boxes = _read_yolo_boxes(source_label_path)
        ball_boxes = [box for box in source_boxes if box.class_id == ball_class_id]
        if skip_empty and not ball_boxes:
            continue

        output_image_path = output_image_dir / image_path.name
        output_label_path = output_label_dir / f"{image_path.stem}.txt"
        _symlink_or_copy(image_path, output_image_path)
        output_label_path.write_text(_format_ball_labels(ball_boxes), encoding="utf-8")

        full_frames += 1
        ball_count += len(ball_boxes)
        if ball_boxes:
            positive_frames += 1

        if include_crops and ball_boxes:
            crop_count += _write_ball_crops(
                image_path=image_path,
                output_image_dir=output_image_dir,
                output_label_dir=output_label_dir,
                ball_boxes=ball_boxes,
                crop_size=crop_size,
                crop_scales=crop_scales,
            )

    return {
        "full_frames": full_frames,
        "positive_frames": positive_frames,
        "balls": ball_count,
        "crops": crop_count,
    }


def _parse_crop_scales(raw_value: str | None, *, default: float) -> list[float]:
    if raw_value is None:
        return [default]
    crop_scales = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not crop_scales:
        raise ValueError("--crop-scales must include at least one positive number")
    if any(value <= 0 for value in crop_scales):
        raise ValueError("--crop-scales values must be positive")
    return crop_scales


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
        boxes.append(
            YoloBox(
                class_id=int(float(parts[0])),
                x_center=float(parts[1]),
                y_center=float(parts[2]),
                width=float(parts[3]),
                height=float(parts[4]),
            )
        )
    return boxes


def _format_ball_labels(ball_boxes: list[YoloBox]) -> str:
    lines = [
        f"0 {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
        for box in ball_boxes
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _write_ball_crops(
    *,
    image_path: Path,
    output_image_dir: Path,
    output_label_dir: Path,
    ball_boxes: list[YoloBox],
    crop_size: int,
    crop_scales: list[float],
) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to create ball-centered crops.") from exc

    image = cv2.imread(str(image_path))
    if image is None:
        return 0

    image_h, image_w = image.shape[:2]
    written = 0
    for index, box in enumerate(ball_boxes, start=1):
        for scale_index, crop_scale in enumerate(crop_scales, start=1):
            crop = _make_crop_bounds(box, image_w, image_h, crop_size, crop_scale)
            crop_x1, crop_y1, crop_x2, crop_y2 = crop
            crop_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop_image.size == 0:
                continue

            crop_h, crop_w = crop_image.shape[:2]
            crop_box = _box_to_crop(box, image_w, image_h, crop_x1, crop_y1, crop_w, crop_h)
            if crop_box is None:
                continue

            stem = f"{image_path.stem}_ballcrop_{index:02d}_s{scale_index:02d}"
            crop_image_path = output_image_dir / f"{stem}.jpg"
            crop_label_path = output_label_dir / f"{stem}.txt"
            cv2.imwrite(str(crop_image_path), crop_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            crop_label_path.write_text(_format_ball_labels([crop_box]), encoding="utf-8")
            written += 1

    return written


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
    x2 = min(image_w, x1 + side)
    y2 = min(image_h, y1 + side)
    return x1, y1, x2, y2


def _box_to_crop(
    box: YoloBox,
    image_w: int,
    image_h: int,
    crop_x1: int,
    crop_y1: int,
    crop_w: int,
    crop_h: int,
) -> YoloBox | None:
    x1 = (box.x_center - box.width / 2) * image_w - crop_x1
    y1 = (box.y_center - box.height / 2) * image_h - crop_y1
    x2 = (box.x_center + box.width / 2) * image_w - crop_x1
    y2 = (box.y_center + box.height / 2) * image_h - crop_y1

    x1 = min(max(0.0, x1), float(crop_w))
    y1 = min(max(0.0, y1), float(crop_h))
    x2 = min(max(0.0, x2), float(crop_w))
    y2 = min(max(0.0, y2), float(crop_h))
    if x2 <= x1 or y2 <= y1:
        return None

    return YoloBox(
        class_id=0,
        x_center=((x1 + x2) / 2) / crop_w,
        y_center=((y1 + y2) / 2) / crop_h,
        width=(x2 - x1) / crop_w,
        height=(y2 - y1) / crop_h,
    )


def _symlink_or_copy(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    try:
        relative_source = os.path.relpath(source, start=target.parent)
        target.symlink_to(relative_source)
    except OSError:
        shutil.copy2(source, target)


def _write_data_yaml(output: Path) -> None:
    data_yaml = f"""path: {output}
train: train/images
val: val/images
test: test/images

names:
  0: ball
"""
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")


def _write_readme(output: Path, source: Path, *, crop_size: int, crop_scales: list[float]) -> None:
    readme = f"""# E-BARD Ball-Focused Dataset

Generated from `{source}` using `scripts/build_ebard_ball_focus_dataset.py`.

This dataset keeps only the E-BARD basketball labels and maps them to class `0: ball`.
Training split also includes ball-centered crops to improve small-object learning.

Generation settings:

- crop_size: {crop_size}
- crop_scales: {", ".join(str(value) for value in crop_scales)}

Source: https://huggingface.co/datasets/GabrieleGiudici/E-BARD-detection
License: CC-BY-4.0
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()

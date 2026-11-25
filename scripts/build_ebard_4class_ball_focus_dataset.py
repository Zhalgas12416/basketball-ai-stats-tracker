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
CANONICAL_NAMES = {
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


@dataclass(frozen=True)
class PixelBox:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a 4-class E-BARD YOLO dataset that preserves ball/rim/player/referee "
            "labels and oversamples ball frames with ball-centered crops."
        )
    )
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "datasets" / "ebard-yolo-original")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets" / "ebard-4class-ball-focus")
    parser.add_argument("--crop-size", type=int, default=320, help="Minimum square crop side in pixels.")
    parser.add_argument(
        "--crop-scales",
        default="6,10,14",
        help="Comma-separated crop-side multiples of ball size. One crop is made per ball per scale.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.15,
        help="Minimum visible fraction of a non-ball box required to keep it in a crop.",
    )
    parser.add_argument(
        "--positive-frame-copies",
        type=int,
        default=0,
        help="Extra full-context copies to add for train frames that contain at least one ball.",
    )
    parser.add_argument("--no-crops", action="store_true", help="Only create full-frame remapped labels.")
    parser.add_argument("--copy-images", action="store_true", help="Copy full-frame images instead of symlinking them.")
    parser.add_argument("--clean-output", action="store_true", help="Delete the output dataset before building.")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    crop_scales = _parse_crop_scales(args.crop_scales)

    if args.clean_output and output.exists():
        shutil.rmtree(output)
    if not source.exists():
        raise FileNotFoundError(f"E-BARD source dataset not found: {source}")

    totals = {
        "full_frames": 0,
        "positive_frames": 0,
        "balls": 0,
        "positive_frame_copies": 0,
        "crops": 0,
    }
    for split in ["train", "val", "test"]:
        result = _build_split(
            source=source,
            output=output,
            split=split,
            crop_size=args.crop_size,
            crop_scales=crop_scales,
            min_visibility=args.min_visibility,
            positive_frame_copies=args.positive_frame_copies,
            include_crops=(split == "train" and not args.no_crops),
            copy_images=args.copy_images,
        )
        for key in totals:
            totals[key] += result[key]
        print(
            f"{split}: full_frames={result['full_frames']} "
            f"positive_frames={result['positive_frames']} balls={result['balls']} "
            f"positive_frame_copies={result['positive_frame_copies']} "
            f"crops={result['crops']}"
        )

    _write_data_yaml(output)
    _write_readme(
        output=output,
        source=source,
        crop_size=args.crop_size,
        crop_scales=crop_scales,
        min_visibility=args.min_visibility,
        positive_frame_copies=args.positive_frame_copies,
    )
    print("\nBuilt 4-class ball-focused dataset")
    print(f"  output: {output}")
    print(f"  full_frames: {totals['full_frames']}")
    print(f"  positive_frames: {totals['positive_frames']}")
    print(f"  balls: {totals['balls']}")
    print(f"  positive_frame_copies: {totals['positive_frame_copies']}")
    print(f"  training_crops: {totals['crops']}")


def _parse_crop_scales(raw_value: str) -> list[float]:
    crop_scales = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not crop_scales:
        raise ValueError("--crop-scales must include at least one positive number")
    if any(value <= 0 for value in crop_scales):
        raise ValueError("--crop-scales values must be positive")
    return crop_scales


def _build_split(
    *,
    source: Path,
    output: Path,
    split: str,
    crop_size: int,
    crop_scales: list[float],
    min_visibility: float,
    positive_frame_copies: int,
    include_crops: bool,
    copy_images: bool,
) -> dict[str, int]:
    image_dir = source / split / "images"
    label_dir = source / split / "labels"
    if not image_dir.exists() or not label_dir.exists():
        return {"full_frames": 0, "positive_frames": 0, "balls": 0, "positive_frame_copies": 0, "crops": 0}

    output_image_dir = output / split / "images"
    output_label_dir = output / split / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    full_frames = 0
    positive_frames = 0
    ball_count = 0
    copied_positive_frames = 0
    crop_count = 0
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
        source_label_path = label_dir / f"{image_path.stem}.txt"
        source_boxes = _read_yolo_boxes(source_label_path)
        boxes = [box for box in source_boxes if box.class_id in CANONICAL_NAMES]
        ball_boxes = [box for box in boxes if box.class_id == 0]

        output_image_path = output_image_dir / image_path.name
        output_label_path = output_label_dir / f"{image_path.stem}.txt"
        _symlink_or_copy(image_path, output_image_path, copy_images=copy_images)
        output_label_path.write_text(_format_yolo_labels(boxes), encoding="utf-8")

        full_frames += 1
        ball_count += len(ball_boxes)
        if ball_boxes:
            positive_frames += 1
            if split == "train" and positive_frame_copies > 0:
                copied_positive_frames += _write_positive_frame_copies(
                    image_path=image_path,
                    output_image_dir=output_image_dir,
                    output_label_dir=output_label_dir,
                    boxes=boxes,
                    copies=positive_frame_copies,
                    copy_images=copy_images,
                )

        if include_crops and ball_boxes:
            crop_count += _write_ball_crops(
                image_path=image_path,
                output_image_dir=output_image_dir,
                output_label_dir=output_label_dir,
                boxes=boxes,
                ball_boxes=ball_boxes,
                crop_size=crop_size,
                crop_scales=crop_scales,
                min_visibility=min_visibility,
            )

    return {
        "full_frames": full_frames,
        "positive_frames": positive_frames,
        "balls": ball_count,
        "positive_frame_copies": copied_positive_frames,
        "crops": crop_count,
    }


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
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        if class_id not in CANONICAL_NAMES:
            continue
        if width <= 0 or height <= 0:
            continue
        if any(value < 0.0 or value > 1.0 for value in [x_center, y_center, width, height]):
            continue
        boxes.append(
            YoloBox(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )
    return boxes


def _format_yolo_labels(boxes: list[YoloBox]) -> str:
    lines = [
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
        for box in boxes
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _write_ball_crops(
    *,
    image_path: Path,
    output_image_dir: Path,
    output_label_dir: Path,
    boxes: list[YoloBox],
    ball_boxes: list[YoloBox],
    crop_size: int,
    crop_scales: list[float],
    min_visibility: float,
) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to create ball-centered crops.") from exc

    image = cv2.imread(str(image_path))
    if image is None:
        return 0

    image_h, image_w = image.shape[:2]
    pixel_boxes = [_to_pixel_box(box, image_w, image_h) for box in boxes]
    written = 0
    for ball_index, ball_box in enumerate(ball_boxes, start=1):
        for scale_index, crop_scale in enumerate(crop_scales, start=1):
            crop = _make_crop_bounds(ball_box, image_w, image_h, crop_size, crop_scale)
            crop_x1, crop_y1, crop_x2, crop_y2 = crop
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
            if not any(box.class_id == 0 for box in crop_labels):
                continue

            stem = f"{image_path.stem}_ballfocus_b{ball_index:02d}_s{scale_index:02d}"
            crop_image_path = output_image_dir / f"{stem}.jpg"
            crop_label_path = output_label_dir / f"{stem}.txt"
            cv2.imwrite(str(crop_image_path), crop_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            crop_label_path.write_text(_format_yolo_labels(crop_labels), encoding="utf-8")
            written += 1

    return written


def _write_positive_frame_copies(
    *,
    image_path: Path,
    output_image_dir: Path,
    output_label_dir: Path,
    boxes: list[YoloBox],
    copies: int,
    copy_images: bool,
) -> int:
    written = 0
    for copy_index in range(1, copies + 1):
        stem = f"{image_path.stem}_ballpositive_c{copy_index:02d}"
        output_image_path = output_image_dir / f"{stem}{image_path.suffix.lower()}"
        output_label_path = output_label_dir / f"{stem}.txt"
        _symlink_or_copy(image_path, output_image_path, copy_images=copy_images)
        output_label_path.write_text(_format_yolo_labels(boxes), encoding="utf-8")
        written += 1
    return written


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
    x2 = min(image_w, x1 + side)
    y2 = min(image_h, y1 + side)
    return x1, y1, x2, y2


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


def _write_data_yaml(output: Path) -> None:
    data_yaml = f"""path: {output}
train: train/images
val: val/images
test: test/images

names:
  0: ball
  1: rim
  2: player
  3: referee
"""
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")


def _write_readme(
    *,
    output: Path,
    source: Path,
    crop_size: int,
    crop_scales: list[float],
    min_visibility: float,
    positive_frame_copies: int,
) -> None:
    readme = f"""# E-BARD 4-Class Ball-Focused Dataset

Generated from `{source}` using `scripts/build_ebard_4class_ball_focus_dataset.py`.

This dataset preserves the E-BARD detection labels and maps them to the app's canonical 4-class schema:

- `0: ball`
- `1: rim`
- `2: player`
- `3: referee`

The full-frame train/validation/test images are preserved. The training split also includes ball-centered crop
images to oversample small-ball examples while keeping any visible rim/player/referee labels inside each crop.
Optionally, full-context ball-positive train frames can also be duplicated so the model sees more ball examples
without changing the broadcast-camera scale.

Generation settings:

- crop_size: {crop_size}
- crop_scales: {", ".join(str(value) for value in crop_scales)}
- min_visibility: {min_visibility}
- positive_frame_copies: {positive_frame_copies}
- output: `{output}`

Source: https://huggingface.co/datasets/GabrieleGiudici/E-BARD-detection
License: CC-BY-4.0
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()

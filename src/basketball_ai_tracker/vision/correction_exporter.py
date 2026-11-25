from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

CLASS_NAMES = ["player", "ball", "rim", "backboard", "referee"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
VIDEO_EXTENSIONS = [".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"]


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_line(self) -> str:
        return (
            f"{self.class_id} "
            f"{self.x_center:.6f} "
            f"{self.y_center:.6f} "
            f"{self.width:.6f} "
            f"{self.height:.6f}"
        )


@dataclass
class ExportSummary:
    game_id: str
    exported_images: int = 0
    exported_labels: int = 0
    exported_boxes: int = 0
    skipped_corrections: list[str] = field(default_factory=list)
    output_dataset: Path | None = None

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "exported_images": self.exported_images,
            "exported_labels": self.exported_labels,
            "exported_boxes": self.exported_boxes,
            "skipped_corrections": self.skipped_corrections,
            "output_dataset": str(self.output_dataset) if self.output_dataset else None,
        }


def load_corrections(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as correction_file:
        payload = json.load(correction_file)
    return list(payload.get("corrections", []))


def bbox_to_yolo(bbox: dict, image_width: int, image_height: int, class_id: int) -> YoloBox:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive.")

    x = float(bbox["x"])
    y = float(bbox["y"])
    width = float(bbox["width"])
    height = float(bbox["height"])

    left = _clamp(x, 0.0, float(image_width))
    top = _clamp(y, 0.0, float(image_height))
    right = _clamp(x + width, 0.0, float(image_width))
    bottom = _clamp(y + height, 0.0, float(image_height))

    clipped_width = right - left
    clipped_height = bottom - top
    if clipped_width <= 0 or clipped_height <= 0:
        raise ValueError("bbox is outside the image bounds.")

    return YoloBox(
        class_id=class_id,
        x_center=((left + right) / 2.0) / image_width,
        y_center=((top + bottom) / 2.0) / image_height,
        width=clipped_width / image_width,
        height=clipped_height / image_height,
    )


def group_corrections_by_timestamp(corrections: Iterable[dict], precision: int = 3) -> dict[float, list[dict]]:
    grouped: dict[float, list[dict]] = {}
    for correction in corrections:
        timestamp = round(float(correction.get("video_time_seconds", 0.0)), precision)
        grouped.setdefault(timestamp, []).append(correction)
    return grouped


def find_video_for_game(game_id: str, upload_dir: Path) -> Path:
    for extension in VIDEO_EXTENSIONS:
        candidate = upload_dir / f"{game_id}{extension}"
        if candidate.exists():
            return candidate
    matches = sorted(upload_dir.glob(f"{game_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No uploaded video found for game_id={game_id}")


def export_corrections_to_yolo_dataset(
    *,
    game_id: str,
    video_path: Path,
    corrections_path: Path,
    output_dataset: Path,
    split: str = "train",
    image_quality: int = 95,
) -> ExportSummary:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install vision dependencies first: pip install -e '.[train]'") from exc

    corrections = load_corrections(corrections_path)
    summary = ExportSummary(game_id=game_id, output_dataset=output_dataset)
    if not corrections:
        write_dataset_yaml(output_dataset)
        return summary

    image_dir = output_dataset / "images" / split
    label_dir = output_dataset / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_yaml(output_dataset)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        for timestamp, timestamp_corrections in group_corrections_by_timestamp(corrections).items():
            frame = _read_frame_at_timestamp(capture, timestamp)
            if frame is None:
                summary.skipped_corrections.extend(
                    str(correction.get("correction_id", f"{timestamp:.3f}"))
                    for correction in timestamp_corrections
                )
                continue

            image_height, image_width = frame.shape[:2]
            frame_number = max(0, int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1)
            stem = f"{game_id}_frame_{frame_number:07d}_{int(round(timestamp * 1000)):010d}ms"
            image_path = image_dir / f"{stem}.jpg"
            label_path = label_dir / f"{stem}.txt"

            lines = []
            for correction in timestamp_corrections:
                label = str(correction.get("label", "")).strip().lower()
                class_id = CLASS_TO_ID.get(label)
                if class_id is None:
                    summary.skipped_corrections.append(str(correction.get("correction_id", label or timestamp)))
                    continue
                try:
                    yolo_box = bbox_to_yolo(correction["bbox"], image_width, image_height, class_id)
                except (KeyError, TypeError, ValueError):
                    summary.skipped_corrections.append(str(correction.get("correction_id", label)))
                    continue
                lines.append(yolo_box.to_line())

            if not lines:
                continue

            ok = cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), image_quality])
            if not ok:
                raise RuntimeError(f"Could not write exported frame: {image_path}")
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            summary.exported_images += 1
            summary.exported_labels += 1
            summary.exported_boxes += len(lines)
    finally:
        capture.release()

    return summary


def write_dataset_yaml(output_dataset: Path) -> Path:
    output_dataset.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dataset / "data.yaml"
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        f"path: .\ntrain: images/train\nval: images/val\n\nnames:\n{names}\n",
        encoding="utf-8",
    )
    return yaml_path


def _read_frame_at_timestamp(capture, timestamp_seconds: float):
    import cv2

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps > 0 and frame_count > 0:
        target_frame = int(max(0.0, timestamp_seconds) * fps)
        target_frame = min(max(0, target_frame), frame_count - 1)
        for frame_number in range(target_frame, max(-1, target_frame - 10), -1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if ok:
                return frame

    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_seconds) * 1000.0)
    ok, frame = capture.read()
    if ok:
        return frame
    return None


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))

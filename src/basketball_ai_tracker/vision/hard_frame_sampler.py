from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from basketball_ai_tracker.vision.correction_exporter import CLASS_NAMES, write_dataset_yaml

DEFAULT_TARGET_CLASSES = ["ball", "referee", "rim", "backboard"]


@dataclass(frozen=True)
class HardFrameCandidate:
    frame_number: int
    timestamp_seconds: float
    score: float
    reasons: list[str] = field(default_factory=list)
    nearby_detections: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "timestamp_seconds": round(self.timestamp_seconds, 4),
            "score": round(self.score, 4),
            "reasons": self.reasons,
            "nearby_detections": self.nearby_detections,
        }


@dataclass
class HardFrameExportSummary:
    game_id: str
    exported_images: int = 0
    exported_label_files: int = 0
    skipped_frames: list[int] = field(default_factory=list)
    output_dataset: Path | None = None
    metadata_path: Path | None = None

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "exported_images": self.exported_images,
            "exported_label_files": self.exported_label_files,
            "skipped_frames": self.skipped_frames,
            "output_dataset": str(self.output_dataset) if self.output_dataset else None,
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
        }


def select_hard_frame_candidates(
    vision: dict,
    *,
    target_classes: list[str] | None = None,
    max_frames: int = 80,
    sample_interval_seconds: float = 0.5,
    min_spacing_seconds: float = 1.0,
    low_confidence_threshold: float = 0.35,
    detection_window_seconds: float = 0.25,
    min_player_count: int = 4,
) -> list[HardFrameCandidate]:
    target_classes = _normalize_classes(target_classes or DEFAULT_TARGET_CLASSES)
    fps = _safe_float(vision.get("fps"), 0.0)
    duration_seconds = _safe_float(vision.get("duration_seconds"), 0.0)
    frame_count = _safe_int(vision.get("frame_count"), 0)
    detections = [_normalize_detection(detection) for detection in vision.get("detections") or []]
    detections = [detection for detection in detections if detection["label"]]

    if fps <= 0.0:
        fps = frame_count / duration_seconds if frame_count > 0 and duration_seconds > 0 else 30.0
    if duration_seconds <= 0.0 and frame_count > 0:
        duration_seconds = frame_count / fps
    if frame_count <= 0 and duration_seconds > 0:
        frame_count = int(round(duration_seconds * fps))

    timestamps = _candidate_timestamps(
        duration_seconds=duration_seconds,
        detections=detections,
        sample_interval_seconds=sample_interval_seconds,
    )
    scored = [
        _score_candidate(
            timestamp_seconds=timestamp,
            fps=fps,
            frame_count=frame_count,
            detections=detections,
            target_classes=target_classes,
            low_confidence_threshold=low_confidence_threshold,
            detection_window_seconds=detection_window_seconds,
            min_player_count=min_player_count,
        )
        for timestamp in timestamps
    ]
    scored = [candidate for candidate in scored if candidate.score > 0]
    scored.sort(key=lambda candidate: (-candidate.score, candidate.timestamp_seconds))
    return _dedupe_by_spacing(scored, max_frames=max_frames, min_spacing_seconds=min_spacing_seconds)


def export_hard_frames_to_yolo_labeling_dataset(
    *,
    game_id: str,
    video_path: Path,
    result_path: Path,
    output_dataset: Path,
    split: str = "train",
    target_classes: list[str] | None = None,
    max_frames: int = 80,
    sample_interval_seconds: float = 0.5,
    min_spacing_seconds: float = 1.0,
    low_confidence_threshold: float = 0.35,
    detection_window_seconds: float = 0.25,
    min_player_count: int = 4,
    image_quality: int = 95,
) -> HardFrameExportSummary:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install vision dependencies first: pip install -e '.[train]'") from exc

    with result_path.open(encoding="utf-8") as result_file:
        result = json.load(result_file)
    vision = result.get("vision") or {}
    candidates = select_hard_frame_candidates(
        vision,
        target_classes=target_classes,
        max_frames=max_frames,
        sample_interval_seconds=sample_interval_seconds,
        min_spacing_seconds=min_spacing_seconds,
        low_confidence_threshold=low_confidence_threshold,
        detection_window_seconds=detection_window_seconds,
        min_player_count=min_player_count,
    )

    image_dir = output_dataset / "images" / split
    label_dir = output_dataset / "labels" / split
    metadata_dir = output_dataset / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_yaml(output_dataset)

    summary = HardFrameExportSummary(game_id=game_id, output_dataset=output_dataset)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    exported_items = []
    try:
        for candidate in candidates:
            frame = _read_frame_at_index(capture, candidate.frame_number)
            if frame is None:
                summary.skipped_frames.append(candidate.frame_number)
                continue

            stem = f"{game_id}_hard_{candidate.frame_number:07d}_{int(round(candidate.timestamp_seconds * 1000)):010d}ms"
            image_path = image_dir / f"{stem}.jpg"
            label_path = label_dir / f"{stem}.txt"
            ok = cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), image_quality])
            if not ok:
                raise RuntimeError(f"Could not write hard frame: {image_path}")
            label_path.write_text("", encoding="utf-8")
            summary.exported_images += 1
            summary.exported_label_files += 1
            exported_items.append(
                {
                    **candidate.to_dict(),
                    "image": str(image_path.relative_to(output_dataset)),
                    "label": str(label_path.relative_to(output_dataset)),
                    "status": "needs_labeling",
                }
            )
    finally:
        capture.release()

    metadata_path = metadata_dir / f"{game_id}-hard-frames.json"
    metadata_path.write_text(
        json.dumps(
            {
                "game_id": game_id,
                "schema": "basketball-hard-frame-export-v1",
                "classes": CLASS_NAMES,
                "target_classes": _normalize_classes(target_classes or DEFAULT_TARGET_CLASSES),
                "settings": {
                    "max_frames": max_frames,
                    "sample_interval_seconds": sample_interval_seconds,
                    "min_spacing_seconds": min_spacing_seconds,
                    "low_confidence_threshold": low_confidence_threshold,
                    "detection_window_seconds": detection_window_seconds,
                    "min_player_count": min_player_count,
                },
                "frames": exported_items,
                "skipped_frames": summary.skipped_frames,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary.metadata_path = metadata_path
    return summary


def _score_candidate(
    *,
    timestamp_seconds: float,
    fps: float,
    frame_count: int,
    detections: list[dict],
    target_classes: list[str],
    low_confidence_threshold: float,
    detection_window_seconds: float,
    min_player_count: int,
) -> HardFrameCandidate:
    nearby = [
        detection
        for detection in detections
        if abs(detection["timestamp_seconds"] - timestamp_seconds) <= detection_window_seconds
    ]
    by_class: dict[str, list[dict]] = {}
    for detection in nearby:
        by_class.setdefault(detection["label"], []).append(detection)

    score = 0.0
    reasons: list[str] = []
    for label in target_classes:
        class_detections = by_class.get(label) or []
        if not class_detections:
            score += _missing_class_weight(label)
            reasons.append(f"missing_{label}")
            continue
        low_confidence = [detection for detection in class_detections if detection["confidence"] < low_confidence_threshold]
        if low_confidence:
            score += 1.0 + len(low_confidence) * 0.25
            reasons.append(f"low_confidence_{label}")

    player_count = len(by_class.get("player") or [])
    if player_count and player_count < min_player_count:
        score += 0.75
        reasons.append("sparse_players")

    frame_number = int(round(timestamp_seconds * fps))
    if frame_count > 0:
        frame_number = min(max(0, frame_number), frame_count - 1)
    nearby_summary = [
        {
            "frame_number": detection["frame_number"],
            "timestamp_seconds": round(detection["timestamp_seconds"], 4),
            "class_name": detection["label"],
            "confidence": round(detection["confidence"], 4),
            "bbox": detection.get("bbox"),
        }
        for detection in sorted(nearby, key=lambda item: (item["label"], -item["confidence"]))[:20]
    ]
    return HardFrameCandidate(
        frame_number=frame_number,
        timestamp_seconds=timestamp_seconds,
        score=score,
        reasons=reasons,
        nearby_detections=nearby_summary,
    )


def _candidate_timestamps(*, duration_seconds: float, detections: list[dict], sample_interval_seconds: float) -> list[float]:
    timestamps = set()
    if duration_seconds > 0 and sample_interval_seconds > 0:
        steps = int(duration_seconds / sample_interval_seconds) + 1
        for index in range(steps + 1):
            timestamp = min(duration_seconds, index * sample_interval_seconds)
            timestamps.add(round(timestamp, 3))
    for detection in detections:
        timestamps.add(round(detection["timestamp_seconds"], 3))
    return sorted(timestamps)


def _dedupe_by_spacing(
    candidates: list[HardFrameCandidate],
    *,
    max_frames: int,
    min_spacing_seconds: float,
) -> list[HardFrameCandidate]:
    selected: list[HardFrameCandidate] = []
    for candidate in candidates:
        if len(selected) >= max_frames:
            break
        if any(abs(candidate.timestamp_seconds - existing.timestamp_seconds) < min_spacing_seconds for existing in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda candidate: candidate.timestamp_seconds)


def _read_frame_at_index(capture, frame_number: int):
    import cv2

    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number))
    ok, frame = capture.read()
    if ok:
        return frame
    for fallback_frame in range(max(0, frame_number - 10), frame_number):
        capture.set(cv2.CAP_PROP_POS_FRAMES, fallback_frame)
        ok, frame = capture.read()
        if ok:
            return frame
    return None


def _normalize_detection(detection: dict) -> dict:
    timestamp = _safe_float(detection.get("time_seconds", detection.get("timestamp_seconds")), 0.0)
    return {
        "frame_number": _safe_int(detection.get("frame_index", detection.get("frame_number")), 0),
        "timestamp_seconds": timestamp,
        "label": str(detection.get("label") or detection.get("class_name") or "").strip().lower(),
        "confidence": _safe_float(detection.get("confidence"), 0.0),
        "bbox": detection.get("bbox") or detection.get("bbox_xyxy"),
    }


def _normalize_classes(classes: list[str]) -> list[str]:
    return [label for label in [str(item).strip().lower() for item in classes] if label in CLASS_NAMES]


def _missing_class_weight(label: str) -> float:
    if label == "ball":
        return 3.0
    if label in {"referee", "rim", "backboard"}:
        return 2.0
    return 1.0


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

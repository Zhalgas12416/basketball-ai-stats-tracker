from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

TARGET_CLASSES = ["player", "ball", "rim", "backboard", "referee"]
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.3


def build_detection_quality_report(
    vision: dict,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    low_confidence_limit: int = 80,
) -> dict:
    detections = list(vision.get("detections") or [])
    sampled_frame_count = _safe_int(vision.get("sampled_frame_count"), 0)
    class_counts = Counter({label: 0 for label in TARGET_CLASSES})
    low_confidence_by_class = Counter({label: 0 for label in TARGET_CLASSES})
    confidence_values: dict[str, list[float]] = defaultdict(list)
    frames_with_detections = set()
    low_confidence_detections = []

    for detection in detections:
        label = _detection_label(detection)
        confidence = _safe_float(detection.get("confidence"), 0.0)
        frame_number = _detection_frame_number(detection)
        if frame_number is not None:
            frames_with_detections.add(frame_number)

        if label in class_counts:
            class_counts[label] += 1
            confidence_values[label].append(confidence)
            if confidence < low_confidence_threshold:
                low_confidence_by_class[label] += 1
                low_confidence_detections.append(_low_confidence_entry(detection, label, frame_number, confidence))

    detected_frame_count = len(frames_with_detections)
    empty_sampled_frame_count = max(0, sampled_frame_count - detected_frame_count)
    sampled_frame_numbers = _sampled_frame_numbers(vision)
    skipped_frame_numbers = sorted(set(sampled_frame_numbers) - frames_with_detections) if sampled_frame_numbers else []

    return {
        "target_classes": TARGET_CLASSES,
        "low_confidence_threshold": low_confidence_threshold,
        "detection_count": len(detections),
        "sampled_frame_count": sampled_frame_count,
        "detected_frame_count": detected_frame_count,
        "empty_sampled_frame_count": empty_sampled_frame_count,
        "empty_sampled_frame_rate": _rate(empty_sampled_frame_count, sampled_frame_count),
        "skipped_frame_numbers": skipped_frame_numbers[:120],
        "skipped_frame_note": _skipped_frame_note(sampled_frame_numbers),
        "class_counts": dict(class_counts),
        "low_confidence_count": sum(low_confidence_by_class.values()),
        "low_confidence_by_class": dict(low_confidence_by_class),
        "average_confidence_by_class": _average_confidence_by_class(confidence_values),
        "low_confidence_detections": sorted(
            low_confidence_detections,
            key=lambda item: (item["confidence"], item["timestamp_seconds"], item["class_name"]),
        )[:low_confidence_limit],
    }


def compare_detection_quality_reports(before: dict, after: dict) -> dict:
    labels = list(dict.fromkeys((before.get("target_classes") or TARGET_CLASSES) + (after.get("target_classes") or TARGET_CLASSES)))
    return {
        "labels": labels,
        "detection_count_delta": _safe_int(after.get("detection_count"), 0) - _safe_int(before.get("detection_count"), 0),
        "detected_frame_count_delta": _safe_int(after.get("detected_frame_count"), 0) - _safe_int(before.get("detected_frame_count"), 0),
        "empty_sampled_frame_count_delta": _safe_int(after.get("empty_sampled_frame_count"), 0) - _safe_int(before.get("empty_sampled_frame_count"), 0),
        "low_confidence_count_delta": _safe_int(after.get("low_confidence_count"), 0) - _safe_int(before.get("low_confidence_count"), 0),
        "class_count_deltas": _count_deltas(before.get("class_counts") or {}, after.get("class_counts") or {}, labels),
        "low_confidence_deltas": _count_deltas(
            before.get("low_confidence_by_class") or {},
            after.get("low_confidence_by_class") or {},
            labels,
        ),
        "average_confidence_deltas": _confidence_deltas(
            before.get("average_confidence_by_class") or {},
            after.get("average_confidence_by_class") or {},
            labels,
        ),
    }


def build_detection_quality_report_from_result_path(
    result_path: Path,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
) -> dict:
    import json

    with result_path.open(encoding="utf-8") as result_file:
        result = json.load(result_file)
    return build_detection_quality_report(result.get("vision") or {}, low_confidence_threshold)


def _detection_label(detection: dict) -> str:
    return str(detection.get("label") or detection.get("class_name") or "").strip().lower()


def _detection_frame_number(detection: dict) -> int | None:
    value = detection.get("frame_index", detection.get("frame_number"))
    if value is None:
        return None
    return _safe_int(value, 0)


def _low_confidence_entry(detection: dict, label: str, frame_number: int | None, confidence: float) -> dict:
    return {
        "frame_number": frame_number,
        "timestamp_seconds": _safe_float(detection.get("time_seconds", detection.get("timestamp_seconds")), 0.0),
        "class_name": label,
        "confidence": round(confidence, 4),
        "bbox": detection.get("bbox") or detection.get("bbox_xyxy"),
    }


def _sampled_frame_numbers(vision: dict) -> list[int]:
    sampled_frames = vision.get("sampled_frames") or []
    numbers = []
    for sampled_frame in sampled_frames:
        if isinstance(sampled_frame, dict):
            value = sampled_frame.get("frame_index", sampled_frame.get("frame_number"))
        else:
            value = sampled_frame
        if value is not None:
            numbers.append(_safe_int(value, 0))
    return numbers


def _skipped_frame_note(sampled_frame_numbers: list[int]) -> str:
    if sampled_frame_numbers:
        return "Skipped frame numbers are exact because sampled frame indexes were recorded."
    return "Skipped count is exact for sampled frames; exact skipped frame numbers need sampled-frame logging in the detector."


def _average_confidence_by_class(confidence_values: dict[str, list[float]]) -> dict[str, float | None]:
    averages: dict[str, float | None] = {}
    for label in TARGET_CLASSES:
        values = confidence_values.get(label) or []
        averages[label] = round(sum(values) / len(values), 4) if values else None
    return averages


def _count_deltas(before_counts: dict, after_counts: dict, labels: list[str]) -> dict[str, int]:
    return {
        label: _safe_int(after_counts.get(label), 0) - _safe_int(before_counts.get(label), 0)
        for label in labels
    }


def _confidence_deltas(before_values: dict, after_values: dict, labels: list[str]) -> dict[str, float | None]:
    deltas: dict[str, float | None] = {}
    for label in labels:
        before = before_values.get(label)
        after = after_values.get(label)
        if before is None or after is None:
            deltas[label] = None
        else:
            deltas[label] = round(float(after) - float(before), 4)
    return deltas


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


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

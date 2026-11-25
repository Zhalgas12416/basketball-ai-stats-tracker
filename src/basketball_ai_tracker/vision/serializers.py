from __future__ import annotations

from basketball_ai_tracker.vision.models import BoundingBox, FrameDetection, ObjectTrack, TrackingResult


def bbox_to_dict(bbox: BoundingBox) -> dict:
    return {
        "x": round(bbox.x, 2),
        "y": round(bbox.y, 2),
        "width": round(bbox.width, 2),
        "height": round(bbox.height, 2),
        "center": [round(value, 2) for value in bbox.center],
    }


def detection_to_dict(detection: FrameDetection) -> dict:
    return {
        "frame_index": detection.frame_index,
        "time_seconds": detection.time_seconds,
        "label": detection.label.value,
        "bbox": bbox_to_dict(detection.bbox),
        "confidence": round(detection.confidence, 4),
    }


def track_to_dict(track: ObjectTrack) -> dict:
    return {
        "track_id": track.track_id,
        "label": track.label.value,
        "first_seen_seconds": track.first_seen_seconds,
        "last_seen_seconds": track.last_seen_seconds,
        "max_confidence": round(track.max_confidence, 4),
        "observation_count": track.observation_count,
        "points": [
            {
                "frame_index": point.frame_index,
                "time_seconds": point.time_seconds,
                "bbox": bbox_to_dict(point.bbox),
                "confidence": round(point.confidence, 4),
            }
            for point in track.points
        ],
    }


def tracking_result_to_dict(result: TrackingResult) -> dict:
    return {
        "fps": result.fps,
        "frame_count": result.frame_count,
        "width": result.width,
        "height": result.height,
        "duration_seconds": result.duration_seconds,
        "sampled_frame_count": result.sampled_frame_count,
        "analysis_region": bbox_to_dict(result.analysis_region) if result.analysis_region else None,
        "detector": result.detector,
        "detection_count": len(result.detections),
        "track_count": len(result.tracks),
        "detection_summary": _detection_summary(result.detections),
        "track_summary": _track_summary(result.tracks),
        "detections": [detection_to_dict(detection) for detection in result.detections],
        "tracks": [track_to_dict(track) for track in result.tracks],
        "warnings": result.warnings,
    }


def _detection_summary(detections: list[FrameDetection]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for detection in detections:
        label = detection.label.value
        summary[label] = summary.get(label, 0) + 1
    return dict(sorted(summary.items()))


def _track_summary(tracks: list[ObjectTrack]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for track in tracks:
        label = track.label.value
        summary[label] = summary.get(label, 0) + 1
    return dict(sorted(summary.items()))

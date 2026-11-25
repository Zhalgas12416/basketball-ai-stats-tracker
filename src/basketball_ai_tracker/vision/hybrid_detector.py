from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from basketball_ai_tracker.vision.models import (
    DetectionLabel,
    FrameDetection,
    ObjectTrack,
    TrackingResult,
)
from basketball_ai_tracker.vision.opencv_detector import OpenCVBasketballDetector
from basketball_ai_tracker.vision.yolo_detector import YOLOBasketballDetector


@dataclass(frozen=True)
class HybridDetectorConfig:
    max_supplemental_player_tracks: int = 80
    min_primary_player_tracks: int = 5


class HybridBasketballDetector:
    """Primary YOLO detector with OpenCV player-track fallback for weak baseline weights."""

    def __init__(
        self,
        primary: YOLOBasketballDetector,
        fallback: OpenCVBasketballDetector | None = None,
        config: HybridDetectorConfig | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback or OpenCVBasketballDetector()
        self.config = config or HybridDetectorConfig()

    def analyze_video(self, video_path: Path) -> TrackingResult:
        result = self.primary.analyze_video(video_path)
        primary_player_tracks = _count_tracks(result.tracks, DetectionLabel.PLAYER)
        if primary_player_tracks >= self.config.min_primary_player_tracks:
            result.detector = self.describe(supplemented_player_tracks=0)
            return result

        fallback_result = self.fallback.analyze_video(video_path)
        supplemental_tracks = _select_tracks(
            fallback_result.tracks,
            DetectionLabel.PLAYER,
            self.config.max_supplemental_player_tracks,
        )
        if not supplemental_tracks:
            result.warnings.append(
                "YOLO produced too few player tracks, and the OpenCV supplement did not find stable player tracks."
            )
            result.detector = self.describe(supplemented_player_tracks=0)
            return result

        result.tracks = _sort_tracks(result.tracks + supplemental_tracks)
        result.detections = _sort_detections(
            result.detections + _detections_from_tracks(supplemental_tracks)
        )
        result.warnings.append(
            f"YOLO produced {primary_player_tracks} player tracks, so {len(supplemental_tracks)} OpenCV player tracks were added as a temporary supplement."
        )
        result.warnings.extend(f"OpenCV supplement: {warning}" for warning in fallback_result.warnings)
        result.detector = self.describe(supplemented_player_tracks=len(supplemental_tracks))
        return result

    def describe(self, supplemented_player_tracks: int | None = None) -> dict[str, object]:
        primary = self.primary.describe()
        fallback = self.fallback.describe()
        return {
            "engine": "hybrid_yolo_opencv",
            "name": "Hybrid YOLO + OpenCV detector",
            "model_path": primary.get("model_path"),
            "model_exists": primary.get("model_exists"),
            "class_names": primary.get("class_names", {}),
            "sample_fps": primary.get("sample_fps"),
            "max_frames": primary.get("max_frames"),
            "confidence": primary.get("confidence"),
            "iou": primary.get("iou"),
            "imgsz": primary.get("imgsz"),
            "auto_crop": primary.get("auto_crop"),
            "primary": primary,
            "fallback": fallback,
            "supplement_when": "too_few_player_tracks",
            "supplemented_player_tracks": supplemented_player_tracks,
            "max_supplemental_player_tracks": self.config.max_supplemental_player_tracks,
            "min_primary_player_tracks": self.config.min_primary_player_tracks,
        }


def _has_tracks(tracks: list[ObjectTrack], label: DetectionLabel) -> bool:
    return any(track.label == label for track in tracks)


def _count_tracks(tracks: list[ObjectTrack], label: DetectionLabel) -> int:
    return sum(1 for track in tracks if track.label == label)


def _select_tracks(
    tracks: list[ObjectTrack],
    label: DetectionLabel,
    limit: int,
) -> list[ObjectTrack]:
    selected = [track for track in tracks if track.label == label]
    selected = sorted(
        selected,
        key=lambda track: (track.observation_count, track.max_confidence),
        reverse=True,
    )
    return _sort_tracks(selected[:limit])


def _detections_from_tracks(tracks: list[ObjectTrack]) -> list[FrameDetection]:
    return [
        FrameDetection(
            frame_index=point.frame_index,
            time_seconds=point.time_seconds,
            label=track.label,
            bbox=point.bbox,
            confidence=point.confidence,
        )
        for track in tracks
        for point in track.points
    ]


def _sort_tracks(tracks: list[ObjectTrack]) -> list[ObjectTrack]:
    return sorted(tracks, key=lambda track: (track.label.value, track.first_seen_seconds, track.track_id))


def _sort_detections(detections: list[FrameDetection]) -> list[FrameDetection]:
    return sorted(
        detections,
        key=lambda detection: (
            detection.frame_index,
            detection.label.value,
            detection.bbox.x,
            detection.bbox.y,
        ),
    )

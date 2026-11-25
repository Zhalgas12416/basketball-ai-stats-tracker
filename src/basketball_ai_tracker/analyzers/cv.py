from __future__ import annotations

from pathlib import Path
from typing import Protocol

from basketball_ai_tracker.models import AnalysisResult, PlayerIdentity
from basketball_ai_tracker.services.stats import build_player_stats
from basketball_ai_tracker.vision.models import DetectionLabel, ObjectTrack
from basketball_ai_tracker.vision.opencv_detector import OpenCVBasketballDetector
from basketball_ai_tracker.vision.serializers import tracking_result_to_dict


class BasketballVisionDetector(Protocol):
    def analyze_video(self, video_path: Path):
        """Run video detection and tracking."""

    def describe(self) -> dict[str, object]:
        """Return user-facing detector metadata."""


class CVBasketballAnalyzer:
    """Video detector/tracker analyzer."""

    def __init__(
        self,
        detector: BasketballVisionDetector | None = None,
        startup_warnings: list[str] | None = None,
    ) -> None:
        self.detector = detector or OpenCVBasketballDetector()
        self.startup_warnings = startup_warnings or []

    def analyze(self, game_id: str, video_path: Path, original_filename: str) -> AnalysisResult:
        tracking = self.detector.analyze_video(video_path)
        player_tracks = [track for track in tracking.tracks if track.label == DetectionLabel.PLAYER]
        players = [_player_from_track(track) for track in player_tracks]
        events = []
        stats = build_player_stats(players, events)
        warnings = self.startup_warnings + _detector_warnings(tracking.detector)
        warnings.extend(tracking.warnings)

        return AnalysisResult(
            game_id=game_id,
            video_filename=original_filename,
            duration_seconds=tracking.duration_seconds,
            players=players,
            events=events,
            stats=stats,
            warnings=warnings,
            vision=tracking_result_to_dict(tracking),
        )

    def describe(self) -> dict[str, object]:
        return {
            "name": "Video detector/tracker analyzer",
            "detector": self.detector.describe(),
            "warnings": self.startup_warnings,
        }


def _player_from_track(track: ObjectTrack) -> PlayerIdentity:
    return PlayerIdentity(
        player_id=track.track_id,
        team="Unknown",
        jersey_number="?",
        display_name=f"Tracked {track.track_id}",
    )


def _detector_warnings(detector: dict[str, object]) -> list[str]:
    engine = detector.get("engine")
    if engine == "hybrid_yolo_opencv":
        return [
            "Hybrid detector active: YOLO is used first, then OpenCV supplements player tracks if YOLO finds no players.",
            "This stage creates real object tracks only; jersey OCR and basketball action recognition are not credited as stats yet.",
            "Current YOLO weights are a quick CPU baseline, so player tracks may come from the temporary OpenCV supplement until a stronger model is trained.",
        ]
    if engine == "yolo":
        return [
            "YOLO detector active: using trained basketball detector weights for object detection.",
            "This stage creates real object tracks only; jersey OCR and basketball action recognition are not credited as stats yet.",
            "Current YOLO weights are a quick CPU baseline, so expect missed/incorrect detections until a longer GPU training run.",
        ]
    return [
        "OpenCV fallback detector active: player and ball tracks are video-derived heuristics, not trained YOLO detections.",
        "Action recognition and jersey OCR are not credited as stats yet; low-confidence review comes next.",
    ]

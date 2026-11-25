import unittest
from pathlib import Path

from basketball_ai_tracker.vision.hybrid_detector import HybridBasketballDetector, HybridDetectorConfig
from basketball_ai_tracker.vision.models import (
    BoundingBox,
    DetectionLabel,
    ObjectTrack,
    TrackingResult,
    TrackPoint,
)


class HybridBasketballDetectorTest(unittest.TestCase):
    def test_supplements_player_tracks_when_yolo_finds_only_ball(self) -> None:
        primary = _FakeDetector(_tracking_result([_track("ball-001", DetectionLabel.BALL, 3)]), "yolo")
        fallback = _FakeDetector(
            _tracking_result(
                [
                    _track("player-001", DetectionLabel.PLAYER, 8),
                    _track("player-002", DetectionLabel.PLAYER, 6),
                    _track("player-003", DetectionLabel.PLAYER, 4),
                ]
            ),
            "opencv",
        )
        detector = HybridBasketballDetector(
            primary=primary,
            fallback=fallback,
            config=HybridDetectorConfig(max_supplemental_player_tracks=2),
        )

        result = detector.analyze_video(Path("game.mp4"))

        self.assertEqual([track.track_id for track in result.tracks], ["ball-001", "player-001", "player-002"])
        self.assertEqual(result.detector["engine"], "hybrid_yolo_opencv")
        self.assertEqual(result.detector["supplemented_player_tracks"], 2)
        self.assertTrue(any("OpenCV player tracks were added" in warning for warning in result.warnings))


class _FakeDetector:
    def __init__(self, result: TrackingResult, engine: str) -> None:
        self.result = result
        self.engine = engine

    def analyze_video(self, video_path: Path) -> TrackingResult:
        return self.result

    def describe(self) -> dict[str, object]:
        return {"engine": self.engine, "name": f"{self.engine} fake detector"}


def _tracking_result(tracks: list[ObjectTrack]) -> TrackingResult:
    return TrackingResult(
        fps=30,
        frame_count=90,
        width=640,
        height=360,
        duration_seconds=3,
        sampled_frame_count=3,
        detections=[],
        tracks=tracks,
        detector={"engine": "fake"},
    )


def _track(track_id: str, label: DetectionLabel, points: int) -> ObjectTrack:
    return ObjectTrack(
        track_id=track_id,
        label=label,
        points=[
            TrackPoint(
                frame_index=index,
                time_seconds=float(index),
                bbox=BoundingBox(x=10 + index, y=20, width=15, height=30),
                confidence=0.8,
            )
            for index in range(points)
        ],
    )


if __name__ == "__main__":
    unittest.main()

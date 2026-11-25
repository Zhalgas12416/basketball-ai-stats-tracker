import unittest

from basketball_ai_tracker.vision.models import BoundingBox, DetectionLabel, FrameDetection
from basketball_ai_tracker.vision.tracker import CentroidTracker, TrackerConfig


class CentroidTrackerTest(unittest.TestCase):
    def test_keeps_same_player_id_across_nearby_frames(self) -> None:
        tracker = CentroidTracker(TrackerConfig(player_max_distance=30, max_missing_frames=2))

        tracker.update([_detection(0, 0.0, DetectionLabel.PLAYER, 10, 20)])
        tracker.update([_detection(1, 0.1, DetectionLabel.PLAYER, 16, 23)])

        tracks = tracker.finish()

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, "player-001")
        self.assertEqual(tracks[0].observation_count, 2)

    def test_creates_new_track_after_missing_too_long(self) -> None:
        tracker = CentroidTracker(TrackerConfig(player_max_distance=30, max_missing_frames=1))

        tracker.update([_detection(0, 0.0, DetectionLabel.PLAYER, 10, 20)])
        tracker.update([])
        tracker.update([])
        tracker.update([_detection(3, 0.3, DetectionLabel.PLAYER, 12, 21)])

        tracks = tracker.finish()
        track_ids = [track.track_id for track in tracks]

        self.assertEqual(track_ids, ["player-001", "player-002"])

    def test_tracks_ball_and_player_independently(self) -> None:
        tracker = CentroidTracker(TrackerConfig(player_max_distance=30, ball_max_distance=20))

        tracker.update(
            [
                _detection(0, 0.0, DetectionLabel.PLAYER, 100, 100),
                _detection(0, 0.0, DetectionLabel.BALL, 40, 40, size=12),
            ]
        )
        tracker.update(
            [
                _detection(1, 0.1, DetectionLabel.PLAYER, 106, 102),
                _detection(1, 0.1, DetectionLabel.BALL, 48, 44, size=12),
            ]
        )

        tracks = tracker.finish()

        self.assertEqual([track.track_id for track in tracks], ["ball-001", "player-001"])
        self.assertTrue(all(track.observation_count == 2 for track in tracks))


def _detection(
    frame_index: int,
    time_seconds: float,
    label: DetectionLabel,
    x: float,
    y: float,
    size: float = 40,
) -> FrameDetection:
    return FrameDetection(
        frame_index=frame_index,
        time_seconds=time_seconds,
        label=label,
        bbox=BoundingBox(x=x, y=y, width=size, height=size * 2 if label == DetectionLabel.PLAYER else size),
        confidence=0.8,
    )


if __name__ == "__main__":
    unittest.main()


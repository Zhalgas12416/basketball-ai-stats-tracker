from __future__ import annotations

import unittest

from basketball_ai_tracker.vision.hard_frame_sampler import select_hard_frame_candidates


class HardFrameSamplerTest(unittest.TestCase):
    def test_selects_frames_when_target_classes_are_missing(self) -> None:
        candidates = select_hard_frame_candidates(
            {
                "fps": 30,
                "frame_count": 120,
                "duration_seconds": 4,
                "detections": [
                    {"frame_index": 0, "time_seconds": 0.0, "label": "player", "confidence": 0.9},
                    {"frame_index": 30, "time_seconds": 1.0, "label": "player", "confidence": 0.8},
                ],
            },
            target_classes=["ball", "referee"],
            max_frames=2,
            sample_interval_seconds=1.0,
            min_spacing_seconds=1.0,
        )

        self.assertEqual(len(candidates), 2)
        self.assertTrue(any("missing_ball" in candidate.reasons for candidate in candidates))
        self.assertTrue(any("missing_referee" in candidate.reasons for candidate in candidates))

    def test_scores_low_confidence_target_detection(self) -> None:
        candidates = select_hard_frame_candidates(
            {
                "fps": 30,
                "frame_count": 90,
                "duration_seconds": 3,
                "detections": [
                    {"frame_index": 30, "time_seconds": 1.0, "label": "ball", "confidence": 0.1},
                    {"frame_index": 30, "time_seconds": 1.0, "label": "referee", "confidence": 0.9},
                ],
            },
            target_classes=["ball"],
            max_frames=10,
            sample_interval_seconds=1.0,
            low_confidence_threshold=0.35,
        )

        self.assertTrue(any("low_confidence_ball" in candidate.reasons for candidate in candidates))

    def test_enforces_minimum_spacing(self) -> None:
        candidates = select_hard_frame_candidates(
            {
                "fps": 10,
                "frame_count": 100,
                "duration_seconds": 10,
                "detections": [],
            },
            target_classes=["ball"],
            max_frames=10,
            sample_interval_seconds=0.5,
            min_spacing_seconds=2.0,
        )

        self.assertGreaterEqual(len(candidates), 4)
        for first, second in zip(candidates, candidates[1:]):
            self.assertGreaterEqual(second.timestamp_seconds - first.timestamp_seconds, 2.0)


if __name__ == "__main__":
    unittest.main()

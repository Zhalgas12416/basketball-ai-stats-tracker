from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from basketball_ai_tracker.vision.quality_report import (
    build_detection_quality_report,
    build_detection_quality_report_from_result_path,
    compare_detection_quality_reports,
)


class DetectionQualityReportTest(unittest.TestCase):
    def test_report_counts_classes_empty_frames_and_low_confidence(self) -> None:
        report = build_detection_quality_report(
            {
                "sampled_frame_count": 5,
                "detections": [
                    {"frame_index": 0, "time_seconds": 0.0, "label": "player", "confidence": 0.8},
                    {"frame_index": 0, "time_seconds": 0.0, "label": "ball", "confidence": 0.2},
                    {"frame_index": 2, "time_seconds": 1.0, "label": "rim", "confidence": 0.29},
                    {"frame_index": 4, "time_seconds": 2.0, "label": "referee", "confidence": 0.9},
                ],
            },
            low_confidence_threshold=0.3,
        )

        self.assertEqual(report["class_counts"]["player"], 1)
        self.assertEqual(report["class_counts"]["ball"], 1)
        self.assertEqual(report["class_counts"]["rim"], 1)
        self.assertEqual(report["class_counts"]["backboard"], 0)
        self.assertEqual(report["class_counts"]["referee"], 1)
        self.assertEqual(report["detected_frame_count"], 3)
        self.assertEqual(report["empty_sampled_frame_count"], 2)
        self.assertEqual(report["low_confidence_count"], 2)
        self.assertEqual(report["low_confidence_by_class"]["ball"], 1)

    def test_report_uses_exact_skipped_frame_numbers_when_sampled_frames_exist(self) -> None:
        report = build_detection_quality_report(
            {
                "sampled_frame_count": 4,
                "sampled_frames": [{"frame_index": 0}, {"frame_index": 3}, {"frame_index": 6}, {"frame_index": 9}],
                "detections": [
                    {"frame_index": 0, "time_seconds": 0.0, "label": "player", "confidence": 0.8},
                    {"frame_index": 6, "time_seconds": 1.0, "label": "player", "confidence": 0.8},
                ],
            },
        )

        self.assertEqual(report["skipped_frame_numbers"], [3, 9])
        self.assertIn("exact", report["skipped_frame_note"])

    def test_compare_reports_returns_before_after_deltas(self) -> None:
        before = build_detection_quality_report(
            {
                "sampled_frame_count": 3,
                "detections": [
                    {"frame_index": 0, "label": "player", "confidence": 0.2},
                    {"frame_index": 1, "label": "ball", "confidence": 0.6},
                ],
            },
        )
        after = build_detection_quality_report(
            {
                "sampled_frame_count": 3,
                "detections": [
                    {"frame_index": 0, "label": "player", "confidence": 0.7},
                    {"frame_index": 1, "label": "ball", "confidence": 0.6},
                    {"frame_index": 2, "label": "rim", "confidence": 0.8},
                ],
            },
        )

        comparison = compare_detection_quality_reports(before, after)

        self.assertEqual(comparison["detection_count_delta"], 1)
        self.assertEqual(comparison["detected_frame_count_delta"], 1)
        self.assertEqual(comparison["empty_sampled_frame_count_delta"], -1)
        self.assertEqual(comparison["low_confidence_count_delta"], -1)
        self.assertEqual(comparison["class_count_deltas"]["rim"], 1)

    def test_build_report_from_result_path_reads_vision_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "vision": {
                            "sampled_frame_count": 1,
                            "detections": [{"frame_index": 0, "label": "player", "confidence": 0.9}],
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = build_detection_quality_report_from_result_path(result_path)

        self.assertEqual(report["class_counts"]["player"], 1)


if __name__ == "__main__":
    unittest.main()

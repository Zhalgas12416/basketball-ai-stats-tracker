from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from basketball_ai_tracker.vision.correction_exporter import (
    CLASS_NAMES,
    bbox_to_yolo,
    find_video_for_game,
    group_corrections_by_timestamp,
    load_corrections,
    write_dataset_yaml,
)


class CorrectionExporterTest(unittest.TestCase):
    def test_class_order_matches_detector_config(self) -> None:
        self.assertEqual(CLASS_NAMES, ["player", "ball", "rim", "backboard", "referee"])

    def test_bbox_to_yolo_normalizes_xywh_box(self) -> None:
        yolo_box = bbox_to_yolo(
            {"x": 10, "y": 20, "width": 30, "height": 40},
            image_width=100,
            image_height=200,
            class_id=1,
        )

        self.assertEqual(yolo_box.class_id, 1)
        self.assertAlmostEqual(yolo_box.x_center, 0.25)
        self.assertAlmostEqual(yolo_box.y_center, 0.2)
        self.assertAlmostEqual(yolo_box.width, 0.3)
        self.assertAlmostEqual(yolo_box.height, 0.2)
        self.assertEqual(yolo_box.to_line(), "1 0.250000 0.200000 0.300000 0.200000")

    def test_bbox_to_yolo_clips_to_image_bounds(self) -> None:
        yolo_box = bbox_to_yolo(
            {"x": -10, "y": 80, "width": 40, "height": 40},
            image_width=100,
            image_height=100,
            class_id=0,
        )

        self.assertAlmostEqual(yolo_box.x_center, 0.15)
        self.assertAlmostEqual(yolo_box.y_center, 0.9)
        self.assertAlmostEqual(yolo_box.width, 0.3)
        self.assertAlmostEqual(yolo_box.height, 0.2)

    def test_group_corrections_by_timestamp_uses_millisecond_precision(self) -> None:
        grouped = group_corrections_by_timestamp(
            [
                {"video_time_seconds": 1.2344, "label": "ball"},
                {"video_time_seconds": 1.23449, "label": "rim"},
                {"video_time_seconds": 1.236, "label": "player"},
            ],
        )

        self.assertEqual(len(grouped[1.234]), 2)
        self.assertEqual(len(grouped[1.236]), 1)

    def test_load_corrections_reads_payload_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.json"
            path.write_text(
                json.dumps({"corrections": [{"label": "player"}]}),
                encoding="utf-8",
            )

            self.assertEqual(load_corrections(path), [{"label": "player"}])

    def test_find_video_for_game_uses_known_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            upload_dir = Path(directory)
            video_path = upload_dir / "abc123.mov"
            video_path.write_bytes(b"fake")

            self.assertEqual(find_video_for_game("abc123", upload_dir), video_path)

    def test_write_dataset_yaml_creates_yolo_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            yaml_path = write_dataset_yaml(Path(directory))

            text = yaml_path.read_text(encoding="utf-8")
            self.assertIn("train: images/train", text)
            self.assertIn("val: images/val", text)
            self.assertIn("0: player", text)
            self.assertIn("4: referee", text)


if __name__ == "__main__":
    unittest.main()

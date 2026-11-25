import unittest

from basketball_ai_tracker.vision.models import DetectionLabel
from basketball_ai_tracker.vision.yolo_detector import (
    YOLOBasketballDetector,
    YOLODetectorConfig,
    normalize_yolo_class_name,
)


class YOLODetectorMappingTest(unittest.TestCase):
    def test_maps_common_class_names_to_internal_labels(self) -> None:
        self.assertEqual(normalize_yolo_class_name("person"), DetectionLabel.PLAYER)
        self.assertEqual(normalize_yolo_class_name("sports ball"), DetectionLabel.BALL)
        self.assertEqual(normalize_yolo_class_name("basketball"), DetectionLabel.BALL)
        self.assertEqual(normalize_yolo_class_name("back-board"), DetectionLabel.BACKBOARD)
        self.assertEqual(normalize_yolo_class_name("unknown"), None)

    def test_uses_lower_ball_confidence_without_lowering_other_classes(self) -> None:
        model = _FakeYOLOModel()
        detector = YOLOBasketballDetector(
            YOLODetectorConfig(confidence=0.15, ball_confidence=0.08)
        )

        detections = detector._predict_frame(model, object(), frame_index=12, time_seconds=0.4)

        self.assertEqual(model.predict_confidence, 0.08)
        self.assertEqual([detection.label for detection in detections], [DetectionLabel.BALL, DetectionLabel.PLAYER])
        self.assertEqual([round(detection.confidence, 2) for detection in detections], [0.09, 0.16])


class _FakeBoxes:
    cls = [0, 2, 2]
    conf = [0.09, 0.10, 0.16]
    xyxy = [
        [10.0, 20.0, 15.0, 25.0],
        [30.0, 40.0, 60.0, 90.0],
        [50.0, 60.0, 80.0, 120.0],
    ]


class _FakeResult:
    names = {0: "ball", 2: "player"}
    boxes = _FakeBoxes()


class _FakeYOLOModel:
    def __init__(self) -> None:
        self.predict_confidence = None

    def predict(self, frame, *, conf, iou, imgsz, verbose):
        self.predict_confidence = conf
        return [_FakeResult()]


if __name__ == "__main__":
    unittest.main()

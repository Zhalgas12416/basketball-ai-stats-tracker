from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from basketball_ai_tracker.vision.models import (
    BoundingBox,
    DetectionLabel,
    FrameDetection,
    TrackingResult,
)
from basketball_ai_tracker.vision.opencv_detector import (
    OpenCVBasketballDetector,
    _crop_frame,
    _translate_detections,
)
from basketball_ai_tracker.vision.tracker import CentroidTracker, TrackerConfig


CLASS_ALIASES = {
    "athlete": DetectionLabel.PLAYER,
    "basketball player": DetectionLabel.PLAYER,
    "basketballplayer": DetectionLabel.PLAYER,
    "human": DetectionLabel.PLAYER,
    "person": DetectionLabel.PLAYER,
    "people": DetectionLabel.PLAYER,
    "player": DetectionLabel.PLAYER,
    "players": DetectionLabel.PLAYER,
    "sports ball": DetectionLabel.BALL,
    "sportsball": DetectionLabel.BALL,
    "ball": DetectionLabel.BALL,
    "basketball": DetectionLabel.BALL,
    "rim": DetectionLabel.RIM,
    "hoop": DetectionLabel.RIM,
    "basket": DetectionLabel.RIM,
    "basketball hoop": DetectionLabel.RIM,
    "basketball rim": DetectionLabel.RIM,
    "backboard": DetectionLabel.BACKBOARD,
    "back board": DetectionLabel.BACKBOARD,
    "basketball backboard": DetectionLabel.BACKBOARD,
    "referee": DetectionLabel.REFEREE,
    "ref": DetectionLabel.REFEREE,
    "official": DetectionLabel.REFEREE,
}


@dataclass(frozen=True)
class YOLODetectorConfig:
    model_path: str | Path = "models/basketball-detector.pt"
    sample_fps: float = 10.0
    max_frames: int = 1800
    confidence: float = 0.15
    ball_confidence: float = 0.08
    iou: float = 0.5
    imgsz: int = 960
    min_track_observations: int = 2
    max_tracks: int = 200
    auto_crop: bool = True


class YOLOBasketballDetector:
    def __init__(
        self,
        config: YOLODetectorConfig | None = None,
        tracker_config: TrackerConfig | None = None,
    ) -> None:
        self.config = config or YOLODetectorConfig()
        self.tracker_config = tracker_config or TrackerConfig()
        self._model = None

    def analyze_video(self, video_path: Path) -> TrackingResult:
        cv2 = _load_cv2()
        capture = cv2.VideoCapture(str(video_path))
        warnings: list[str] = []

        if not capture.isOpened():
            return TrackingResult(
                fps=0.0,
                frame_count=0,
                width=0,
                height=0,
                duration_seconds=0.0,
                sampled_frame_count=0,
                detections=[],
                tracks=[],
                analysis_region=None,
                warnings=[f"OpenCV could not read {video_path.name}."],
                detector=self.describe(),
            )

        model = self._load_model()
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        stride = max(1, int(round((fps or self.config.sample_fps) / self.config.sample_fps)))
        analysis_region = None
        if self.config.auto_crop:
            analysis_region = OpenCVBasketballDetector()._detect_analysis_region(cv2, capture)
            if analysis_region is not None:
                warnings.append("Auto-cropped analysis to the likely game-video region inside the uploaded frame.")

        tracker = CentroidTracker(self.tracker_config)
        all_detections: list[FrameDetection] = []
        sampled_frames = 0
        frame_index = 0

        try:
            while sampled_frames < self.config.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % stride != 0:
                    frame_index += 1
                    continue

                time_seconds = frame_index / fps if fps > 0 else float(sampled_frames)
                frame_for_detection = _crop_frame(frame, analysis_region)
                detections = self._predict_frame(model, frame_for_detection, frame_index, time_seconds)
                detections = _translate_detections(detections, analysis_region)
                tracker.update(detections)
                all_detections.extend(detections)
                sampled_frames += 1
                frame_index += 1
        finally:
            capture.release()

        if sampled_frames >= self.config.max_frames:
            warnings.append(f"Stopped after {self.config.max_frames} sampled frames.")

        tracks = self._filter_tracks(tracker.finish(), warnings)

        return TrackingResult(
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_seconds=duration,
            sampled_frame_count=sampled_frames,
            detections=all_detections,
            tracks=tracks,
            analysis_region=analysis_region,
            warnings=warnings,
            detector=self.describe(model),
        )

    def describe(self, model=None) -> dict[str, object]:
        model_path = Path(self.config.model_path).expanduser()
        loaded_model = model or self._model
        return {
            "engine": "yolo",
            "name": "YOLO basketball detector",
            "model_path": str(model_path.resolve() if model_path.exists() else model_path),
            "model_exists": model_path.exists(),
            "class_names": _model_class_names(loaded_model),
            "sample_fps": self.config.sample_fps,
            "max_frames": self.config.max_frames,
            "confidence": self.config.confidence,
            "ball_confidence": self.config.ball_confidence,
            "iou": self.config.iou,
            "imgsz": self.config.imgsz,
            "min_track_observations": self.config.min_track_observations,
            "max_tracks": self.config.max_tracks,
            "auto_crop": self.config.auto_crop,
        }

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install training dependencies with: pip install -e '.[train]'") from exc
        self._model = YOLO(str(self.config.model_path))
        return self._model

    def _predict_frame(self, model, frame, frame_index: int, time_seconds: float) -> list[FrameDetection]:
        prediction_confidence = min(self.config.confidence, self.config.ball_confidence)
        results = model.predict(
            frame,
            conf=prediction_confidence,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []

        detections: list[FrameDetection] = []
        for cls_tensor, conf_tensor, xyxy_tensor in zip(boxes.cls, boxes.conf, boxes.xyxy):
            source_name = result.names[int(cls_tensor)]
            label = normalize_yolo_class_name(source_name)
            if label is None:
                continue
            minimum_confidence = (
                self.config.ball_confidence
                if label is DetectionLabel.BALL
                else self.config.confidence
            )
            if float(conf_tensor) < minimum_confidence:
                continue
            x1, y1, x2, y2 = [float(value) for value in xyxy_tensor]
            detections.append(
                FrameDetection(
                    frame_index=frame_index,
                    time_seconds=time_seconds,
                    label=label,
                    bbox=BoundingBox(x=x1, y=y1, width=max(0.0, x2 - x1), height=max(0.0, y2 - y1)),
                    confidence=float(conf_tensor),
                )
            )
        return detections

    def _filter_tracks(self, tracks, warnings: list[str]):
        filtered = [
            track
            for track in tracks
            if track.observation_count >= self.config.min_track_observations
        ]
        if len(filtered) > self.config.max_tracks:
            warnings.append(f"Kept the top {self.config.max_tracks} tracks by observation count.")
            filtered = sorted(
                filtered,
                key=lambda track: (track.observation_count, track.max_confidence),
                reverse=True,
            )[: self.config.max_tracks]
        return sorted(filtered, key=lambda track: (track.label.value, track.first_seen_seconds, track.track_id))


def normalize_yolo_class_name(name: str) -> DetectionLabel | None:
    normalized = name.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return CLASS_ALIASES.get(normalized)


def _model_class_names(model) -> dict[str, str]:
    if model is None:
        return {}
    names = getattr(model, "names", {}) or {}
    if isinstance(names, dict):
        return {str(key): str(value) for key, value in sorted(names.items(), key=lambda item: str(item[0]))}
    return {str(index): str(value) for index, value in enumerate(names)}


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for video inference. Install with: pip install -e '.[cv]'") from exc
    return cv2

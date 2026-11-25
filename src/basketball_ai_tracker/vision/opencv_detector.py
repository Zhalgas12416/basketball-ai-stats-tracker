from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from basketball_ai_tracker.vision.models import (
    BoundingBox,
    DetectionLabel,
    FrameDetection,
    TrackingResult,
)
from basketball_ai_tracker.vision.tracker import CentroidTracker, TrackerConfig


@dataclass(frozen=True)
class OpenCVDetectorConfig:
    sample_fps: float = 8.0
    max_frames: int = 1200
    warmup_sampled_frames: int = 8
    min_player_height_ratio: float = 0.07
    max_player_height_ratio: float = 0.82
    min_player_area_ratio: float = 0.0008
    max_player_area_ratio: float = 0.12
    min_ball_area: float = 18.0
    max_ball_area_ratio: float = 0.001
    max_player_candidates_per_frame: int = 18
    max_ball_candidates_per_frame: int = 4
    min_player_track_observations: int = 3
    min_ball_track_observations: int = 4
    max_tracks: int = 120
    nms_iou_threshold: float = 0.35


class OpenCVBasketballDetector:
    def __init__(
        self,
        config: OpenCVDetectorConfig | None = None,
        tracker_config: TrackerConfig | None = None,
    ) -> None:
        self.config = config or OpenCVDetectorConfig()
        self.tracker_config = tracker_config or TrackerConfig()

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

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        stride = max(1, int(round((fps or self.config.sample_fps) / self.config.sample_fps)))
        subtractor = cv2.createBackgroundSubtractorMOG2(history=180, varThreshold=24, detectShadows=True)
        tracker = CentroidTracker(self.tracker_config)
        all_detections: list[FrameDetection] = []
        sampled_frames = 0
        frame_index = 0
        analysis_region = self._detect_analysis_region(cv2, capture)
        if analysis_region is not None:
            warnings.append(
                "Auto-cropped analysis to the likely game-video region inside the uploaded frame."
            )

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
                detections = self._detect_frame(cv2, subtractor, frame_for_detection, frame_index, time_seconds)
                detections = _translate_detections(detections, analysis_region)
                if sampled_frames >= self.config.warmup_sampled_frames:
                    tracker.update(detections)
                    all_detections.extend(detections)
                sampled_frames += 1
                frame_index += 1
        finally:
            capture.release()

        if sampled_frames >= self.config.max_frames:
            warnings.append(
                f"Stopped after {self.config.max_frames} sampled frames; lower sample_fps or increase max_frames for full-game runs."
            )

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
            detector=self.describe(),
        )

    def describe(self) -> dict[str, object]:
        return {
            "engine": "opencv",
            "name": "OpenCV motion/color detector",
            "model_path": None,
            "sample_fps": self.config.sample_fps,
            "max_frames": self.config.max_frames,
            "min_player_track_observations": self.config.min_player_track_observations,
            "min_ball_track_observations": self.config.min_ball_track_observations,
        }

    def _detect_frame(
        self,
        cv2,
        subtractor,
        frame,
        frame_index: int,
        time_seconds: float,
    ) -> list[FrameDetection]:
        height, width = frame.shape[:2]
        foreground = subtractor.apply(frame)
        foreground = cv2.threshold(foreground, 220, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = [
            detection
            for contour in contours
            for detection in self._classify_motion_contour(cv2, contour, frame_index, time_seconds, width, height)
        ]
        detections.extend(self._detect_orange_ball_candidates(cv2, frame, frame_index, time_seconds))
        detections = _non_max_suppression(detections, self.config.nms_iou_threshold)
        return self._cap_frame_detections(detections)

    def _classify_motion_contour(
        self,
        cv2,
        contour,
        frame_index: int,
        time_seconds: float,
        frame_width: int,
        frame_height: int,
    ) -> list[FrameDetection]:
        x, y, width, height = cv2.boundingRect(contour)
        bbox = BoundingBox(float(x), float(y), float(width), float(height))
        area = bbox.area
        frame_area = max(1.0, float(frame_width * frame_height))
        area_ratio = area / frame_area
        height_ratio = height / max(1.0, float(frame_height))
        aspect = height / max(1.0, float(width))
        contour_area = float(cv2.contourArea(contour))
        extent = contour_area / area if area > 0 else 0.0
        detections: list[FrameDetection] = []

        if (
            self.config.min_player_area_ratio <= area_ratio <= self.config.max_player_area_ratio
            and self.config.min_player_height_ratio <= height_ratio <= self.config.max_player_height_ratio
            and aspect >= 1.05
            and extent >= 0.22
        ):
            confidence = min(0.92, 0.35 + height_ratio + extent * 0.35)
            detections.append(FrameDetection(frame_index, time_seconds, DetectionLabel.PLAYER, bbox, confidence))

        max_ball_area = self.config.max_ball_area_ratio * frame_area
        if self.config.min_ball_area <= area <= max_ball_area and 0.6 <= aspect <= 1.65:
            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0
            if perimeter > 0:
                circularity = 4.0 * 3.14159 * contour_area / (perimeter * perimeter)
            if circularity >= 0.45:
                confidence = min(0.82, 0.3 + circularity * 0.5)
                detections.append(FrameDetection(frame_index, time_seconds, DetectionLabel.BALL, bbox, confidence))

        return detections

    def _detect_orange_ball_candidates(self, cv2, frame, frame_index: int, time_seconds: float) -> list[FrameDetection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_orange = (4, 65, 45)
        upper_orange = (28, 255, 255)
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = max(1.0, float(frame.shape[0] * frame.shape[1]))
        max_ball_area = self.config.max_ball_area_ratio * frame_area
        detections: list[FrameDetection] = []

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            bbox = BoundingBox(float(x), float(y), float(width), float(height))
            area = bbox.area
            aspect = width / max(1.0, float(height))
            if not (self.config.min_ball_area <= area <= max_ball_area and 0.65 <= aspect <= 1.55):
                continue
            contour_area = float(cv2.contourArea(contour))
            fill_ratio = contour_area / area if area > 0 else 0.0
            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0
            if perimeter > 0:
                circularity = 4.0 * 3.14159 * contour_area / (perimeter * perimeter)
            if fill_ratio < 0.34 or circularity < 0.45:
                continue
            confidence = min(0.78, 0.34 + fill_ratio * 0.28 + circularity * 0.28)
            detections.append(FrameDetection(frame_index, time_seconds, DetectionLabel.BALL, bbox, confidence))

        return detections

    def _cap_frame_detections(self, detections: list[FrameDetection]) -> list[FrameDetection]:
        players = [detection for detection in detections if detection.label == DetectionLabel.PLAYER]
        balls = [detection for detection in detections if detection.label == DetectionLabel.BALL]
        players = sorted(players, key=lambda item: (item.confidence, item.bbox.area), reverse=True)[
            : self.config.max_player_candidates_per_frame
        ]
        balls = sorted(balls, key=lambda item: item.confidence, reverse=True)[
            : self.config.max_ball_candidates_per_frame
        ]
        return sorted(players + balls, key=lambda item: (item.label.value, item.bbox.x, item.bbox.y))

    def _filter_tracks(self, tracks, warnings: list[str]):
        filtered = []
        for track in tracks:
            if track.label == DetectionLabel.BALL:
                if track.observation_count < self.config.min_ball_track_observations:
                    continue
            elif track.observation_count < self.config.min_player_track_observations:
                continue
            filtered.append(track)

        if len(filtered) > self.config.max_tracks:
            warnings.append(f"Kept the top {self.config.max_tracks} tracks by observation count.")
            filtered = sorted(
                filtered,
                key=lambda track: (track.observation_count, track.max_confidence),
                reverse=True,
            )[: self.config.max_tracks]

        return sorted(filtered, key=lambda track: (track.label.value, track.first_seen_seconds, track.track_id))

    def _detect_analysis_region(self, cv2, capture) -> BoundingBox | None:
        current_position = capture.get(cv2.CAP_PROP_POS_FRAMES)
        ok, frame = capture.read()
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_position)
        if not ok:
            return None

        frame_height, frame_width = frame.shape[:2]
        projection_region = _detect_embedded_video_region_by_projection(cv2, frame)
        if projection_region is not None:
            return projection_region

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        mask = ((gray < 232) & ((saturation > 18) | (gray < 185))).astype("uint8") * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (39, 39))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[tuple[float, BoundingBox]] = []
        frame_area = float(frame_width * frame_height)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area_ratio = (width * height) / frame_area
            aspect = width / max(1.0, float(height))
            if not (0.18 <= area_ratio <= 0.82):
                continue
            if not (1.15 <= aspect <= 2.35):
                continue
            if width < frame_width * 0.42 or height < frame_height * 0.26:
                continue
            if y < frame_height * 0.12:
                continue
            candidates.append((area_ratio, BoundingBox(float(x), float(y), float(width), float(height))))

        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]


def _non_max_suppression(detections: list[FrameDetection], iou_threshold: float) -> list[FrameDetection]:
    kept: list[FrameDetection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if any(detection.label == existing.label and detection.bbox.iou(existing.bbox) > iou_threshold for existing in kept):
            continue
        kept.append(detection)
    return sorted(kept, key=lambda item: (item.label.value, item.bbox.x, item.bbox.y))


def _crop_frame(frame, analysis_region: BoundingBox | None):
    if analysis_region is None:
        return frame
    x = int(max(0, analysis_region.x))
    y = int(max(0, analysis_region.y))
    right = int(max(x + 1, analysis_region.x + analysis_region.width))
    bottom = int(max(y + 1, analysis_region.y + analysis_region.height))
    return frame[y:bottom, x:right]


def _translate_detections(
    detections: list[FrameDetection],
    analysis_region: BoundingBox | None,
) -> list[FrameDetection]:
    if analysis_region is None:
        return detections
    return [
        FrameDetection(
            frame_index=detection.frame_index,
            time_seconds=detection.time_seconds,
            label=detection.label,
            bbox=detection.bbox.translated(analysis_region.x, analysis_region.y),
            confidence=detection.confidence,
        )
        for detection in detections
    ]


def _detect_embedded_video_region_by_projection(cv2, frame) -> BoundingBox | None:
    frame_height, frame_width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_mask = (gray < 190).astype("uint8")
    left_content_width = int(frame_width * 0.72)
    row_ratios = dark_mask[:, :left_content_width].mean(axis=1)
    row_segments = _segments_above_threshold(row_ratios, threshold=0.35, min_length=int(frame_height * 0.18))

    candidate_rows = [
        segment
        for segment in row_segments
        if segment[0] >= int(frame_height * 0.12) and segment[1] <= int(frame_height * 0.9)
    ]
    if not candidate_rows:
        return None

    row_start, row_end = max(candidate_rows, key=lambda segment: segment[1] - segment[0])
    col_ratios = dark_mask[row_start:row_end, :].mean(axis=0)
    col_segments = _segments_above_threshold(col_ratios, threshold=0.35, min_length=int(frame_width * 0.25))
    candidates: list[BoundingBox] = []

    for col_start, col_end in col_segments:
        width = col_end - col_start
        height = row_end - row_start
        aspect = width / max(1.0, float(height))
        area_ratio = (width * height) / float(frame_width * frame_height)
        if not (1.25 <= aspect <= 2.25):
            continue
        if not (0.16 <= area_ratio <= 0.75):
            continue
        if col_start > frame_width * 0.25:
            continue
        candidates.append(BoundingBox(float(col_start), float(row_start), float(width), float(height)))

    if not candidates:
        return None
    return max(candidates, key=lambda bbox: bbox.area)


def _segments_above_threshold(values, threshold: float, min_length: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value > threshold and start is None:
            start = index
        elif (value <= threshold or index == len(values) - 1) and start is not None:
            end = index if value <= threshold else index + 1
            if end - start >= min_length:
                segments.append((start, end))
            start = None
    return segments


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the real detector. Install with: pip install -e '.[cv]'") from exc
    return cv2

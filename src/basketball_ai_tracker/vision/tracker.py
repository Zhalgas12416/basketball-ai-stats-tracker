from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from basketball_ai_tracker.vision.models import DetectionLabel, FrameDetection, ObjectTrack, TrackPoint


@dataclass
class TrackerConfig:
    max_missing_frames: int = 8
    player_max_distance: float = 90.0
    ball_max_distance: float = 70.0


@dataclass
class _ActiveTrack:
    track: ObjectTrack
    missing_frames: int = 0


class CentroidTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._next_ids = {label: 1 for label in DetectionLabel}
        self._active: dict[str, _ActiveTrack] = {}
        self._finished: list[ObjectTrack] = []

    def update(self, detections: list[FrameDetection]) -> None:
        remaining_detections = set(range(len(detections)))
        matched_tracks: set[str] = set()
        candidates: list[tuple[float, str, int]] = []

        for track_id, active in self._active.items():
            if not active.track.points:
                continue
            last_point = active.track.points[-1]
            for index, detection in enumerate(detections):
                if detection.label != active.track.label:
                    continue
                distance = _distance(last_point.bbox.center, detection.bbox.center)
                if distance <= self._max_distance(detection.label):
                    candidates.append((distance, track_id, index))

        for _, track_id, detection_index in sorted(candidates, key=lambda item: item[0]):
            if track_id in matched_tracks or detection_index not in remaining_detections:
                continue
            self._append_detection(self._active[track_id], detections[detection_index])
            matched_tracks.add(track_id)
            remaining_detections.remove(detection_index)

        for track_id, active in list(self._active.items()):
            if track_id in matched_tracks:
                continue
            active.missing_frames += 1
            if active.missing_frames > self.config.max_missing_frames:
                self._finished.append(active.track)
                del self._active[track_id]

        for detection_index in sorted(remaining_detections):
            detection = detections[detection_index]
            track_id = self._new_track_id(detection.label)
            track = ObjectTrack(track_id=track_id, label=detection.label)
            active = _ActiveTrack(track=track)
            self._append_detection(active, detection)
            self._active[track_id] = active

    def finish(self) -> list[ObjectTrack]:
        tracks = self._finished + [active.track for active in self._active.values()]
        return sorted(tracks, key=lambda track: (track.label.value, track.first_seen_seconds, track.track_id))

    def _append_detection(self, active: _ActiveTrack, detection: FrameDetection) -> None:
        active.track.points.append(
            TrackPoint(
                frame_index=detection.frame_index,
                time_seconds=detection.time_seconds,
                bbox=detection.bbox,
                confidence=detection.confidence,
            )
        )
        active.missing_frames = 0

    def _new_track_id(self, label: DetectionLabel) -> str:
        next_id = self._next_ids[label]
        self._next_ids[label] += 1
        prefix = label.value
        return f"{prefix}-{next_id:03d}"

    def _max_distance(self, label: DetectionLabel) -> float:
        if label == DetectionLabel.BALL:
            return self.config.ball_max_distance
        return self.config.player_max_distance


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])

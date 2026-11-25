from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class DetectionLabel(StrEnum):
    PLAYER = "player"
    BALL = "ball"
    RIM = "rim"
    BACKBOARD = "backboard"
    REFEREE = "referee"


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def iou(self, other: BoundingBox) -> float:
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.x + self.width, other.x + other.width)
        bottom = min(self.y + self.height, other.y + other.height)
        overlap_width = max(0.0, right - left)
        overlap_height = max(0.0, bottom - top)
        intersection = overlap_width * overlap_height
        union = self.area + other.area - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def translated(self, offset_x: float, offset_y: float) -> BoundingBox:
        return BoundingBox(
            x=self.x + offset_x,
            y=self.y + offset_y,
            width=self.width,
            height=self.height,
        )


@dataclass(frozen=True)
class FrameDetection:
    frame_index: int
    time_seconds: float
    label: DetectionLabel
    bbox: BoundingBox
    confidence: float


@dataclass(frozen=True)
class TrackPoint:
    frame_index: int
    time_seconds: float
    bbox: BoundingBox
    confidence: float


@dataclass
class ObjectTrack:
    track_id: str
    label: DetectionLabel
    points: list[TrackPoint] = field(default_factory=list)

    @property
    def first_seen_seconds(self) -> float:
        return self.points[0].time_seconds if self.points else 0.0

    @property
    def last_seen_seconds(self) -> float:
        return self.points[-1].time_seconds if self.points else 0.0

    @property
    def max_confidence(self) -> float:
        if not self.points:
            return 0.0
        return max(point.confidence for point in self.points)

    @property
    def observation_count(self) -> int:
        return len(self.points)


@dataclass
class TrackingResult:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float
    sampled_frame_count: int
    detections: list[FrameDetection]
    tracks: list[ObjectTrack]
    analysis_region: BoundingBox | None = None
    warnings: list[str] = field(default_factory=list)
    detector: dict[str, object] = field(default_factory=dict)

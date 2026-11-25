from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class EventType(StrEnum):
    FREE_THROW_MADE = "free_throw_made"
    FREE_THROW_MISSED = "free_throw_missed"
    TWO_POINTER_MADE = "two_pointer_made"
    TWO_POINTER_MISSED = "two_pointer_missed"
    THREE_POINTER_MADE = "three_pointer_made"
    THREE_POINTER_MISSED = "three_pointer_missed"
    ASSIST = "assist"
    OFFENSIVE_REBOUND = "offensive_rebound"
    DEFENSIVE_REBOUND = "defensive_rebound"
    STEAL = "steal"
    TURNOVER = "turnover"
    BLOCK = "block"
    FOUL = "foul"


@dataclass(frozen=True)
class PlayerIdentity:
    player_id: str
    team: str
    jersey_number: str
    display_name: str | None = None

    @property
    def label(self) -> str:
        if self.display_name:
            return self.display_name
        return f"{self.team} #{self.jersey_number}"


@dataclass
class StatEvent:
    event_id: str
    event_type: EventType
    player_id: str
    period: int
    video_time_seconds: float
    game_clock: str | None = None
    points: int = 0
    related_player_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = "manual"
    notes: str | None = None


@dataclass
class PlayerStats:
    player: PlayerIdentity
    points: int = 0
    assists: int = 0
    offensive_rebounds: int = 0
    defensive_rebounds: int = 0
    steals: int = 0
    turnovers: int = 0
    blocks: int = 0
    fouls: int = 0
    field_goals_made: int = 0
    field_goals_attempted: int = 0
    three_points_made: int = 0
    three_points_attempted: int = 0
    free_throws_made: int = 0
    free_throws_attempted: int = 0

    @property
    def rebounds(self) -> int:
        return self.offensive_rebounds + self.defensive_rebounds


@dataclass
class AnalysisResult:
    game_id: str
    video_filename: str
    duration_seconds: float
    players: list[PlayerIdentity]
    events: list[StatEvent]
    stats: list[PlayerStats]
    warnings: list[str] = field(default_factory=list)
    vision: dict | None = None

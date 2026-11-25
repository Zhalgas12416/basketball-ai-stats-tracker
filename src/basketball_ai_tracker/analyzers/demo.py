from __future__ import annotations

from pathlib import Path

from basketball_ai_tracker.models import AnalysisResult, EventType, PlayerIdentity, StatEvent
from basketball_ai_tracker.services.stats import build_player_stats


class DemoBasketballAnalyzer:
    """Deterministic analyzer used until detector/tracker models are connected."""

    def __init__(self, warnings: list[str] | None = None) -> None:
        self.warnings = warnings or [
            "Demo analyzer active: replace this with the detector/tracker/action-recognition pipeline for real game stats."
        ]

    def analyze(self, game_id: str, video_path: Path, original_filename: str) -> AnalysisResult:
        duration = _read_duration_seconds(video_path) or 600.0
        players = _demo_players()
        events = _demo_events(duration)
        stats = build_player_stats(players, events)
        return AnalysisResult(
            game_id=game_id,
            video_filename=original_filename,
            duration_seconds=duration,
            players=players,
            events=events,
            stats=stats,
            warnings=self.warnings,
        )

    def describe(self) -> dict[str, object]:
        return {
            "name": "Demo analyzer",
            "detector": {
                "engine": "demo",
                "name": "Deterministic demo data",
                "model_path": None,
            },
            "warnings": self.warnings,
        }


def _read_duration_seconds(video_path: Path) -> float | None:
    try:
        import cv2
    except ImportError:
        return None

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if frames <= 0 or fps <= 0:
            return None
        return float(frames / fps)
    finally:
        capture.release()


def _demo_players() -> list[PlayerIdentity]:
    return [
        PlayerIdentity(player_id="home-01", team="Home", jersey_number="1"),
        PlayerIdentity(player_id="home-07", team="Home", jersey_number="7"),
        PlayerIdentity(player_id="home-13", team="Home", jersey_number="13"),
        PlayerIdentity(player_id="home-21", team="Home", jersey_number="21"),
        PlayerIdentity(player_id="home-34", team="Home", jersey_number="34"),
        PlayerIdentity(player_id="away-03", team="Away", jersey_number="3"),
        PlayerIdentity(player_id="away-10", team="Away", jersey_number="10"),
        PlayerIdentity(player_id="away-14", team="Away", jersey_number="14"),
        PlayerIdentity(player_id="away-24", team="Away", jersey_number="24"),
        PlayerIdentity(player_id="away-32", team="Away", jersey_number="32"),
    ]


def _demo_events(duration: float) -> list[StatEvent]:
    raw_events = [
        (14, EventType.TWO_POINTER_MADE, "home-01", 2, ["home-07"], "Cut layup from left side"),
        (14, EventType.ASSIST, "home-07", 0, ["home-01"], "Pass led directly to made basket"),
        (39, EventType.DEFENSIVE_REBOUND, "away-32", 0, [], "Secured missed jumper"),
        (53, EventType.STEAL, "home-13", 0, ["away-10"], "Deflected wing pass"),
        (53, EventType.TURNOVER, "away-10", 0, ["home-13"], "Live-ball turnover"),
        (74, EventType.THREE_POINTER_MADE, "home-13", 3, ["home-01"], "Transition corner three"),
        (74, EventType.ASSIST, "home-01", 0, ["home-13"], "Kick-ahead pass"),
        (109, EventType.TWO_POINTER_MADE, "away-24", 2, ["away-03"], "Drive finish"),
        (109, EventType.ASSIST, "away-03", 0, ["away-24"], "Paint touch assist"),
        (146, EventType.BLOCK, "home-34", 0, ["away-24"], "Blocked layup attempt"),
        (148, EventType.DEFENSIVE_REBOUND, "home-21", 0, [], "Recovered blocked shot"),
        (183, EventType.THREE_POINTER_MISSED, "away-14", 0, [], "Above-break three missed"),
        (186, EventType.OFFENSIVE_REBOUND, "away-32", 0, [], "Tip-out rebound"),
        (210, EventType.FREE_THROW_MADE, "home-07", 1, [], "First free throw made"),
        (211, EventType.FREE_THROW_MISSED, "home-07", 0, [], "Second free throw missed"),
        (244, EventType.TURNOVER, "home-01", 0, ["away-03"], "Lost dribble under pressure"),
        (244, EventType.STEAL, "away-03", 0, ["home-01"], "Stripped ball handler"),
    ]

    max_time = max(1.0, duration - 1.0)
    events: list[StatEvent] = []
    for index, (seconds, event_type, player_id, points, related, notes) in enumerate(raw_events, start=1):
        event_time = min(float(seconds), max_time)
        events.append(
            StatEvent(
                event_id=f"demo-{index:03d}",
                event_type=event_type,
                player_id=player_id,
                period=1,
                video_time_seconds=event_time,
                points=points,
                related_player_ids=list(related),
                confidence=0.82,
                source="demo",
                notes=notes,
            )
        )
    return events

from __future__ import annotations

from collections.abc import Iterable

from basketball_ai_tracker.models import EventType, PlayerIdentity, PlayerStats, StatEvent


def build_player_stats(
    players: Iterable[PlayerIdentity],
    events: Iterable[StatEvent],
) -> list[PlayerStats]:
    event_list = list(events)
    stats_by_player = {player.player_id: PlayerStats(player=player) for player in players}

    for event in event_list:
        if event.player_id not in stats_by_player:
            stats_by_player[event.player_id] = PlayerStats(
                player=PlayerIdentity(
                    player_id=event.player_id,
                    team="Unknown",
                    jersey_number="?",
                    display_name=f"Unknown player {event.player_id}",
                )
            )

        player_stats = stats_by_player[event.player_id]
        _apply_event(player_stats, event)

    return sorted(
        stats_by_player.values(),
        key=lambda row: (row.player.team, _jersey_sort_key(row.player.jersey_number), row.player.player_id),
    )


def _apply_event(stats: PlayerStats, event: StatEvent) -> None:
    if event.event_type == EventType.FREE_THROW_MADE:
        stats.points += event.points or 1
        stats.free_throws_made += 1
        stats.free_throws_attempted += 1
    elif event.event_type == EventType.FREE_THROW_MISSED:
        stats.free_throws_attempted += 1
    elif event.event_type == EventType.TWO_POINTER_MADE:
        stats.points += event.points or 2
        stats.field_goals_made += 1
        stats.field_goals_attempted += 1
    elif event.event_type == EventType.TWO_POINTER_MISSED:
        stats.field_goals_attempted += 1
    elif event.event_type == EventType.THREE_POINTER_MADE:
        stats.points += event.points or 3
        stats.field_goals_made += 1
        stats.field_goals_attempted += 1
        stats.three_points_made += 1
        stats.three_points_attempted += 1
    elif event.event_type == EventType.THREE_POINTER_MISSED:
        stats.field_goals_attempted += 1
        stats.three_points_attempted += 1
    elif event.event_type == EventType.ASSIST:
        stats.assists += 1
    elif event.event_type == EventType.OFFENSIVE_REBOUND:
        stats.offensive_rebounds += 1
    elif event.event_type == EventType.DEFENSIVE_REBOUND:
        stats.defensive_rebounds += 1
    elif event.event_type == EventType.STEAL:
        stats.steals += 1
    elif event.event_type == EventType.TURNOVER:
        stats.turnovers += 1
    elif event.event_type == EventType.BLOCK:
        stats.blocks += 1
    elif event.event_type == EventType.FOUL:
        stats.fouls += 1


def _jersey_sort_key(number: str) -> tuple[int, int | str]:
    if number.isdigit():
        return (0, int(number))
    return (1, number)


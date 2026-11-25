from __future__ import annotations

from basketball_ai_tracker.models import AnalysisResult, PlayerIdentity, PlayerStats, StatEvent


def format_timecode(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def player_to_dict(player: PlayerIdentity) -> dict:
    return {
        "player_id": player.player_id,
        "team": player.team,
        "jersey_number": player.jersey_number,
        "display_name": player.display_name,
        "label": player.label,
    }


def event_to_dict(event: StatEvent) -> dict:
    return {
        "event_id": event.event_id,
        "event_type": event.event_type.value,
        "player_id": event.player_id,
        "period": event.period,
        "video_time_seconds": event.video_time_seconds,
        "timecode": format_timecode(event.video_time_seconds),
        "game_clock": event.game_clock,
        "points": event.points,
        "related_player_ids": event.related_player_ids,
        "confidence": event.confidence,
        "source": event.source,
        "notes": event.notes,
    }


def stats_to_dict(stats: PlayerStats) -> dict:
    return {
        "player": player_to_dict(stats.player),
        "points": stats.points,
        "assists": stats.assists,
        "rebounds": stats.rebounds,
        "offensive_rebounds": stats.offensive_rebounds,
        "defensive_rebounds": stats.defensive_rebounds,
        "steals": stats.steals,
        "turnovers": stats.turnovers,
        "blocks": stats.blocks,
        "fouls": stats.fouls,
        "field_goals_made": stats.field_goals_made,
        "field_goals_attempted": stats.field_goals_attempted,
        "three_points_made": stats.three_points_made,
        "three_points_attempted": stats.three_points_attempted,
        "free_throws_made": stats.free_throws_made,
        "free_throws_attempted": stats.free_throws_attempted,
    }


def result_to_dict(result: AnalysisResult) -> dict:
    payload = {
        "game_id": result.game_id,
        "video_filename": result.video_filename,
        "duration_seconds": result.duration_seconds,
        "duration_timecode": format_timecode(result.duration_seconds),
        "players": [player_to_dict(player) for player in result.players],
        "events": [event_to_dict(event) for event in result.events],
        "stats": [stats_to_dict(row) for row in result.stats],
        "warnings": result.warnings,
    }
    if result.vision is not None:
        payload["vision"] = result.vision
    return payload

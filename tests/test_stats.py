import unittest

from basketball_ai_tracker.models import EventType, PlayerIdentity, StatEvent
from basketball_ai_tracker.services.serializers import format_timecode
from basketball_ai_tracker.services.stats import build_player_stats


class BuildPlayerStatsTest(unittest.TestCase):
    def test_counts_core_basketball_events(self) -> None:
        players = [
            PlayerIdentity(player_id="home-01", team="Home", jersey_number="1"),
            PlayerIdentity(player_id="home-07", team="Home", jersey_number="7"),
        ]
        events = [
            StatEvent("e1", EventType.TWO_POINTER_MADE, "home-01", 1, 109, points=2),
            StatEvent("e2", EventType.ASSIST, "home-07", 1, 109, related_player_ids=["home-01"]),
            StatEvent("e3", EventType.THREE_POINTER_MISSED, "home-01", 1, 140),
            StatEvent("e4", EventType.OFFENSIVE_REBOUND, "home-07", 1, 142),
            StatEvent("e5", EventType.STEAL, "home-01", 1, 155),
            StatEvent("e6", EventType.TURNOVER, "home-07", 1, 155),
            StatEvent("e7", EventType.BLOCK, "home-01", 1, 180),
        ]

        stats = {row.player.player_id: row for row in build_player_stats(players, events)}

        self.assertEqual(stats["home-01"].points, 2)
        self.assertEqual(stats["home-01"].field_goals_made, 1)
        self.assertEqual(stats["home-01"].field_goals_attempted, 2)
        self.assertEqual(stats["home-01"].three_points_attempted, 1)
        self.assertEqual(stats["home-01"].steals, 1)
        self.assertEqual(stats["home-01"].blocks, 1)
        self.assertEqual(stats["home-07"].assists, 1)
        self.assertEqual(stats["home-07"].rebounds, 1)
        self.assertEqual(stats["home-07"].turnovers, 1)

    def test_keeps_unknown_event_players_visible(self) -> None:
        stats = build_player_stats(
            players=[],
            events=[StatEvent("e1", EventType.FREE_THROW_MADE, "mystery-42", 1, 12, points=1)],
        )

        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].player.display_name, "Unknown player mystery-42")
        self.assertEqual(stats[0].points, 1)
        self.assertEqual(stats[0].free_throws_made, 1)

    def test_format_timecode_handles_minutes_and_hours(self) -> None:
        self.assertEqual(format_timecode(109), "1:49")
        self.assertEqual(format_timecode(3661), "1:01:01")


if __name__ == "__main__":
    unittest.main()


from __future__ import annotations

from pathlib import Path
from typing import Protocol

from basketball_ai_tracker.models import AnalysisResult


class VideoAnalyzer(Protocol):
    def analyze(self, game_id: str, video_path: Path, original_filename: str) -> AnalysisResult:
        """Analyze a video file and return detected players, events, and box-score stats."""


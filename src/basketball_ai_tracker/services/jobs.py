from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock


@dataclass
class JobRecord:
    game_id: str
    original_filename: str
    video_path: Path
    result_path: Path
    status: str = "queued"
    progress: float = 0.0
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class InMemoryJobStore:
    def __init__(self) -> None:
        self._records: dict[str, JobRecord] = {}
        self._lock = Lock()

    def create(self, record: JobRecord) -> JobRecord:
        with self._lock:
            self._records[record.game_id] = record
            return record

    def get(self, game_id: str) -> JobRecord | None:
        with self._lock:
            return self._records.get(game_id)

    def update(self, game_id: str, **changes: object) -> JobRecord:
        with self._lock:
            record = self._records[game_id]
            for key, value in changes.items():
                setattr(record, key, value)
            record.updated_at = datetime.now(UTC)
            return record


def job_to_dict(record: JobRecord) -> dict:
    return {
        "game_id": record.game_id,
        "original_filename": record.original_filename,
        "status": record.status,
        "progress": record.progress,
        "error": record.error,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }

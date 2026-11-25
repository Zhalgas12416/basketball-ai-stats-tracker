from __future__ import annotations

import json
import mimetypes
import re
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from basketball_ai_tracker.analyzers.factory import create_default_analyzer
from basketball_ai_tracker.services.jobs import InMemoryJobStore, JobRecord, job_to_dict
from basketball_ai_tracker.services.serializers import format_timecode, result_to_dict
from basketball_ai_tracker.vision.correction_exporter import export_corrections_to_yolo_dataset, find_video_for_game
from basketball_ai_tracker.vision.hard_frame_sampler import (
    DEFAULT_TARGET_CLASSES,
    select_hard_frame_candidates,
)
from basketball_ai_tracker.vision.quality_report import (
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    build_detection_quality_report,
    compare_detection_quality_reports,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = PROJECT_ROOT / "web"
UPLOAD_DIR = PROJECT_ROOT / "storage" / "uploads"
RESULT_DIR = PROJECT_ROOT / "storage" / "results"
CORRECTION_DIR = PROJECT_ROOT / "storage" / "corrections"
CORRECTION_DATASET_DIR = PROJECT_ROOT / "datasets" / "basketball-corrections"
CORRECTION_LABELS = {"player", "ball", "rim", "referee", "backboard"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CORRECTION_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Basketball AI Tracker", version="0.1.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

job_store = InMemoryJobStore()
analyzer = create_default_analyzer()


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "analyzer": _describe_analyzer()}


@app.post("/api/games")
async def upload_game(background_tasks: BackgroundTasks, video: UploadFile = File(...)) -> dict:
    if not video.filename:
        raise HTTPException(status_code=400, detail="A video filename is required.")

    game_id = uuid4().hex
    safe_name = _safe_filename(video.filename)
    suffix = Path(safe_name).suffix or ".mp4"
    video_path = UPLOAD_DIR / f"{game_id}{suffix}"
    result_path = RESULT_DIR / f"{game_id}.json"

    with video_path.open("wb") as target:
        shutil.copyfileobj(video.file, target)

    record = job_store.create(
        JobRecord(
            game_id=game_id,
            original_filename=safe_name,
            video_path=video_path,
            result_path=result_path,
        )
    )
    background_tasks.add_task(_run_analysis, game_id)

    return {"job": job_to_dict(record)}


@app.get("/api/games/{game_id}")
def get_game(game_id: str) -> dict:
    record = _get_record(game_id)
    payload = {"job": job_to_dict(record)}
    if record.status == "completed":
        payload["result"] = _read_result(record)
    return payload


@app.get("/api/games/{game_id}/stats")
def get_stats(game_id: str) -> dict:
    record = _get_record(game_id)
    result = _require_result(record)
    return {"game_id": game_id, "stats": result["stats"]}


@app.get("/api/games/{game_id}/events")
def get_events(game_id: str) -> dict:
    record = _get_record(game_id)
    result = _require_result(record)
    return {"game_id": game_id, "events": result["events"]}


@app.get("/api/games/{game_id}/quality")
def get_detection_quality(game_id: str, threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD) -> dict:
    record = _get_record(game_id)
    result = _require_result(record)
    return {
        "game_id": game_id,
        "quality": build_detection_quality_report(result.get("vision") or {}, threshold),
    }


@app.post("/api/games/{game_id}/hard-frames")
def get_hard_frames(game_id: str, payload: dict = Body(default={})) -> dict:
    record = _get_record(game_id)
    result = _require_result(record)
    vision = result.get("vision") or {}
    if not vision:
        raise HTTPException(status_code=409, detail="Hard-frame sampling requires a completed vision result.")

    candidates = select_hard_frame_candidates(
        vision,
        target_classes=_classes_from_payload(payload),
        max_frames=_int_payload(payload, "max_frames", 40),
        sample_interval_seconds=_float_payload(payload, "sample_interval_seconds", 0.5),
        min_spacing_seconds=_float_payload(payload, "min_spacing_seconds", 1.0),
        low_confidence_threshold=_float_payload(payload, "low_confidence_threshold", 0.35),
        detection_window_seconds=_float_payload(payload, "detection_window_seconds", 0.25),
        min_player_count=_int_payload(payload, "min_player_count", 4),
    )
    return {
        "game_id": game_id,
        "schema": "basketball-hard-frame-queue-v1",
        "frames": [candidate.to_dict() for candidate in candidates],
    }


@app.post("/api/quality/compare")
def compare_detection_quality(payload: dict = Body(...)) -> dict:
    before = payload.get("before") or {}
    after = payload.get("after") or {}
    threshold = float(payload.get("threshold", DEFAULT_LOW_CONFIDENCE_THRESHOLD))
    before_report = before if "class_counts" in before else build_detection_quality_report(before.get("vision") or before, threshold)
    after_report = after if "class_counts" in after else build_detection_quality_report(after.get("vision") or after, threshold)
    return {
        "before": before_report,
        "after": after_report,
        "comparison": compare_detection_quality_reports(before_report, after_report),
    }


@app.get("/api/games/{game_id}/video")
def get_video(game_id: str) -> FileResponse:
    record = _get_record(game_id)
    media_type = mimetypes.guess_type(record.video_path.name)[0] or "video/mp4"
    return FileResponse(
        record.video_path,
        media_type=media_type,
        filename=record.original_filename,
    )


@app.get("/api/games/{game_id}/corrections")
def get_corrections(game_id: str) -> dict:
    _get_record(game_id)
    return {"game_id": game_id, "corrections": _read_corrections(game_id)}


@app.post("/api/games/{game_id}/corrections")
def create_correction(game_id: str, payload: dict = Body(...)) -> dict:
    _get_record(game_id)
    correction = _validate_correction(payload)
    corrections = _read_corrections(game_id)
    corrections.append(correction)
    _write_corrections(game_id, corrections)
    return {"game_id": game_id, "correction": correction}


@app.delete("/api/games/{game_id}/corrections/{correction_id}")
def delete_correction(game_id: str, correction_id: str) -> dict:
    _get_record(game_id)
    corrections = _read_corrections(game_id)
    remaining = [correction for correction in corrections if correction.get("correction_id") != correction_id]
    if len(remaining) == len(corrections):
        raise HTTPException(status_code=404, detail="Correction not found.")
    _write_corrections(game_id, remaining)
    return {"game_id": game_id, "deleted": correction_id, "corrections": remaining}


@app.get("/api/games/{game_id}/corrections/export")
def export_corrections(game_id: str) -> dict:
    _get_record(game_id)
    return {
        "game_id": game_id,
        "schema": "basketball-detection-corrections-v1",
        "labels": sorted(CORRECTION_LABELS),
        "corrections": _read_corrections(game_id),
    }


@app.post("/api/games/{game_id}/corrections/export-yolo")
def export_corrections_yolo(game_id: str, payload: dict = Body(default={})) -> dict:
    record = _get_record(game_id)
    corrections_path = _correction_path(game_id)
    if not corrections_path.exists() or not _read_corrections(game_id):
        raise HTTPException(status_code=409, detail="No corrections have been saved for this game yet.")

    output_dataset = Path(str(payload.get("output_dataset") or CORRECTION_DATASET_DIR)).expanduser()
    if not output_dataset.is_absolute():
        output_dataset = PROJECT_ROOT / output_dataset
    try:
        summary = export_corrections_to_yolo_dataset(
            game_id=game_id,
            video_path=record.video_path,
            corrections_path=corrections_path,
            output_dataset=output_dataset,
            split=str(payload.get("split") or "train"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "game_id": game_id,
        "schema": "basketball-correction-yolo-export-v1",
        "summary": summary.to_dict(),
        "next_step": (
            "Label hard frames, export corrections, then merge datasets/basketball-corrections "
            "into your training dataset before retraining."
        ),
    }


def _run_analysis(game_id: str) -> None:
    record = job_store.update(game_id, status="processing", progress=0.2, error=None)
    try:
        result = analyzer.analyze(
            game_id=record.game_id,
            video_path=record.video_path,
            original_filename=record.original_filename,
        )
        with record.result_path.open("w", encoding="utf-8") as result_file:
            json.dump(result_to_dict(result), result_file, indent=2)
        job_store.update(game_id, status="completed", progress=1.0)
    except Exception as exc:
        job_store.update(game_id, status="failed", progress=1.0, error=str(exc))


def _get_record(game_id: str) -> JobRecord:
    record = job_store.get(game_id)
    if record is None:
        record = _recover_record_from_storage(game_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Game not found.")
    return record


def _recover_record_from_storage(game_id: str) -> JobRecord | None:
    result_path = RESULT_DIR / f"{game_id}.json"
    if not result_path.exists():
        return None
    try:
        video_path = find_video_for_game(game_id, UPLOAD_DIR)
    except FileNotFoundError:
        return None
    record = JobRecord(
        game_id=game_id,
        original_filename=video_path.name,
        video_path=video_path,
        result_path=result_path,
        status="completed",
        progress=1.0,
    )
    return job_store.create(record)


def _require_result(record: JobRecord) -> dict:
    if record.status != "completed":
        raise HTTPException(status_code=409, detail=f"Analysis is {record.status}.")
    return _read_result(record)


def _read_result(record: JobRecord) -> dict:
    with record.result_path.open(encoding="utf-8") as result_file:
        return json.load(result_file)


def _describe_analyzer() -> dict:
    describe = getattr(analyzer, "describe", None)
    if callable(describe):
        return describe()
    return {"name": type(analyzer).__name__, "detector": {"engine": "unknown"}}


def _correction_path(game_id: str) -> Path:
    return CORRECTION_DIR / f"{game_id}.json"


def _read_corrections(game_id: str) -> list[dict]:
    path = _correction_path(game_id)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as correction_file:
        payload = json.load(correction_file)
    return list(payload.get("corrections", []))


def _write_corrections(game_id: str, corrections: list[dict]) -> None:
    payload = {
        "game_id": game_id,
        "schema": "basketball-detection-corrections-v1",
        "corrections": corrections,
    }
    with _correction_path(game_id).open("w", encoding="utf-8") as correction_file:
        json.dump(payload, correction_file, indent=2)


def _validate_correction(payload: dict) -> dict:
    label = str(payload.get("label", "")).strip().lower()
    if label not in CORRECTION_LABELS:
        raise HTTPException(status_code=400, detail=f"Correction label must be one of: {', '.join(sorted(CORRECTION_LABELS))}.")

    bbox_payload = payload.get("bbox") or {}
    try:
        bbox = {
            "x": max(0.0, float(bbox_payload["x"])),
            "y": max(0.0, float(bbox_payload["y"])),
            "width": float(bbox_payload["width"]),
            "height": float(bbox_payload["height"]),
        }
        video_time_seconds = max(0.0, float(payload.get("video_time_seconds", 0)))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Correction requires numeric bbox x, y, width, height, and video_time_seconds.") from exc

    if bbox["width"] <= 1 or bbox["height"] <= 1:
        raise HTTPException(status_code=400, detail="Correction bbox must be larger than 1 pixel.")

    notes = payload.get("notes")
    return {
        "correction_id": uuid4().hex,
        "label": label,
        "bbox": bbox,
        "video_time_seconds": video_time_seconds,
        "timecode": format_timecode(video_time_seconds),
        "source": "manual_overlay",
        "notes": str(notes).strip() if notes else None,
    }


def _safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return cleaned or "uploaded-game.mp4"


def _classes_from_payload(payload: dict) -> list[str]:
    raw_classes = payload.get("classes", DEFAULT_TARGET_CLASSES)
    if isinstance(raw_classes, str):
        values = raw_classes.split(",")
    else:
        values = list(raw_classes or [])
    classes = [str(value).strip().lower() for value in values if str(value).strip()]
    return classes or list(DEFAULT_TARGET_CLASSES)


def _int_payload(payload: dict, key: str, default: int) -> int:
    try:
        return int(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def _float_payload(payload: dict, key: str, default: float) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default

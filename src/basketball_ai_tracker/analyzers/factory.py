from __future__ import annotations

import os
from pathlib import Path

from basketball_ai_tracker.analyzers.base import VideoAnalyzer
from basketball_ai_tracker.analyzers.cv import CVBasketballAnalyzer
from basketball_ai_tracker.analyzers.demo import DemoBasketballAnalyzer
from basketball_ai_tracker.vision.opencv_detector import OpenCVBasketballDetector, OpenCVDetectorConfig
from basketball_ai_tracker.vision.hybrid_detector import HybridBasketballDetector, HybridDetectorConfig
from basketball_ai_tracker.vision.yolo_detector import YOLOBasketballDetector, YOLODetectorConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def create_default_analyzer() -> VideoAnalyzer:
    _ensure_writable_runtime_dirs()
    weights_path = _resolve_weights_path()
    if weights_path is not None:
        try:
            import ultralytics

            yolo_detector = YOLOBasketballDetector(_build_yolo_config(weights_path))
            return CVBasketballAnalyzer(
                detector=HybridBasketballDetector(
                    primary=yolo_detector,
                    fallback=OpenCVBasketballDetector(_build_opencv_config()),
                    config=HybridDetectorConfig(
                        max_supplemental_player_tracks=_int_env(
                            "BASKETBALL_HYBRID_MAX_PLAYER_TRACKS",
                            HybridDetectorConfig.max_supplemental_player_tracks,
                        ),
                        min_primary_player_tracks=_int_env(
                            "BASKETBALL_HYBRID_MIN_PRIMARY_PLAYER_TRACKS",
                            HybridDetectorConfig.min_primary_player_tracks,
                        )
                    ),
                )
            )
        except Exception as exc:
            startup_warnings = [
                f"Trained YOLO weights were found at {weights_path}, but YOLO could not start: {exc}",
                "Falling back to the OpenCV detector until the training dependencies are available.",
            ]
            return CVBasketballAnalyzer(startup_warnings=startup_warnings)

    try:
        import cv2
    except ImportError:
        return DemoBasketballAnalyzer(
            warnings=[
                "OpenCV is not installed, so the demo analyzer is active. Install with: pip install -e '.[cv]'"
            ]
        )
    return CVBasketballAnalyzer()


def _resolve_weights_path() -> Path | None:
    configured = os.environ.get("BASKETBALL_YOLO_WEIGHTS")
    if configured:
        path = Path(configured).expanduser()
        return path if path.exists() else None

    default_path = PROJECT_ROOT / "models" / "basketball-detector.pt"
    return default_path if default_path.exists() else None


def _build_yolo_config(weights_path: Path) -> YOLODetectorConfig:
    return YOLODetectorConfig(
        model_path=weights_path,
        sample_fps=_float_env("BASKETBALL_YOLO_SAMPLE_FPS", YOLODetectorConfig.sample_fps),
        max_frames=_int_env("BASKETBALL_YOLO_MAX_FRAMES", YOLODetectorConfig.max_frames),
        confidence=_float_env("BASKETBALL_YOLO_CONF", YOLODetectorConfig.confidence),
        ball_confidence=_float_env("BASKETBALL_YOLO_BALL_CONF", YOLODetectorConfig.ball_confidence),
        iou=_float_env("BASKETBALL_YOLO_IOU", YOLODetectorConfig.iou),
        imgsz=_int_env("BASKETBALL_YOLO_IMGSZ", YOLODetectorConfig.imgsz),
        min_track_observations=_int_env(
            "BASKETBALL_YOLO_MIN_TRACK_OBSERVATIONS",
            YOLODetectorConfig.min_track_observations,
        ),
        max_tracks=_int_env("BASKETBALL_YOLO_MAX_TRACKS", YOLODetectorConfig.max_tracks),
        auto_crop=_bool_env("BASKETBALL_YOLO_AUTO_CROP", YOLODetectorConfig.auto_crop),
    )


def _build_opencv_config() -> OpenCVDetectorConfig:
    return OpenCVDetectorConfig(
        sample_fps=_float_env("BASKETBALL_OPENCV_SAMPLE_FPS", OpenCVDetectorConfig.sample_fps),
        max_frames=_int_env("BASKETBALL_OPENCV_MAX_FRAMES", OpenCVDetectorConfig.max_frames),
        warmup_sampled_frames=_int_env(
            "BASKETBALL_OPENCV_WARMUP_FRAMES",
            OpenCVDetectorConfig.warmup_sampled_frames,
        ),
        min_player_height_ratio=_float_env(
            "BASKETBALL_OPENCV_MIN_PLAYER_HEIGHT_RATIO",
            OpenCVDetectorConfig.min_player_height_ratio,
        ),
        max_player_height_ratio=_float_env(
            "BASKETBALL_OPENCV_MAX_PLAYER_HEIGHT_RATIO",
            OpenCVDetectorConfig.max_player_height_ratio,
        ),
        min_player_area_ratio=_float_env(
            "BASKETBALL_OPENCV_MIN_PLAYER_AREA_RATIO",
            OpenCVDetectorConfig.min_player_area_ratio,
        ),
        max_player_area_ratio=_float_env(
            "BASKETBALL_OPENCV_MAX_PLAYER_AREA_RATIO",
            OpenCVDetectorConfig.max_player_area_ratio,
        ),
        max_player_candidates_per_frame=_int_env(
            "BASKETBALL_OPENCV_MAX_PLAYER_CANDIDATES",
            OpenCVDetectorConfig.max_player_candidates_per_frame,
        ),
        min_player_track_observations=_int_env(
            "BASKETBALL_OPENCV_MIN_PLAYER_TRACK_OBSERVATIONS",
            OpenCVDetectorConfig.min_player_track_observations,
        ),
        max_tracks=_int_env("BASKETBALL_OPENCV_MAX_TRACKS", OpenCVDetectorConfig.max_tracks),
    )


def _ensure_writable_runtime_dirs() -> None:
    yolo_config_dir = PROJECT_ROOT / ".ultralytics"
    matplotlib_config_dir = PROJECT_ROOT / ".matplotlib"
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))


def _float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}

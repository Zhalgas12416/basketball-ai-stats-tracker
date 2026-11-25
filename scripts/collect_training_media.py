from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect permitted basketball media and extract YOLO training frames."
    )
    parser.add_argument("--sources", type=Path, required=True, help="Text file with one local path or direct URL per line.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets" / "basketball")
    parser.add_argument("--frame-step-seconds", type=float, default=1.0)
    parser.add_argument("--max-frames-per-video", type=int, default=400)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--no-auto-crop", action="store_true", help="Do not crop embedded game-video regions.")
    args = parser.parse_args()

    dataset = args.dataset
    raw_images = dataset / "raw" / "images"
    raw_videos = dataset / "raw" / "videos"
    train_images = dataset / "images" / "train"
    val_images = dataset / "images" / "val"
    for directory in [raw_images, raw_videos, train_images, val_images]:
        directory.mkdir(parents=True, exist_ok=True)

    sources = _read_sources(args.sources)
    imported_images = 0
    extracted_frames = 0

    for index, source in enumerate(sources, start=1):
        media_path = _materialize_source(source, raw_images, raw_videos)
        suffix = media_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            target_dir = _split_dir(index, args.val_ratio, train_images, val_images)
            target = target_dir / f"{media_path.stem}{media_path.suffix.lower()}"
            shutil.copy2(media_path, target)
            imported_images += 1
        elif suffix in VIDEO_EXTENSIONS:
            extracted_frames += _extract_video_frames(
                media_path,
                dataset,
                index,
                args.frame_step_seconds,
                args.max_frames_per_video,
                args.val_ratio,
                not args.no_auto_crop,
            )
        else:
            print(f"Skipped unsupported source: {source}")

    print(f"Imported images: {imported_images}")
    print(f"Extracted frames: {extracted_frames}")
    print(f"Dataset images are in: {dataset / 'images'}")


def _read_sources(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _materialize_source(source: str, raw_images: Path, raw_videos: Path) -> Path:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        return _download_direct_url(source, raw_images, raw_videos)

    local_path = Path(source).expanduser()
    if not local_path.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")
    return local_path


def _download_direct_url(url: str, raw_images: Path, raw_videos: Path) -> Path:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: pip install -e '.[train]'") from exc

    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        target_dir = raw_images
    elif suffix in VIDEO_EXTENSIONS:
        target_dir = raw_videos
    else:
        raise ValueError(f"URL must end with a supported image/video extension: {url}")

    target = target_dir / _safe_filename(Path(parsed.path).name or f"download{suffix}")
    if target.exists():
        return target

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with target.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    return target


def _extract_video_frames(
    video_path: Path,
    dataset: Path,
    source_index: int,
    frame_step_seconds: float,
    max_frames: int,
    val_ratio: float,
    auto_crop: bool,
) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install CV dependencies with: pip install -e '.[cv]'") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Could not open video: {video_path}")
        return 0

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    step = max(1, int(round(fps * frame_step_seconds)))
    analysis_region = None
    if auto_crop:
        from basketball_ai_tracker.vision.opencv_detector import OpenCVBasketballDetector, _crop_frame

        analysis_region = OpenCVBasketballDetector()._detect_analysis_region(cv2, capture)
    else:
        _crop_frame = None

    frame_index = 0
    saved = 0

    try:
        while saved < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step != 0:
                frame_index += 1
                continue

            split = "val" if _is_val_item(source_index + saved, val_ratio) else "train"
            target_dir = dataset / "images" / split
            target = target_dir / f"{video_path.stem}_f{frame_index:06d}.jpg"
            if analysis_region is not None and _crop_frame is not None:
                frame = _crop_frame(frame, analysis_region)
            cv2.imwrite(str(target), frame)
            saved += 1
            frame_index += 1
    finally:
        capture.release()

    return saved


def _split_dir(index: int, val_ratio: float, train_dir: Path, val_dir: Path) -> Path:
    return val_dir if _is_val_item(index, val_ratio) else train_dir


def _is_val_item(index: int, val_ratio: float) -> bool:
    if val_ratio <= 0:
        return False
    bucket = max(2, round(1 / val_ratio))
    return index % bucket == 0


def _safe_filename(name: str) -> str:
    return "".join(character if character.isalnum() or character in ".-_" else "_" for character in name)


if __name__ == "__main__":
    main()

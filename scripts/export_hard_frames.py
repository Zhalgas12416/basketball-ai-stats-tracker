from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from basketball_ai_tracker.vision.correction_exporter import find_video_for_game
from basketball_ai_tracker.vision.hard_frame_sampler import (
    DEFAULT_TARGET_CLASSES,
    export_hard_frames_to_yolo_labeling_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export hard/weak detection frames for manual YOLO labeling.",
    )
    parser.add_argument("--game-id", required=True, help="Game id to sample from.")
    parser.add_argument("--video", type=Path, help="Video path. Defaults to storage/uploads/{game_id}.*")
    parser.add_argument("--result", type=Path, help="Result JSON path. Defaults to storage/results/{game_id}.json")
    parser.add_argument("--uploads-dir", type=Path, default=PROJECT_ROOT / "storage" / "uploads")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets" / "basketball-hard-frames")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--classes", default=",".join(DEFAULT_TARGET_CLASSES), help="Comma-separated classes to target.")
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--sample-interval", type=float, default=0.5, help="Seconds between candidate timestamps.")
    parser.add_argument("--min-spacing", type=float, default=1.0, help="Minimum seconds between exported frames.")
    parser.add_argument("--low-confidence", type=float, default=0.35)
    parser.add_argument("--window", type=float, default=0.25, help="Nearby detection window in seconds.")
    parser.add_argument("--min-player-count", type=int, default=4)
    parser.add_argument("--image-quality", type=int, default=95)
    args = parser.parse_args()

    video_path = args.video or find_video_for_game(args.game_id, args.uploads_dir)
    result_path = args.result or PROJECT_ROOT / "storage" / "results" / f"{args.game_id}.json"
    if not result_path.exists():
        raise FileNotFoundError(f"No result JSON found: {result_path}")

    target_classes = [label.strip() for label in args.classes.split(",") if label.strip()]
    summary = export_hard_frames_to_yolo_labeling_dataset(
        game_id=args.game_id,
        video_path=video_path,
        result_path=result_path,
        output_dataset=args.output,
        split=args.split,
        target_classes=target_classes,
        max_frames=args.max_frames,
        sample_interval_seconds=args.sample_interval,
        min_spacing_seconds=args.min_spacing,
        low_confidence_threshold=args.low_confidence,
        detection_window_seconds=args.window,
        min_player_count=args.min_player_count,
        image_quality=args.image_quality,
    )
    print(
        f"{args.game_id}: {summary.exported_images} hard frames, "
        f"{summary.exported_label_files} empty label files, "
        f"{len(summary.skipped_frames)} skipped"
    )
    print(json.dumps(summary.to_dict(), indent=2))
    print("These empty .txt files are placeholders. Label them before using this export for training.")


if __name__ == "__main__":
    main()

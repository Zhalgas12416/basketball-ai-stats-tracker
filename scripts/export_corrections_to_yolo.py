from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from basketball_ai_tracker.vision.correction_exporter import (
    export_corrections_to_yolo_dataset,
    find_video_for_game,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export saved review/correction boxes into a YOLO training dataset.",
    )
    parser.add_argument("--game-id", help="Game id to export. Omit with --all to export every correction file.")
    parser.add_argument("--all", action="store_true", help="Export every storage/corrections/*.json file.")
    parser.add_argument("--video", type=Path, help="Video path. Only valid with --game-id.")
    parser.add_argument("--corrections", type=Path, help="Correction JSON path. Only valid with --game-id.")
    parser.add_argument("--uploads-dir", type=Path, default=PROJECT_ROOT / "storage" / "uploads")
    parser.add_argument("--corrections-dir", type=Path, default=PROJECT_ROOT / "storage" / "corrections")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets" / "basketball-corrections")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--image-quality", type=int, default=95)
    args = parser.parse_args()

    if args.all and (args.game_id or args.video or args.corrections):
        parser.error("--all cannot be combined with --game-id, --video, or --corrections.")
    if not args.all and not args.game_id:
        parser.error("Provide --game-id or --all.")
    if args.video and not args.game_id:
        parser.error("--video requires --game-id.")
    if args.corrections and not args.game_id:
        parser.error("--corrections requires --game-id.")

    summaries = []
    for game_id, corrections_path, video_path in _iter_exports(args):
        summary = export_corrections_to_yolo_dataset(
            game_id=game_id,
            video_path=video_path,
            corrections_path=corrections_path,
            output_dataset=args.output,
            split=args.split,
            image_quality=args.image_quality,
        )
        summaries.append(summary.to_dict())
        print(
            f"{game_id}: {summary.exported_images} images, "
            f"{summary.exported_boxes} boxes, "
            f"{len(summary.skipped_corrections)} skipped"
        )

    print(json.dumps({"summaries": summaries}, indent=2))


def _iter_exports(args: argparse.Namespace):
    if args.all:
        for corrections_path in sorted(args.corrections_dir.glob("*.json")):
            game_id = corrections_path.stem
            try:
                video_path = find_video_for_game(game_id, args.uploads_dir)
            except FileNotFoundError as exc:
                print(f"Skipping {game_id}: {exc}", file=sys.stderr)
                continue
            yield game_id, corrections_path, video_path
        return

    game_id = args.game_id
    corrections_path = args.corrections or args.corrections_dir / f"{game_id}.json"
    video_path = args.video or find_video_for_game(game_id, args.uploads_dir)
    if not corrections_path.exists():
        raise FileNotFoundError(f"No correction JSON found: {corrections_path}")
    yield game_id, corrections_path, video_path


if __name__ == "__main__":
    main()

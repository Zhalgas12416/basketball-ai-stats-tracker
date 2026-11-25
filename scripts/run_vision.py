from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from basketball_ai_tracker.vision.opencv_detector import OpenCVBasketballDetector, OpenCVDetectorConfig
from basketball_ai_tracker.vision.serializers import tracking_result_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenCV player/ball detection and tracking.")
    parser.add_argument("video", type=Path, help="Path to a local basketball video.")
    parser.add_argument("--sample-fps", type=float, default=6.0, help="Frames per second to sample.")
    parser.add_argument("--max-frames", type=int, default=1800, help="Maximum sampled frames to analyze.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    detector = OpenCVBasketballDetector(
        OpenCVDetectorConfig(sample_fps=args.sample_fps, max_frames=args.max_frames)
    )
    result = detector.analyze_video(args.video)
    payload = tracking_result_to_dict(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(json.dumps(_summarize(payload), indent=2))


def _summarize(payload: dict) -> dict:
    return {
        "duration_seconds": payload["duration_seconds"],
        "sampled_frame_count": payload["sampled_frame_count"],
        "detection_count": payload["detection_count"],
        "track_count": payload["track_count"],
        "tracks": [
            {
                "track_id": track["track_id"],
                "label": track["label"],
                "first_seen_seconds": track["first_seen_seconds"],
                "last_seen_seconds": track["last_seen_seconds"],
                "observation_count": track["observation_count"],
                "max_confidence": track["max_confidence"],
            }
            for track in payload["tracks"][:30]
        ],
        "warnings": payload["warnings"],
    }


if __name__ == "__main__":
    main()


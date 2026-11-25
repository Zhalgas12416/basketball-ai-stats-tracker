from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from basketball_ai_tracker.vision.quality_report import (
    build_detection_quality_report_from_result_path,
    compare_detection_quality_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare detection quality between two analysis result JSON files.")
    parser.add_argument("--before", type=Path, required=True, help="Result JSON from the earlier model/run.")
    parser.add_argument("--after", type=Path, required=True, help="Result JSON from the later model/run.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Low-confidence threshold.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a readable table.")
    args = parser.parse_args()

    before = build_detection_quality_report_from_result_path(args.before, args.threshold)
    after = build_detection_quality_report_from_result_path(args.after, args.threshold)
    comparison = compare_detection_quality_reports(before, after)

    payload = {"before": before, "after": after, "comparison": comparison}
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Before: {args.before}")
    print(f"After:  {args.after}")
    print("")
    print("Overall")
    print(f"  detections:      {before['detection_count']} -> {after['detection_count']} ({_signed(comparison['detection_count_delta'])})")
    print(f"  detected frames: {before['detected_frame_count']} -> {after['detected_frame_count']} ({_signed(comparison['detected_frame_count_delta'])})")
    print(f"  empty sampled:   {before['empty_sampled_frame_count']} -> {after['empty_sampled_frame_count']} ({_signed(comparison['empty_sampled_frame_count_delta'])})")
    print(f"  low confidence:  {before['low_confidence_count']} -> {after['low_confidence_count']} ({_signed(comparison['low_confidence_count_delta'])})")
    print("")
    print("Per class")
    for label in comparison["labels"]:
        before_count = before["class_counts"].get(label, 0)
        after_count = after["class_counts"].get(label, 0)
        delta = comparison["class_count_deltas"][label]
        low_delta = comparison["low_confidence_deltas"][label]
        avg_delta = comparison["average_confidence_deltas"][label]
        avg_delta_text = "n/a" if avg_delta is None else _signed_float(avg_delta)
        print(
            f"  {label:10} count {before_count:5} -> {after_count:5} ({_signed(delta):>5})"
            f" | low-conf {_signed(low_delta):>5}"
            f" | avg-conf {avg_delta_text}"
        )


def _signed(value: int) -> str:
    return f"{value:+d}"


def _signed_float(value: float) -> str:
    return f"{value:+.4f}"


if __name__ == "__main__":
    main()

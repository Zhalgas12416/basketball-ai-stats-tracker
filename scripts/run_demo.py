from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from basketball_ai_tracker.analyzers.demo import DemoBasketballAnalyzer
from basketball_ai_tracker.services.serializers import result_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the demo analyzer on a basketball video.")
    parser.add_argument("video", type=Path, help="Path to a local video file.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    analyzer = DemoBasketballAnalyzer()
    result = analyzer.analyze(uuid4().hex, args.video, args.video.name)
    payload = result_to_dict(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()


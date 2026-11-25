from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from basketball_ai_tracker.vision.model_importer import (
    build_model_compatibility_report,
    class_names_from_ultralytics_weights,
    install_yolo_weights,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and optionally install external YOLO basketball detector weights.",
    )
    parser.add_argument("weights", type=Path, help="Path to an external .pt weights file.")
    parser.add_argument("--install", action="store_true", help="Copy weights into models/basketball-detector.pt.")
    parser.add_argument("--target", type=Path, default=PROJECT_ROOT / "models" / "basketball-detector.pt")
    parser.add_argument("--overwrite", action="store_true", help="Replace the current target weights if present.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    args = parser.parse_args()

    class_names = class_names_from_ultralytics_weights(args.weights)
    report = build_model_compatibility_report(class_names)
    installed_to = None
    if args.install:
        installed_to = install_yolo_weights(args.weights, args.target, overwrite=args.overwrite)

    payload = {
        "weights": str(args.weights.expanduser()),
        "class_names": class_names,
        "compatibility": report,
        "installed_to": str(installed_to) if installed_to else None,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Weights: {payload['weights']}")
    print(f"Classes: {', '.join(class_names) if class_names else 'none found'}")
    print("\nMapped classes")
    for label, names in report["mapped_classes"].items():
        print(f"  {label}: {', '.join(names) if names else '-'}")
    if report["unmapped_classes"]:
        print(f"\nUnmapped classes: {', '.join(report['unmapped_classes'])}")
    if report["missing_target_classes"]:
        print(f"Missing target classes: {', '.join(report['missing_target_classes'])}")
        print("This model can still be useful, but our app will only use the mapped classes.")
    else:
        print("All target classes are covered.")
    if installed_to:
        print(f"\nInstalled weights to: {installed_to}")
        print("Restart the app server so the analyzer loads the new model.")


if __name__ == "__main__":
    main()

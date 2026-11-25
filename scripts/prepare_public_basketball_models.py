from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from basketball_ai_tracker.vision.model_importer import (
    EBARD_LICENSE,
    EBARD_MODEL_CARD_URL,
    EBARD_YOLO_MODEL_URL,
    build_model_compatibility_report,
    class_names_from_ultralytics_weights,
    download_file,
    install_yolo_weights,
    write_external_model_attribution,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download public basketball detector models and prepare transfer-learning commands.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "external")
    parser.add_argument("--overwrite-download", action="store_true")
    parser.add_argument("--install-ebard", action="store_true", help="Install E-BARD directly as the active app detector.")
    parser.add_argument("--install-target", type=Path, default=PROJECT_ROOT / "models" / "basketball-detector.pt")
    parser.add_argument("--overwrite-install", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser()
    ebard_weights = download_file(
        EBARD_YOLO_MODEL_URL,
        output_dir / "ebard-yolov8n-bodd.pt",
        overwrite=args.overwrite_download,
    )
    class_names = class_names_from_ultralytics_weights(ebard_weights)
    report = build_model_compatibility_report(class_names)
    attribution = write_external_model_attribution(
        output_dir,
        [
            {
                "name": "E-BARD YOLOv8n Basketball Object Detection Model",
                "source": EBARD_MODEL_CARD_URL,
                "license": EBARD_LICENSE,
                "local_file": str(ebard_weights),
            }
        ],
    )

    installed_to = None
    if args.install_ebard:
        installed_to = install_yolo_weights(
            ebard_weights,
            args.install_target,
            overwrite=args.overwrite_install,
        )

    fine_tune_command = [
        ".venv/bin/python",
        "scripts/train_detector.py",
        "--data",
        "configs/basketball-detector.yaml",
        "--model",
        str(ebard_weights),
        "--epochs",
        "100",
        "--imgsz",
        "1280",
        "--batch",
        "8",
        "--name",
        "basketball-detector-ebard-transfer",
        "--export-to",
        "models/basketball-detector.pt",
    ]
    payload = {
        "models": [
            {
                "name": "E-BARD YOLOv8n",
                "path": str(ebard_weights),
                "source": EBARD_MODEL_CARD_URL,
                "license": EBARD_LICENSE,
                "class_names": class_names,
                "compatibility": report,
            }
        ],
        "attribution": str(attribution),
        "installed_to": str(installed_to) if installed_to else None,
        "recommended_fine_tune_command": fine_tune_command,
        "notes": [
            "E-BARD covers ball, rim/hoop, player, and referee.",
            "E-BARD does not include backboard, so fine-tune on this project's 5-class dataset before expecting backboard detections.",
            "The GitHub hoop-tracker repository is useful training tooling reference, but it does not publish detector weights or a dataset in the repository.",
        ],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Downloaded E-BARD YOLO weights: {ebard_weights}")
    print(f"Attribution written to: {attribution}")
    print(f"Classes: {', '.join(class_names)}")
    print("\nMapped classes")
    for label, names in report["mapped_classes"].items():
        print(f"  {label}: {', '.join(names) if names else '-'}")
    if report["missing_target_classes"]:
        print(f"\nMissing target classes: {', '.join(report['missing_target_classes'])}")
    if installed_to:
        print(f"\nInstalled active app detector: {installed_to}")
    print("\nRecommended 5-class fine-tuning command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in fine_tune_command))


if __name__ == "__main__":
    main()

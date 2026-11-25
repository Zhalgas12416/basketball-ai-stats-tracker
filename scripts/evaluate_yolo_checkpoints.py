from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more YOLO checkpoints on the same dataset split."
    )
    parser.add_argument("--data", type=Path, required=True, help="YOLO data.yaml to evaluate against.")
    parser.add_argument("--model", type=Path, action="append", required=True, help="Checkpoint path. Repeatable.")
    parser.add_argument("--imgsz", type=int, default=704)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "models" / "runs")
    parser.add_argument("--name-prefix", default="eval")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: pip install -e '.[train]'") from exc

    rows: list[dict[str, float | int | str]] = []
    for model_path in args.model:
        model_name = model_path.stem
        print(f"\n=== {model_name} ===")
        model = YOLO(str(model_path))
        metrics = model.val(
            data=str(args.data),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            plots=False,
            verbose=False,
            project=str(args.project),
            name=f"{args.name_prefix}-{model_name}-img{args.imgsz}",
        )

        box = metrics.box
        for class_id, class_name in metrics.names.items():
            row = {
                "model": model_name,
                "model_path": str(model_path),
                "class_id": int(class_id),
                "class_name": str(class_name),
                "precision": float(box.p[class_id]),
                "recall": float(box.r[class_id]),
                "map50": float(box.ap50[class_id]),
                "map50_95": float(box.ap[class_id]),
            }
            rows.append(row)
            print(
                f"{class_name:10} P={row['precision']:.3f} "
                f"R={row['recall']:.3f} mAP50={row['map50']:.3f} "
                f"mAP50-95={row['map50_95']:.3f}"
            )
        rows.append(
            {
                "model": model_name,
                "model_path": str(model_path),
                "class_id": -1,
                "class_name": "all",
                "precision": float(box.mp),
                "recall": float(box.mr),
                "map50": float(box.map50),
                "map50_95": float(box.map),
            }
        )
        print(f"{'all':10} P={box.mp:.3f} R={box.mr:.3f} mAP50={box.map50:.3f} mAP50-95={box.map:.3f}")

    ball_rows = [row for row in rows if row["class_name"] == "ball"]
    ball_rows.sort(key=lambda row: (row["recall"], row["map50"], row["map50_95"]), reverse=True)
    if ball_rows:
        print("\nBall ranking by recall, then mAP50")
        for rank, row in enumerate(ball_rows, start=1):
            print(
                f"{rank}. {row['model']}: R={row['recall']:.3f} "
                f"mAP50={row['map50']:.3f} P={row['precision']:.3f}"
            )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nWrote JSON metrics to {args.out}")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV metrics to {args.csv_out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class YoloBox:
    x_center: float
    y_center: float
    width: float
    height: float

    def xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.x_center - self.width / 2,
            self.y_center - self.height / 2,
            self.x_center + self.width / 2,
            self.y_center + self.height / 2,
        )


@dataclass(frozen=True)
class BallLabel:
    image_path: str
    label_path: str
    image_width: int
    image_height: int
    box: YoloBox

    @property
    def width_px(self) -> float:
        return self.box.width * self.image_width

    @property
    def height_px(self) -> float:
        return self.box.height * self.image_height

    @property
    def area_px(self) -> float:
        return self.width_px * self.height_px


@dataclass(frozen=True)
class Prediction:
    confidence: float
    class_id: int
    box: YoloBox


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ball detection misses by ball size and confidence.")
    parser.add_argument("--data", type=Path, required=True, help="YOLO data.yaml.")
    parser.add_argument("--model", type=Path, required=True, help="YOLO checkpoint.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.01, help="Low prediction confidence used for analysis.")
    parser.add_argument("--iou", type=float, default=0.25, help="IoU threshold for matching a predicted ball.")
    parser.add_argument(
        "--size-buckets",
        default="0:20,20:28,28:40,40:9999",
        help="Comma-separated pixel width buckets formatted as min:max.",
    )
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    parser.add_argument("--csv-out", type=Path, help="Optional per-label CSV output path.")
    args = parser.parse_args()

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install training dependencies before running this script.") from exc

    dataset = _read_data_yaml(args.data)
    labels = _read_ball_labels(dataset, args.split, cv2)
    model = YOLO(str(args.model))
    predictions_by_image = _predict_by_image(model, labels, args.imgsz, args.conf, args.device)
    buckets = _parse_size_buckets(args.size_buckets)

    rows = []
    bucket_stats = {
        bucket_name: {"labels": 0, "matched": 0, "missed": 0, "low_conf_match": 0}
        for bucket_name, _, _ in buckets
    }
    for label in labels:
        predictions = predictions_by_image.get(label.image_path, [])
        best_ball = _best_match(label.box, [prediction for prediction in predictions if prediction.class_id == 0])
        best_any = _best_match(label.box, predictions)
        matched = best_ball["iou"] >= args.iou
        bucket_name = _bucket_for_width(label.width_px, buckets)
        stats = bucket_stats[bucket_name]
        stats["labels"] += 1
        stats["matched" if matched else "missed"] += 1
        if matched and best_ball["confidence"] < 0.10:
            stats["low_conf_match"] += 1
        rows.append(
            {
                "image_path": label.image_path,
                "label_path": label.label_path,
                "width_px": label.width_px,
                "height_px": label.height_px,
                "area_px": label.area_px,
                "bucket": bucket_name,
                "matched": matched,
                "best_ball_iou": best_ball["iou"],
                "best_ball_confidence": best_ball["confidence"],
                "best_any_iou": best_any["iou"],
                "best_any_confidence": best_any["confidence"],
                "best_any_class_id": best_any["class_id"],
            }
        )

    summary = {
        "data": str(args.data),
        "model": str(args.model),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "labels": len(labels),
        "matched": sum(1 for row in rows if row["matched"]),
        "missed": sum(1 for row in rows if not row["matched"]),
        "buckets": bucket_stats,
    }
    for stats in bucket_stats.values():
        stats["recall"] = _safe_divide(stats["matched"], stats["labels"])

    print(json.dumps(summary, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
        print(f"Wrote JSON analysis to {args.out}")
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV analysis to {args.csv_out}")


def _read_data_yaml(data_yaml: Path) -> dict:
    import yaml

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload["path"]).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return {
        "path": root.resolve(),
        "train": payload["train"],
        "val": payload["val"],
        "test": payload.get("test"),
    }


def _read_ball_labels(dataset: dict, split: str, cv2) -> list[BallLabel]:
    image_dir = dataset["path"] / dataset[split]
    label_dir = dataset["path"] / dataset[split].replace("/images", "/labels")
    labels: list[BallLabel] = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image_h, image_w = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) < 5 or int(float(parts[0])) != 0:
                continue
            values = [float(value) for value in parts[1:5]]
            labels.append(
                BallLabel(
                    image_path=str(image_path),
                    label_path=str(label_path),
                    image_width=image_w,
                    image_height=image_h,
                    box=YoloBox(values[0], values[1], values[2], values[3]),
                )
            )
    return labels


def _predict_by_image(model, labels: list[BallLabel], imgsz: int, conf: float, device: str) -> dict[str, list[Prediction]]:
    predictions_by_image: dict[str, list[Prediction]] = {}
    for image_path in sorted({label.image_path for label in labels}):
        result = model.predict(source=image_path, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        predictions: list[Prediction] = []
        if result.boxes is not None:
            for class_id, confidence, box in zip(
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.xywhn.cpu().tolist(),
            ):
                predictions.append(
                    Prediction(
                        confidence=float(confidence),
                        class_id=int(class_id),
                        box=YoloBox(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    )
                )
        predictions_by_image[image_path] = predictions
    return predictions_by_image


def _best_match(label: YoloBox, predictions: list[Prediction]) -> dict[str, float | int]:
    best = {"iou": 0.0, "confidence": 0.0, "class_id": -1}
    for prediction in predictions:
        iou = _iou(label, prediction.box)
        if iou > best["iou"]:
            best = {"iou": iou, "confidence": prediction.confidence, "class_id": prediction.class_id}
    return best


def _parse_size_buckets(raw_value: str) -> list[tuple[str, float, float]]:
    buckets = []
    for part in raw_value.split(","):
        if not part.strip():
            continue
        min_value, max_value = [float(value) for value in part.split(":", 1)]
        buckets.append((f"{min_value:g}-{max_value:g}px", min_value, max_value))
    if not buckets:
        raise ValueError("--size-buckets must include at least one bucket")
    return buckets


def _bucket_for_width(width: float, buckets: list[tuple[str, float, float]]) -> str:
    for bucket_name, min_value, max_value in buckets:
        if min_value <= width < max_value:
            return bucket_name
    return buckets[-1][0]


def _iou(first: YoloBox, second: YoloBox) -> float:
    ax1, ay1, ax2, ay2 = first.xyxy()
    bx1, by1, bx2, by2 = second.xyxy()
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return _safe_divide(intersection, first_area + second_area - intersection)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    main()

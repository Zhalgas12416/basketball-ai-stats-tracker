from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
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
class Prediction:
    confidence: float
    box: YoloBox


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep ball confidence thresholds on a YOLO dataset split.")
    parser.add_argument("--data", type=Path, required=True, help="YOLO data.yaml.")
    parser.add_argument("--model", type=Path, required=True, help="YOLO checkpoint.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iou", type=float, default=0.25)
    parser.add_argument(
        "--thresholds",
        default="0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.40,0.50",
        help="Comma-separated confidence thresholds to score.",
    )
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    parser.add_argument("--csv-out", type=Path, help="Optional CSV output path.")
    args = parser.parse_args()

    thresholds = sorted(_parse_thresholds(args.thresholds))
    if not thresholds:
        raise SystemExit("--thresholds must include at least one value")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: pip install -e '.[train]'") from exc

    dataset = _read_data_yaml(args.data)
    image_dir = dataset["path"] / dataset[args.split]
    label_dir = dataset["path"] / dataset[args.split].replace("/images", "/labels")
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

    model = YOLO(str(args.model))
    min_conf = min(thresholds)
    scored_images = []
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = _read_ball_labels(label_path)
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=min_conf,
            device=args.device,
            verbose=False,
        )[0]
        predictions = _read_ball_predictions(result)
        scored_images.append((labels, predictions))

    rows = []
    for threshold in thresholds:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        frames_with_gt_ball = 0
        frames_with_pred_ball = 0
        for labels, predictions in scored_images:
            threshold_predictions = [prediction for prediction in predictions if prediction.confidence >= threshold]
            if labels:
                frames_with_gt_ball += 1
            if threshold_predictions:
                frames_with_pred_ball += 1

            tp, fp, fn = _match_counts(labels, threshold_predictions, args.iou)
            true_positives += tp
            false_positives += fp
            false_negatives += fn

        precision = _safe_divide(true_positives, true_positives + false_positives)
        recall = _safe_divide(true_positives, true_positives + false_negatives)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        rows.append(
            {
                "model": args.model.stem,
                "threshold": threshold,
                "iou": args.iou,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "frames": len(scored_images),
                "frames_with_gt_ball": frames_with_gt_ball,
                "frames_with_pred_ball": frames_with_pred_ball,
            }
        )

    print("threshold precision recall f1 tp fp fn")
    for row in rows:
        print(
            f"{row['threshold']:.3f} "
            f"{row['precision']:.3f} "
            f"{row['recall']:.3f} "
            f"{row['f1']:.3f} "
            f"{row['true_positives']} "
            f"{row['false_positives']} "
            f"{row['false_negatives']}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Wrote JSON sweep to {args.out}")
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV sweep to {args.csv_out}")


def _parse_thresholds(raw_value: str) -> list[float]:
    return [float(part.strip()) for part in raw_value.split(",") if part.strip()]


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


def _read_ball_labels(label_path: Path) -> list[YoloBox]:
    if not label_path.exists():
        return []
    labels: list[YoloBox] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5 or int(float(parts[0])) != 0:
            continue
        values = [float(value) for value in parts[1:5]]
        labels.append(YoloBox(values[0], values[1], values[2], values[3]))
    return labels


def _read_ball_predictions(result) -> list[Prediction]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    predictions: list[Prediction] = []
    for class_id, confidence, box in zip(
        result.boxes.cls.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.xywhn.cpu().tolist(),
    ):
        if int(class_id) != 0:
            continue
        predictions.append(
            Prediction(
                confidence=float(confidence),
                box=YoloBox(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            )
        )
    return predictions


def _match_counts(labels: list[YoloBox], predictions: list[Prediction], iou_threshold: float) -> tuple[int, int, int]:
    matched_label_indexes: set[int] = set()
    true_positives = 0
    false_positives = 0

    for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        best_index = -1
        best_iou = 0.0
        for index, label in enumerate(labels):
            if index in matched_label_indexes:
                continue
            iou = _iou(prediction.box, label)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched_label_indexes.add(best_index)
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(labels) - len(matched_label_indexes)
    return true_positives, false_positives, false_negatives


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
    union = first_area + second_area - intersection
    return _safe_divide(intersection, union)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    main()

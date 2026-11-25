from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_NAMES = ["player", "ball", "rim", "backboard", "referee"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a YOLO detection dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/basketball"))
    args = parser.parse_args()

    class_names = _read_class_names(args.dataset)
    total_errors = 0
    for split in ["train", "val", "test"]:
        result = _validate_split(args.dataset, split, class_names)
        if result["missing_split"]:
            continue
        total_errors += result["errors"]
        print(f"\n{split}")
        print(f"  images: {result['images']}")
        print(f"  labels: {result['label_files']}")
        print(f"  empty_label_files: {result['empty_label_files']}")
        print(f"  missing_label_files: {result['missing_label_files']}")
        print(f"  labels_without_images: {result['labels_without_images']}")
        print(f"  errors: {result['errors']}")
        for class_id, count in sorted(result["class_counts"].items()):
            class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
            print(f"  {class_name}: {count}")

    if total_errors:
        raise SystemExit(1)


def _validate_split(dataset: Path, split: str, class_names: list[str]) -> dict:
    image_dir, label_dir = _split_dirs(dataset, split)
    if image_dir is None or label_dir is None:
        return {
            "missing_split": True,
            "images": 0,
            "label_files": 0,
            "empty_label_files": 0,
            "missing_label_files": 0,
            "labels_without_images": 0,
            "class_counts": Counter(),
            "errors": 0,
        }
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    labels = sorted(path for path in label_dir.iterdir() if path.suffix == ".txt")
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    class_counts = Counter()
    empty_label_files = 0
    errors = 0

    for label_path in labels:
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty_label_files += 1
            continue
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                errors += 1
                continue
            try:
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
            except ValueError:
                errors += 1
                continue
            if class_id < 0 or class_id >= len(class_names):
                errors += 1
                continue
            if any(value < 0.0 or value > 1.0 for value in values):
                errors += 1
                continue
            if values[2] <= 0.0 or values[3] <= 0.0:
                errors += 1
                continue
            class_counts[class_id] += 1

    return {
        "missing_split": False,
        "images": len(images),
        "label_files": len(labels),
        "empty_label_files": empty_label_files,
        "missing_label_files": len(image_stems - label_stems),
        "labels_without_images": len(label_stems - image_stems),
        "class_counts": class_counts,
        "errors": errors,
    }


def _split_dirs(dataset: Path, split: str) -> tuple[Path | None, Path | None]:
    candidates = [
        (dataset / split / "images", dataset / split / "labels"),
        (dataset / "images" / split, dataset / "labels" / split),
    ]
    for image_dir, label_dir in candidates:
        if image_dir.exists() and label_dir.exists():
            return image_dir, label_dir
    return None, None


def _read_class_names(dataset: Path) -> list[str]:
    data_yaml = dataset / "data.yaml"
    if not data_yaml.exists():
        return CLASS_NAMES
    try:
        import yaml
    except ImportError:
        return CLASS_NAMES
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = payload.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        return [str(name) for _, name in sorted((int(key), value) for key, value in names.items())]
    return CLASS_NAMES


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_NAMES = ["player", "ball", "rim", "backboard", "referee"]
CLASS_ALIASES = {
    "person": "player",
    "people": "player",
    "player": "player",
    "players": "player",
    "basketball player": "player",
    "basketball-player": "player",
    "human": "player",
    "player in possession": "player",
    "player jump shot": "player",
    "player layup dunk": "player",
    "player shot block": "player",
    "ball": "ball",
    "basketball": "ball",
    "ball in basket": "ball",
    "sports ball": "ball",
    "sports-ball": "ball",
    "rim": "rim",
    "hoop": "rim",
    "basket": "rim",
    "net": "rim",
    "basketball net": "rim",
    "basketball rim": "rim",
    "basketball hoop": "rim",
    "backboard": "backboard",
    "back board": "backboard",
    "back-board": "backboard",
    "basketball backboard": "backboard",
    "ref": "referee",
    "referee": "referee",
    "official": "referee",
}


@dataclass(frozen=True)
class SplitPaths:
    source_split: str
    split: str
    image_dir: Path
    label_dir: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge YOLO/Roboflow datasets into one basketball dataset.")
    parser.add_argument("datasets", nargs="+", type=Path, help="Dataset folders or .zip exports to merge.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets" / "basketball")
    parser.add_argument("--skip-empty", action="store_true", help="Skip images with no mapped labels.")
    parser.add_argument("--clean-output", action="store_true", help="Delete existing merged images/labels first.")
    args = parser.parse_args()

    output = args.output
    if args.clean_output:
        _clean_output_dirs(output)
    _ensure_output_dirs(output)

    summary = Counter()
    skipped_unknown = Counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        for dataset_index, input_path in enumerate(args.datasets, start=1):
            dataset_path = _materialize_dataset(input_path.expanduser().resolve(), tmp_root, dataset_index)
            source_names = _load_source_names(dataset_path)
            class_map = _build_class_map(source_names)
            split_paths = _find_split_paths(dataset_path)
            dataset_slug = _safe_slug(input_path.stem if input_path.suffix == ".zip" else input_path.name)

            print(f"Merging {input_path}")
            print(f"  classes: {source_names}")
            print(f"  mapped: {class_map}")

            for split in split_paths:
                split_summary, split_skipped = _merge_split(
                    dataset_index,
                    dataset_slug,
                    split,
                    output,
                    class_map,
                    args.skip_empty,
                )
                summary.update(split_summary)
                skipped_unknown.update(split_skipped)

    print("\nMerged dataset summary")
    for key, count in sorted(summary.items()):
        print(f"  {key}: {count}")
    if skipped_unknown:
        print("\nSkipped unknown source classes")
        for key, count in sorted(skipped_unknown.items()):
            print(f"  {key}: {count}")


def _ensure_output_dirs(output: Path) -> None:
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def _clean_output_dirs(output: Path) -> None:
    for relative_path in ["images/train", "images/val", "labels/train", "labels/val"]:
        directory = output / relative_path
        if directory.exists():
            shutil.rmtree(directory)


def _materialize_dataset(input_path: Path, tmp_root: Path, dataset_index: int) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    if input_path.suffix.lower() != ".zip":
        return input_path

    target = tmp_root / f"dataset_{dataset_index:02d}"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_path) as archive:
        archive.extractall(target)
    return target


def _load_source_names(dataset_path: Path) -> dict[int, str]:
    data_yaml = _find_data_yaml(dataset_path)
    if data_yaml is None:
        return {}

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read data.yaml.") from exc

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = payload.get("names", {})
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    return {}


def _find_data_yaml(dataset_path: Path) -> Path | None:
    for candidate in [dataset_path / "data.yaml", dataset_path / "data.yml"]:
        if candidate.exists():
            return candidate
    matches = list(dataset_path.rglob("data.yaml")) + list(dataset_path.rglob("data.yml"))
    return matches[0] if matches else None


def _build_class_map(source_names: dict[int, str]) -> dict[int, int]:
    if not source_names:
        return {index: index for index in range(len(TARGET_NAMES))}

    target_index = {name: index for index, name in enumerate(TARGET_NAMES)}
    class_map: dict[int, int] = {}
    for source_index, source_name in source_names.items():
        normalized = _normalize_class_name(source_name)
        target_name = CLASS_ALIASES.get(normalized)
        if target_name is None:
            continue
        class_map[source_index] = target_index[target_name]
    return class_map


def _find_split_paths(dataset_path: Path) -> list[SplitPaths]:
    split_paths: list[SplitPaths] = []
    for source_split, target_split in [("train", "train"), ("valid", "val"), ("val", "val"), ("test", "val")]:
        candidates = [
            (dataset_path / source_split / "images", dataset_path / source_split / "labels"),
            (dataset_path / "images" / source_split, dataset_path / "labels" / source_split),
        ]
        for image_dir, label_dir in candidates:
            if image_dir.exists() and label_dir.exists():
                split_paths.append(SplitPaths(source_split, target_split, image_dir, label_dir))
                break

    if split_paths:
        return split_paths

    image_dir = dataset_path / "images"
    label_dir = dataset_path / "labels"
    if image_dir.exists() and label_dir.exists():
        return [SplitPaths("train", "train", image_dir, label_dir)]

    raise ValueError(f"Could not find YOLO image/label folders in {dataset_path}")


def _merge_split(
    dataset_index: int,
    dataset_slug: str,
    split: SplitPaths,
    output: Path,
    class_map: dict[int, int],
    skip_empty: bool,
) -> tuple[Counter, Counter]:
    summary = Counter()
    skipped_unknown = Counter()
    images = sorted(path for path in split.image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

    for image_index, image_path in enumerate(images, start=1):
        label_path = split.label_dir / f"{image_path.stem}.txt"
        mapped_lines, skipped = _read_mapped_label_lines(label_path, class_map)
        skipped_unknown.update(skipped)
        if skip_empty and not mapped_lines:
            summary[f"{split.split}_empty_skipped"] += 1
            continue

        target_stem = f"d{dataset_index:02d}_{dataset_slug}_{split.source_split}_{image_index:06d}"
        target_image = output / "images" / split.split / f"{target_stem}{image_path.suffix.lower()}"
        target_label = output / "labels" / split.split / f"{target_stem}.txt"
        shutil.copy2(image_path, target_image)
        target_label.write_text("\n".join(mapped_lines) + ("\n" if mapped_lines else ""), encoding="utf-8")

        summary[f"{split.split}_images"] += 1
        for line in mapped_lines:
            class_id = int(line.split()[0])
            summary[f"{split.split}_{TARGET_NAMES[class_id]}"] += 1

    return summary, skipped_unknown


def _read_mapped_label_lines(label_path: Path, class_map: dict[int, int]) -> tuple[list[str], Counter]:
    if not label_path.exists():
        return [], Counter()

    mapped_lines: list[str] = []
    skipped = Counter()
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        source_class = int(float(parts[0]))
        target_class = class_map.get(source_class)
        if target_class is None:
            skipped[str(source_class)] += 1
            continue
        mapped_lines.append(" ".join([str(target_class), *parts[1:5]]))
    return mapped_lines, skipped


def _normalize_class_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


def _safe_slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_") or "dataset"


if __name__ == "__main__":
    main()

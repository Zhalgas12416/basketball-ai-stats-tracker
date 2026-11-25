from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a YOLO dataset from a base dataset plus extra hard train examples."
    )
    parser.add_argument("--base", type=Path, required=True, help="Base YOLO data.yaml.")
    parser.add_argument(
        "--extra",
        type=Path,
        action="append",
        required=True,
        help="Extra YOLO data.yaml with hard train examples. Repeat to mix multiple hard datasets.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-limit", type=int, default=0, help="Optional max base train images to include.")
    parser.add_argument("--extra-copies", type=int, default=1, help="How many times to include extra train examples.")
    parser.add_argument("--copy-files", action="store_true", help="Copy instead of symlinking.")
    args = parser.parse_args()

    base = _read_data_yaml(args.base)
    extras = [_read_data_yaml(extra_data) for extra_data in args.extra]
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    base_train_count = _copy_split(
        source_root=base["path"],
        split_path=base["train"],
        output_root=output,
        output_split="train",
        prefix="base",
        limit=args.base_limit or None,
        copy_files=args.copy_files,
    )
    extra_train_count = 0
    for extra_index, extra in enumerate(extras, start=1):
        for copy_index in range(1, args.extra_copies + 1):
            extra_train_count += _copy_split(
                source_root=extra["path"],
                split_path=extra["train"],
                output_root=output,
                output_split="train",
                prefix=f"hard{extra_index:02d}_{copy_index:02d}",
                limit=None,
                copy_files=args.copy_files,
            )

    val_count = _copy_split(
        source_root=base["path"],
        split_path=base["val"],
        output_root=output,
        output_split="val",
        prefix="base",
        limit=None,
        copy_files=args.copy_files,
    )
    test_count = 0
    if base.get("test"):
        test_count = _copy_split(
            source_root=base["path"],
            split_path=base["test"],
            output_root=output,
            output_split="test",
            prefix="base",
            limit=None,
            copy_files=args.copy_files,
        )

    _write_data_yaml(output, base["names"])
    _write_readme(
        output=output,
        base_data=args.base,
        extra_data=args.extra,
        base_train_count=base_train_count,
        extra_train_count=extra_train_count,
        val_count=val_count,
        test_count=test_count,
        extra_copies=args.extra_copies,
    )
    print(f"Built hard-mix dataset at {output}")
    print(f"  base_train: {base_train_count}")
    print(f"  extra_train: {extra_train_count}")
    print(f"  val: {val_count}")
    print(f"  test: {test_count}")


def _read_data_yaml(data_yaml: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YOLO data.yaml files.") from exc

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    source_root = Path(payload["path"]).expanduser()
    if not source_root.is_absolute():
        source_root = (data_yaml.parent / source_root).resolve()
    names = payload.get("names", {})
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}
    else:
        names = {int(index): name for index, name in names.items()}
    return {
        "path": source_root.resolve(),
        "train": payload["train"],
        "val": payload["val"],
        "test": payload.get("test"),
        "names": names,
    }


def _copy_split(
    *,
    source_root: Path,
    split_path: str,
    output_root: Path,
    output_split: str,
    prefix: str,
    limit: int | None,
    copy_files: bool,
) -> int:
    image_dir = source_root / split_path
    label_dir = source_root / split_path.replace("/images", "/labels")
    output_image_dir = output_root / output_split / "images"
    output_label_dir = output_root / output_split / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    for image_path in image_paths:
        if limit is not None and count >= limit:
            break
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        target_stem = f"{prefix}_{image_path.stem}"
        _symlink_or_copy(
            image_path,
            output_image_dir / f"{target_stem}{image_path.suffix.lower()}",
            copy_files=copy_files,
        )
        _symlink_or_copy(
            label_path,
            output_label_dir / f"{target_stem}.txt",
            copy_files=copy_files,
        )
        count += 1
    return count


def _symlink_or_copy(source: Path, target: Path, *, copy_files: bool) -> None:
    if target.exists() or target.is_symlink():
        return
    if copy_files:
        shutil.copy2(source, target)
        return
    try:
        relative_source = os.path.relpath(source, start=target.parent)
        target.symlink_to(relative_source)
    except OSError:
        shutil.copy2(source, target)


def _write_data_yaml(output: Path, names: dict[int, str]) -> None:
    names_yaml = "\n".join(f"  {class_id}: {class_name}" for class_id, class_name in sorted(names.items()))
    payload = f"""path: {output}
train: train/images
val: val/images
test: test/images

names:
{names_yaml}
"""
    (output / "data.yaml").write_text(payload, encoding="utf-8")


def _write_readme(
    *,
    output: Path,
    base_data: Path,
    extra_data: list[Path],
    base_train_count: int,
    extra_train_count: int,
    val_count: int,
    test_count: int,
    extra_copies: int,
) -> None:
    readme = f"""# YOLO Hard-Mix Dataset

Created with `scripts/build_yolo_hard_mix_dataset.py`.

- base: `{base_data}`
- extra: `{", ".join(str(path) for path in extra_data)}`
- extra_copies: {extra_copies}
- base_train: {base_train_count}
- extra_train: {extra_train_count}
- val: {val_count}
- test: {test_count}
- output: `{output}`
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MergeYOLODatasetsTest(unittest.TestCase):
    def test_merges_roboflow_style_yolo_exports_with_class_remap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_a = root / "dataset_a"
            dataset_b = root / "dataset_b"
            output = root / "merged"
            _write_dataset(
                dataset_a,
                split="train",
                image_name="a.jpg",
                label_text="0 0.5 0.5 0.2 0.4\n1 0.2 0.2 0.05 0.05\n",
                data_yaml="names:\n  0: person\n  1: sports ball\n",
            )
            _write_dataset(
                dataset_b,
                split="valid",
                image_name="b.png",
                label_text="0 0.6 0.6 0.1 0.1\n",
                data_yaml="names:\n  0: back-board\n",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/merge_yolo_datasets.py",
                    str(dataset_a),
                    str(dataset_b),
                    "--output",
                    str(output),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            train_labels = sorted((output / "labels" / "train").glob("*.txt"))
            val_labels = sorted((output / "labels" / "val").glob("*.txt"))
            self.assertEqual(len(train_labels), 1)
            self.assertEqual(len(val_labels), 1)
            self.assertEqual(train_labels[0].read_text(encoding="utf-8").splitlines()[0].split()[0], "0")
            self.assertEqual(train_labels[0].read_text(encoding="utf-8").splitlines()[1].split()[0], "1")
            self.assertEqual(val_labels[0].read_text(encoding="utf-8").split()[0], "3")

    def test_keeps_valid_and_test_files_when_both_map_to_val(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset"
            output = root / "merged"
            _write_dataset(
                dataset,
                split="valid",
                image_name="same.jpg",
                label_text="0 0.5 0.5 0.2 0.4\n",
                data_yaml="names:\n  0: player\n",
            )
            _write_dataset(
                dataset,
                split="test",
                image_name="same.jpg",
                label_text="0 0.4 0.4 0.2 0.4\n",
                data_yaml="names:\n  0: player\n",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/merge_yolo_datasets.py",
                    str(dataset),
                    "--output",
                    str(output),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(list((output / "labels" / "val").glob("*.txt"))), 2)


def _write_dataset(dataset: Path, split: str, image_name: str, label_text: str, data_yaml: str) -> None:
    image_dir = dataset / split / "images"
    label_dir = dataset / split / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (dataset / "data.yaml").write_text(data_yaml, encoding="utf-8")
    (image_dir / image_name).write_bytes(b"fake image bytes")
    (label_dir / f"{Path(image_name).stem}.txt").write_text(label_text, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

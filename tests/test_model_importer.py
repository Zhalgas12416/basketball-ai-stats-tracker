from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from basketball_ai_tracker.vision.model_importer import (
    build_model_compatibility_report,
    install_yolo_weights,
    write_external_model_attribution,
)


class ModelImporterTest(unittest.TestCase):
    def test_reports_mapped_and_missing_external_classes(self) -> None:
        report = build_model_compatibility_report(
            ["basketball player", "basketball", "basketball hoop", "official", "scoreboard"]
        )

        self.assertEqual(report["mapped_classes"]["player"], ["basketball player"])
        self.assertEqual(report["mapped_classes"]["ball"], ["basketball"])
        self.assertEqual(report["mapped_classes"]["rim"], ["basketball hoop"])
        self.assertEqual(report["mapped_classes"]["referee"], ["official"])
        self.assertEqual(report["unmapped_classes"], ["scoreboard"])
        self.assertEqual(report["missing_target_classes"], ["backboard"])
        self.assertFalse(report["compatible"])

    def test_installs_weights_without_requiring_ultralytics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "external.pt"
            target = root / "models" / "basketball-detector.pt"
            source.write_bytes(b"weights")

            installed = install_yolo_weights(source, target)

            self.assertEqual(installed, target)
            self.assertEqual(target.read_bytes(), b"weights")

    def test_writes_external_model_attribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_external_model_attribution(
                Path(directory),
                [
                    {
                        "name": "Example Model",
                        "source": "https://example.test/model",
                        "license": "cc-by-4.0",
                        "local_file": "models/external/example.pt",
                    }
                ],
            )

            text = path.read_text(encoding="utf-8")
            self.assertIn("Example Model", text)
            self.assertIn("cc-by-4.0", text)


if __name__ == "__main__":
    unittest.main()

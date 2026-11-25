import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    from basketball_ai_tracker.api import main as api_main
except ImportError:
    TestClient = None
    api_main = None


@unittest.skipIf(TestClient is None, "FastAPI test dependencies are not installed.")
class ApiSmokeTest(unittest.TestCase):
    def test_health_reports_active_analyzer(self) -> None:
        assert api_main is not None
        client = TestClient(api_main.app)

        response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("analyzer", payload)
        self.assertIn("detector", payload["analyzer"])
        self.assertIn("engine", payload["analyzer"]["detector"])

    def test_upload_game_produces_timestamped_result(self) -> None:
        assert api_main is not None
        client = TestClient(api_main.app)

        upload = client.post(
            "/api/games",
            files={"video": ("game.mp4", b"fake video bytes", "video/mp4")},
        )
        self.assertEqual(upload.status_code, 200)

        game_id = upload.json()["job"]["game_id"]
        try:
            video_response = client.get(f"/api/games/{game_id}/video")
            self.assertEqual(video_response.status_code, 200)
            self.assertEqual(video_response.content, b"fake video bytes")

            response = client.get(f"/api/games/{game_id}")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["job"]["status"], "completed")
            self.assertIn("stats", payload["result"])
            self.assertIn("warnings", payload["result"])

            quality = client.get(f"/api/games/{game_id}/quality")
            self.assertEqual(quality.status_code, 200)
            self.assertIn("class_counts", quality.json()["quality"])
        finally:
            record = api_main.job_store.get(game_id)
            if record is not None:
                record.video_path.unlink(missing_ok=True)
                record.result_path.unlink(missing_ok=True)

    def test_game_corrections_can_be_saved_exported_and_deleted(self) -> None:
        assert api_main is not None
        client = TestClient(api_main.app)

        upload = client.post(
            "/api/games",
            files={"video": ("game.mp4", b"fake video bytes", "video/mp4")},
        )
        self.assertEqual(upload.status_code, 200)
        game_id = upload.json()["job"]["game_id"]

        try:
            create = client.post(
                f"/api/games/{game_id}/corrections",
                json={
                    "label": "player",
                    "bbox": {"x": 10, "y": 20, "width": 30, "height": 40},
                    "video_time_seconds": 109,
                },
            )
            self.assertEqual(create.status_code, 200)
            correction = create.json()["correction"]
            self.assertEqual(correction["label"], "player")
            self.assertEqual(correction["timecode"], "1:49")

            listed = client.get(f"/api/games/{game_id}/corrections")
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(len(listed.json()["corrections"]), 1)

            exported = client.get(f"/api/games/{game_id}/corrections/export")
            self.assertEqual(exported.status_code, 200)
            self.assertEqual(exported.json()["schema"], "basketball-detection-corrections-v1")

            deleted = client.delete(f"/api/games/{game_id}/corrections/{correction['correction_id']}")
            self.assertEqual(deleted.status_code, 200)
            self.assertEqual(deleted.json()["corrections"], [])
        finally:
            record = api_main.job_store.get(game_id)
            if record is not None:
                record.video_path.unlink(missing_ok=True)
                record.result_path.unlink(missing_ok=True)
            api_main._correction_path(game_id).unlink(missing_ok=True)

    def test_quality_compare_accepts_vision_payloads(self) -> None:
        assert api_main is not None
        client = TestClient(api_main.app)

        response = client.post(
            "/api/quality/compare",
            json={
                "before": {
                    "vision": {
                        "sampled_frame_count": 2,
                        "detections": [{"frame_index": 0, "label": "player", "confidence": 0.2}],
                    }
                },
                "after": {
                    "vision": {
                        "sampled_frame_count": 2,
                        "detections": [
                            {"frame_index": 0, "label": "player", "confidence": 0.8},
                            {"frame_index": 1, "label": "ball", "confidence": 0.7},
                        ],
                    }
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        comparison = response.json()["comparison"]
        self.assertEqual(comparison["detection_count_delta"], 1)
        self.assertEqual(comparison["low_confidence_count_delta"], -1)

    def test_hard_frame_queue_uses_completed_vision_result(self) -> None:
        assert api_main is not None
        client = TestClient(api_main.app)
        game_id = "test-hard-frame-api"
        video_path = api_main.UPLOAD_DIR / f"{game_id}.mp4"
        result_path = api_main.RESULT_DIR / f"{game_id}.json"
        record = api_main.JobRecord(
            game_id=game_id,
            original_filename="game.mp4",
            video_path=video_path,
            result_path=result_path,
            status="completed",
            progress=1.0,
        )
        video_path.write_bytes(b"fake video bytes")
        result_path.write_text(
            """
            {
              "game_id": "test-hard-frame-api",
              "stats": [],
              "events": [],
              "vision": {
                "fps": 30,
                "frame_count": 90,
                "duration_seconds": 3,
                "detections": [
                  {"frame_index": 30, "time_seconds": 1.0, "label": "player", "confidence": 0.9}
                ]
              }
            }
            """,
            encoding="utf-8",
        )
        api_main.job_store.create(record)

        try:
            response = client.post(
                f"/api/games/{game_id}/hard-frames",
                json={"classes": ["ball", "referee"], "max_frames": 2},
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["schema"], "basketball-hard-frame-queue-v1")
            self.assertLessEqual(len(payload["frames"]), 2)
            self.assertTrue(payload["frames"])
            self.assertIn("missing_ball", payload["frames"][0]["reasons"])
        finally:
            api_main.job_store._records.pop(game_id, None)
            video_path.unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

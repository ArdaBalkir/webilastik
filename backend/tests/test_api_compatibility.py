from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch

import numpy as np

from backend.classifier import Classifier
from backend import server, session_allocator


class ApiCompatibilityTests(unittest.TestCase):
    def test_browser_facing_routes_are_preserved(self):
        paths = {route.path for route in server.app.routes}
        expected = {
            "/health",
            "/dzi-info",
            "/list-sources",
            "/train",
            "/train-multi",
            "/predict/{classifier_id}/{level}/{tile_spec}",
            "/export",
            "/export/{job_id}",
            "/batch-export",
            "/batch-export/{job_id}",
            "/export-zip",
            "/headless-run",
        }
        self.assertTrue(expected.issubset(paths), expected - paths)

    def test_train_response_keeps_original_fields(self):
        fields = server.TrainResponse.model_fields
        self.assertIn("classifier_id", fields)
        self.assertIn("num_classes", fields)
        self.assertIn("generation", fields)
        self.assertIn("worker_id", fields)

    def test_health_keeps_status_and_classifier_count(self):
        report = asyncio.run(server.health())
        self.assertIn("status", report)
        self.assertIn("classifiers", report)
        self.assertEqual(report["scope"], "single_process_single_vm")
        self.assertIn("workers", report)

    def test_cpu_random_forest_uses_explicit_inner_thread_budget(self):
        with patch.dict(os.environ, {"COMPUTE_THREADS_PER_TASK": "1"}):
            classifier = Classifier(force_cpu=True)
            classifier.fit(
                np.array([[0.0], [1.0], [0.1], [0.9]], dtype=np.float32),
                np.array([0, 1, 0, 1], dtype=np.int32),
            )
        self.assertEqual(classifier.compute_threads, 1)
        self.assertEqual(classifier._clf.n_jobs, 1)

    def test_bulk_allocator_remains_slurm_path(self):
        report = asyncio.run(session_allocator.health())
        self.assertEqual(report["role"], "slurm_bulk_export_allocator")
        self.assertEqual(report["execution_path"], "ssh_sbatch")


if __name__ == "__main__":
    unittest.main()

"""GPU-path tests that run without a GPU (numpy/scipy stand in for cupy)."""
import asyncio
import io
import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

from backend import device, features, headless_cli, session_allocator
from backend.classifier import Classifier
from backend.encoding import encode_prediction_png

FILTERS = ["gaussianSmoothing", "laplacianOfGaussian", "gaussianGradientMagnitude",
           "hessianOfGaussianEigenvalues", "structureTensorEigenvalues", "differenceOfGaussians"]
SCALES = [0.7, 1.6]


class DeviceSelection(unittest.TestCase):
    def tearDown(self):
        device._active = None

    def test_modes_without_gpu(self):
        with patch.object(device, "gpu_status", return_value=(False, "no cupy")):
            self.assertEqual(device.configure("auto"), "cpu")
            self.assertFalse(device.use_gpu_features())
            with self.assertRaises(device.GpuUnavailable):
                device.configure("gpu")
        self.assertEqual(device.configure("cpu"), "cpu")
        self.assertFalse(device.use_gpu_classifier(cuml_importable=True))

    def test_unconfigured_keeps_server_autodetect(self):
        device._active = None
        self.assertTrue(device.use_gpu_classifier(cuml_importable=True))
        self.assertFalse(device.use_gpu_features())

    def test_cli_exits_3_when_gpu_requested_but_missing(self):
        tmp = pathlib.Path(tempfile.mkdtemp())
        try:
            ann = tmp / "a.json"
            ann.write_text("[]")
            with patch.object(device, "gpu_status", return_value=(False, "no cupy")), \
                    self.assertRaises(SystemExit) as ex:
                headless_cli.main(["run", "--annotations", str(ann), "--p-source", "x",
                                   "--output-dir", "y", "--device", "gpu"])
            self.assertEqual(ex.exception.code, device.GPU_UNAVAILABLE_EXIT)
        finally:
            shutil.rmtree(tmp)


class DeviceFeatures(unittest.TestCase):
    def test_device_features_match_scipy_formulas(self):
        rng = np.random.default_rng(0)
        tile = rng.integers(0, 255, (40, 56, 3), dtype=np.uint8)
        got = features.extract_features_device(tile, FILTERS, SCALES, ndi, np)
        chans = []
        for ch in range(3):
            plane = tile[:, :, ch].astype(np.float32)
            for s in SCALES:
                for f in FILTERS:
                    r = features._scipy_apply(plane, f, s)
                    chans += [r] if r.ndim == 2 else [r[:, :, i] for i in range(r.shape[-1])]
        want = np.stack(chans, -1).reshape(40 * 56, -1)
        self.assertEqual(got.shape, want.shape)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


class BatchedPrediction(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        X = rng.random((300, 18))  # 3 RGB x 2 scales x 3 filters.astype(np.float32)
        y = np.where(X[:, 0] > 0.5, 2, 5).astype(np.int32)  # non-contiguous labels
        device._active = "cpu"
        self.clf = Classifier(n_estimators=5, max_depth=4)
        self.clf.fit(X, y)
        device._active = None
        self.feat = lambda a: features.extract_features_device(a, FILTERS[:3], SCALES, ndi, np)
        # 3x2 grid, ragged edge tiles, one missing, one undecodable
        self.tiles = {}
        for c in range(3):
            for r in range(2):
                w, h = (32 if c < 2 else 17), (32 if r < 1 else 9)
                self.tiles[(c, r)] = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        del self.tiles[(1, 1)]
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def load(self, c, r):
        if (c, r) == (0, 1):
            raise OSError("corrupt jpeg")
        return self.tiles.get((c, r))

    def test_batched_output_matches_per_tile(self):
        coords = [(c, r) for r in range(2) for c in range(3)]
        errors = headless_cli._predict_tiles_batched(
            coords, self.load, self.feat, self.clf.predict_proba, np.concatenate,
            self.clf.classes_.tolist(), self.tmp, workers=3, batch_tiles=4)
        self.assertEqual(errors, ["0,1: corrupt jpeg"])
        written = sorted(p.name for p in self.tmp.iterdir())
        self.assertEqual(written, ["0_0.png", "1_0.png", "2_0.png", "2_1.png"])
        for (c, r), arr in self.tiles.items():
            if (c, r) == (0, 1):
                continue
            h, w = arr.shape[:2]
            proba = self.clf.predict_proba(self.feat(arr)).reshape(h, w, -1)
            want = encode_prediction_png(proba, self.clf.classes_.tolist())
            self.assertEqual((self.tmp / f"{c}_{r}.png").read_bytes(), want, (c, r))

    def test_device_error_aborts_for_cpu_retry(self):
        def boom(_):
            raise MemoryError("out of memory allocating 1 GB")
        with self.assertRaises(headless_cli.GpuFailure):
            headless_cli._predict_tiles_batched(
                [(0, 0)], self.load, self.feat, boom, np.concatenate, [2, 5], self.tmp, 2)


def _bash():
    # On Windows, bash.exe in System32 is the WSL launcher; prefer Git Bash.
    for c in (r"C:\Program Files\Git\bin\bash.exe", shutil.which("bash")):
        if c and pathlib.Path(c).exists() and "system32" not in c.lower():
            return c
    return None


def _script(device_name):
    return session_allocator._build_sbatch_script(
        job_id="j1", annotations_b64="W10=", t_source=None, p_source="https://x/p/",
        output_dir="https://x/o/", features_json="{}", level=None, token="t'k",
        cpus=8, cache_namespace="a/b", user_id="u", training_sources=[], device=device_name)


class SbatchScripts(unittest.TestCase):
    def test_gpu_script_requests_gpu_and_falls_back(self):
        s = _script("gpu")
        self.assertIn("#SBATCH --partition=gpus\n#SBATCH --gres=gpu:1", s)
        self.assertIn('if [ "$EXIT_CODE" -eq 3 ]', s)
        self.assertIn("conda activate webilastik2-gpu", s)
        self.assertIn("_wi2_run cpu", s)

    def test_cpu_script_unchanged_partition(self):
        s = _script("cpu")
        self.assertIn("#SBATCH --partition=batch", s)
        self.assertNotIn("gres", s)
        self.assertNotIn("webilastik2-gpu", s)

    @unittest.skipUnless(_bash(), "bash not available")
    def test_scripts_are_valid_bash(self):
        for d in ("gpu", "cpu"):
            r = subprocess.run([_bash(), "-n"], input=_script(d), text=True, encoding="utf-8", capture_output=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
        manual = pathlib.Path(__file__).parent.parent / "scripts" / "jusuf_run_gpu.sh"
        r = subprocess.run([_bash(), "-n", str(manual)], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_submission_falls_back_to_cpu_when_gpu_rejected(self):
        req = session_allocator.HeadlessJobRequest(
            annotations=[], p_source="https://x/p/", output_dir="https://x/o/", device="auto")
        ssh = AsyncMock(side_effect=["", RuntimeError("Invalid account or partition"), "4242"])
        with patch.object(session_allocator, "_identity", return_value={"sub": "u"}), \
                patch.object(session_allocator, "validate_url"), \
                patch.object(session_allocator, "_ssh_run", ssh):
            job = asyncio.run(session_allocator.create_headless_job(req, "Bearer t"))
        try:
            self.assertEqual(job["partition"], "batch")
            self.assertEqual(job["slurm_job_id"], "4242")
            self.assertIn("mkdir -p", ssh.call_args_list[0].args[0])
            self.assertIn("--partition=gpus", ssh.call_args_list[1].kwargs["stdin"])
            self.assertIn("--partition=batch", ssh.call_args_list[2].kwargs["stdin"])
        finally:
            session_allocator._jobs.pop(job["job_id"], None)

    def test_cpu_request_never_tries_gpu(self):
        req = session_allocator.HeadlessJobRequest(
            annotations=[], p_source="https://x/p/", output_dir="https://x/o/", device="cpu")
        ssh = AsyncMock(return_value="7")
        with patch.object(session_allocator, "_identity", return_value={"sub": "u"}), \
                patch.object(session_allocator, "validate_url"), \
                patch.object(session_allocator, "_ssh_run", ssh):
            job = asyncio.run(session_allocator.create_headless_job(req, "Bearer t"))
        session_allocator._jobs.pop(job["job_id"], None)
        self.assertEqual(ssh.call_count, 2)  # workspace mkdir + one sbatch
        self.assertEqual(job["partition"], "batch")


if __name__ == "__main__":
    unittest.main()

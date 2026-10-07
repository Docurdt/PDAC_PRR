"""Functional release checks using synthetic data and verified weight archives."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from pdac_prr.inference import Predictor
from pdac_prr.weights import install, verify


ROOT = Path(__file__).resolve().parents[1]


class ArchiveInstallationTests(unittest.TestCase):
    def make_bundle(self, directory):
        bundle = Path(directory)
        data = b"synthetic-checkpoint-for-installer-test"
        name = "model_epoch_1_outloop_1_innerloop_1_val_loss_0.pth"
        item = {"name": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        (bundle / "model_manifest.json").write_text(json.dumps({
            "models": {"prrpath": {"expected_checkpoints": [item]}}
        }))
        return bundle, data, f"weights/prrpath/{name}"

    def test_verified_archive_installs_and_valid_existing_file_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, data, name = self.make_bundle(directory)
            archive = bundle / "weights.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(name, data)
            self.assertEqual(install(archive, bundle)["installed"], [name])
            self.assertTrue(verify(bundle)["complete"])
            self.assertEqual(install(archive, bundle)["retained"], [name])

    def test_changed_content_is_rejected_before_checkpoint_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, data, name = self.make_bundle(directory)
            archive = bundle / "weights.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(name, b"X" * len(data))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                install(archive, bundle)
            self.assertFalse((bundle / name).exists())

    def test_path_outside_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle, data, _ = self.make_bundle(directory)
            archive = bundle / "weights.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../outside.pth", data)
            with self.assertRaisesRegex(ValueError, "Unexpected archive entry"):
                install(archive, bundle)


class FrozenSyntheticInferenceTests(unittest.TestCase):
    def test_prrpath_ensemble_matches_original_class_reference(self):
        import torch
        torch.set_num_threads(2)
        if not verify(ROOT, ["prrpath"])["complete"]:
            self.skipTest("Install the verified PRRpath weight archive for this integration check")
        expected = json.loads((ROOT / "examples/synthetic_path_expected.json").read_text())
        with np.load(ROOT / "examples/synthetic_path_features.npz", allow_pickle=False) as data:
            result = Predictor(ROOT).predict("prrpath", {"ls_features": data["ls_features"]})
        np.testing.assert_allclose(result["scores"], expected["scores"], rtol=0, atol=1e-7)
        np.testing.assert_allclose(result["fold_scores"], expected["fold_scores"], rtol=0, atol=1e-7)
        self.assertEqual(result["checkpoint_count"], 25)


if __name__ == "__main__":
    unittest.main()

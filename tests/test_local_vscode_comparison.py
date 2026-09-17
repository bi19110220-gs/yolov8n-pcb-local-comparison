import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class LocalVSCodeComparisonTests(unittest.TestCase):
    def test_authority_snapshots_are_validation_only_and_hash_attested(self):
        from tools.run_local_vscode_comparison import ORIGINAL_TRAIN_MANIFEST, ORIGINAL_VAL_MANIFEST, ENHANCED_TRAIN_MANIFEST, ENHANCED_VAL_MANIFEST, prepare_data_authorities

        with TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            for manifest in (ORIGINAL_TRAIN_MANIFEST, ORIGINAL_VAL_MANIFEST, ENHANCED_TRAIN_MANIFEST, ENHANCED_VAL_MANIFEST):
                for line in manifest.read_text(encoding="utf-8").splitlines():
                    parts = line.replace("\\", "/").split("/")
                    split, name = parts[-2:]
                    image = dataset / "images" / split / name
                    label = dataset / "labels" / split / (Path(name).stem + ".txt")
                    image.parent.mkdir(parents=True, exist_ok=True)
                    label.parent.mkdir(parents=True, exist_ok=True)
                    image.touch()
                    label.touch()
            with patch("tools.run_local_vscode_comparison.DATASET_ROOT", dataset):
                records = prepare_data_authorities(Path(tmp))

            self.assertEqual(
                records["original"]["train_sha256"],
                "b453321adecbf13015bfc4022aaf787dc6c61135af62f662e393390c26642392",
            )
            self.assertEqual(
                records["original"]["val_sha256"],
                "bdafa2aee661aac03ab8ca7fad1ae551bbd754006454f3ab7a97289e776412a0",
            )
            self.assertEqual(
                records["enhanced"]["train_sha256"],
                "8a9a0712524b457e18c0553bb8b409737d599bcd2c13be66abd75c12489ba6e6",
            )
            for record in records.values():
                text = Path(record["data_yaml"]).read_text(encoding="utf-8")
                self.assertIn("train:", text)
                self.assertIn("val:", text)
                self.assertNotIn("test:", text)
                self.assertFalse(record["test_split_used"])
                self.assertTrue(Path(record["val_manifest"]).is_file())

    def test_original_arguments_match_grouped_v1_baseline_and_use_gpu(self):
        from tools.run_local_vscode_comparison import original_train_args

        args = original_train_args(Path("authority.yaml"), Path("output"))

        self.assertEqual(args["epochs"], 100)
        self.assertEqual(args["imgsz"], 640)
        self.assertEqual(args["batch"], -1)
        self.assertEqual(args["seed"], 42)
        self.assertEqual(args["device"], 0)
        self.assertEqual(args["project"], str(Path("output") / "runs"))
        self.assertEqual(args["name"], "original_grouped_v1_yolov8n")
        self.assertNotIn("test", args)

    def test_enhanced_arguments_match_trial044_except_declared_gpu_adaptation(self):
        from tools.run_local_vscode_comparison import enhanced_train_args

        config = json.loads(
            (
                ROOT
                / "configs"
                / "trial044_gpu_adaptation"
                / "historical_cpu_trial044.json"
            ).read_text(encoding="utf-8")
        )
        args = enhanced_train_args(config, Path("authority.yaml"), Path("output"))

        self.assertEqual(args["epochs"], 60)
        self.assertEqual(args["imgsz"], 1024)
        self.assertEqual(args["batch"], 3)
        self.assertEqual(args["seed"], 22002)
        self.assertEqual(args["box"], 9.0)
        self.assertEqual(args["cls"], 1.4)
        self.assertEqual(args["dfl"], 1.8)
        self.assertEqual(args["device"], 0)
        self.assertEqual(args["name"], "enhanced_trial044_gpu_adaptation")
        self.assertNotIn("test", args)

    def test_output_directory_must_be_new_or_empty(self):
        from tools.run_local_vscode_comparison import require_new_output_directory

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty"
            empty.mkdir()
            require_new_output_directory(empty)

            occupied = root / "occupied"
            occupied.mkdir()
            (occupied / "preserve.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                require_new_output_directory(occupied)

    def test_powershell_launcher_uses_package_module_import_context(self):
        launcher = (ROOT / "tools" / "run_local_vscode_comparison.ps1").read_text(
            encoding="utf-8"
        )

        self.assertIn('[switch]$ResumeEnhancedOnly', launcher)
        self.assertIn('-m "tools.run_local_vscode_comparison"', launcher)
        self.assertNotIn(
            '& $pythonExe -u "tools\\run_local_vscode_comparison.py"', launcher
        )

if __name__ == "__main__":
    unittest.main()

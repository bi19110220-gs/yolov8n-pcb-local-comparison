import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from dashboard_launcher import launch_if_direct
from inspector_core import (
    AW09_SHA256, INFERENCE_SETTINGS, CheckpointVerificationError,
    load_image_bytes, predict_image, verify_aw09_checkpoint,
)

PACKAGE = Path(__file__).resolve().parents[2]


def test_aw09_identity_and_reject_missing_or_changed_checkpoint(tmp_path):
    assert verify_aw09_checkpoint(PACKAGE).name == "aw09_best.pt"
    with pytest.raises(CheckpointVerificationError, match="missing"):
        verify_aw09_checkpoint(tmp_path)
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights/aw09_best.pt").write_bytes(b"changed")
    with pytest.raises(CheckpointVerificationError, match="hash verification failed"):
        verify_aw09_checkpoint(tmp_path)


def test_rgb_upload_reaches_yolo_as_pil_with_preserved_colours():
    output = io.BytesIO()
    Image.new("RGB", (8, 8), (240, 20, 10)).save(output, format="PNG")
    image = load_image_bytes(output.getvalue())
    calls = []
    model = SimpleNamespace(predict=lambda **kwargs: calls.append(kwargs) or ["result"])
    assert predict_image(model, image, .35, "cpu") == ["result"]
    assert calls[0]["source"] is image
    assert image.getpixel((0, 0)) == (240, 20, 10)
    assert calls[0] == dict(source=image, conf=.35, device="cpu", verbose=False, **INFERENCE_SETTINGS)


def test_metadata_retains_recorded_metrics_and_model_identity():
    metadata = json.loads((PACKAGE / "weights/aw09_metadata.json").read_text())
    assert metadata["checkpoint_sha256"] == AW09_SHA256
    assert metadata["best_epoch"] == 38
    assert metadata["lr0"] == .01 and metadata["weight_decay"] == .001
    assert metadata["validation"]["map50_95"] == .5389454885300056
    assert metadata["test"]["map50_95"] == .4673596481511472
    assert metadata["test"]["per_class"][3]["recall"] == .8370989244746896


def test_direct_python_launches_same_interpreter_locally(monkeypatch, tmp_path):
    import dashboard_launcher
    import streamlit.runtime.scriptrunner
    monkeypatch.setattr(streamlit.runtime.scriptrunner, "get_script_run_ctx", lambda **_: None)
    monkeypatch.setattr(dashboard_launcher.sys, "argv", ["app.py", "--server.port=8502"])
    calls = []
    monkeypatch.setattr(dashboard_launcher.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(returncode=7))
    script = tmp_path / "app.py"
    with pytest.raises(SystemExit) as error:
        launch_if_direct(str(script))
    assert error.value.code == 7
    command, kwargs = calls[0]
    assert command[:5] == [dashboard_launcher.sys.executable, "-m", "streamlit", "run", str(script)]
    assert "--server.address=127.0.0.1" in command
    assert "--server.headless=false" in command
    assert command[-1] == "--server.port=8502"
    assert kwargs["cwd"] == tmp_path


def test_streamlit_execution_does_not_launch_recursively(monkeypatch):
    import dashboard_launcher
    import streamlit.runtime.scriptrunner
    monkeypatch.setattr(streamlit.runtime.scriptrunner, "get_script_run_ctx", lambda **_: object())
    def forbidden(*_, **__):
        pytest.fail("Recursive Streamlit launch")
    monkeypatch.setattr(dashboard_launcher.subprocess, "run", forbidden)
    launch_if_direct("app.py")

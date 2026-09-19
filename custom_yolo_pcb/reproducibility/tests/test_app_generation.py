import hashlib
from pathlib import Path

import pytest

from app_template import APP_SOURCE, GENERATED_MARKER
from generate_app import GeneratedAppConflict, write_generated_app


PACKAGE = Path(__file__).resolve().parents[2]


def test_committed_app_matches_canonical_generator(tmp_path):
    output = tmp_path / "app.py"
    result = write_generated_app(output)
    committed = PACKAGE / "app.py"
    assert output.read_bytes() == committed.read_bytes()
    assert result.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert result.replaced is True


def test_generator_is_idempotent_and_refuses_handwritten_content(tmp_path):
    output = tmp_path / "app.py"
    first = write_generated_app(output)
    second = write_generated_app(output)
    assert first.replaced is True
    assert second.replaced is False
    output.write_text("print('handwritten')\n", encoding="utf-8")
    with pytest.raises(GeneratedAppConflict, match="Refusing to replace"):
        write_generated_app(output)


def test_app_source_contains_fixed_model_and_explicit_interaction_contract():
    assert APP_SOURCE.startswith(GENERATED_MARKER)
    for text in (
        "PCB Quality Inspector — Enhanced YOLOv8n",
        "@st.cache_resource",
        '"Inspect PCB"',
        'st.tabs(["Detection result", "Original image"])',
        "**INFERENCE_SETTINGS",
        "min_value=0.05",
        "max_value=0.95",
        "value=0.25",
        "step=0.05",
        "No defects detected",
        "Defects detected — review required",
        "Inspection could not be completed",
        "download_button",
    ):
        assert text in APP_SOURCE
    assert "file_uploader" in APP_SOURCE
    assert "model uploader" not in APP_SOURCE.lower()
    assert "save(upload" not in APP_SOURCE.lower()


def test_app_import_compiles_without_starting_streamlit():
    compile((PACKAGE / "app.py").read_text(encoding="utf-8"), "app.py", "exec")


def test_clinical_light_theme_covers_sidebar_and_uploader():
    config = (PACKAGE / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert 'base = "light"' in config
    assert 'primaryColor = "#087f5b"' in config
    assert 'backgroundColor = "#f4f7fa"' in config
    assert '[data-testid="stSidebar"]' in APP_SOURCE
    assert '[data-testid="stFileUploaderDropzone"]' in APP_SOURCE


def test_app_uses_current_streamlit_width_api():
    assert "use_container_width" not in APP_SOURCE
    assert 'width="stretch"' in APP_SOURCE


def test_app_reuses_model_class_colours_for_badges_and_counts():
    assert "CLASS_COLOURS[name]" in APP_SOURCE
    assert "CLASS_COLOURS[top_defect]" in APP_SOURCE
    assert 'class="class-count"' in APP_SOURCE
    assert "class_count_cards(summary)" in APP_SOURCE

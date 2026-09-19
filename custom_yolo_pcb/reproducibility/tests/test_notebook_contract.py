import ast
from pathlib import Path

import nbformat


PACKAGE = Path(__file__).resolve().parents[2]
NOTEBOOK = PACKAGE / "PCB_Quality_Inspector.ipynb"


def notebook_source():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return notebook, "\n".join(cell.source for cell in notebook.cells)


def test_notebook_is_valid_and_uses_safe_defaults():
    notebook, source = notebook_source()
    assert notebook.nbformat == 4
    assert "RUN_TRAINING = False" in source
    assert "RUN_RECOVERY = False" in source
    assert "LAUNCH_STREAMLIT = True" in source
    assert "RECOVERY_OUTPUT_DIR = None" in source
    assert "PCB_NOTEBOOK_VERIFY_ONLY" in source
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)


def test_notebook_order_is_beginner_safe_and_deterministic():
    notebook, _ = notebook_source()
    expected_ids = [
        "kernel-check",
        "environment-summary",
        "dataset-check",
        "publication-check",
        "metric-table",
        "optional-training",
        "optional-recovery",
        "generate-app",
        "launch-streamlit",
        "stop-streamlit",
    ]
    cell_ids = [cell.id for cell in notebook.cells]
    positions = [cell_ids.index(cell_id) for cell_id in expected_ids]
    assert positions == sorted(positions)


def test_notebook_contains_no_machine_specific_paths_or_test_split_access():
    _, source = notebook_source()
    windows_user_root = chr(67) + ":" + "\\" + "Users" + "\\"
    slash_user_root = chr(67) + ":/" + "Users" + "/"
    assert windows_user_root not in source
    assert slash_user_root not in source
    assert "split='test'" not in source
    assert 'split="test"' not in source
    assert "trial044_best.pt" in source
    assert "http://localhost:8501" in source


def test_notebook_keeps_outputs_bounded():
    notebook, _ = notebook_source()
    assert len(notebook.cells) <= 24
    total_text_output = 0
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            total_text_output += len(text)
    assert total_text_output < 25_000


def test_notebook_charts_use_accessible_relative_markdown_images():
    _, source = notebook_source()
    assert "NotebookImage" not in source
    assert 'display(Markdown(f"![{title}]({relative_chart})"))' in source

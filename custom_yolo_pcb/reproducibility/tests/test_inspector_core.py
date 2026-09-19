import csv
import io
from pathlib import Path

import pytest
from PIL import Image

from inspector_core import (
    EXPECTED_CLASSES,
    INFERENCE_SETTINGS,
    CheckpointVerificationError,
    UploadValidationError,
    assert_class_mapping,
    detections_to_csv,
    load_image_bytes,
    normalise_detections,
    render_annotated_png,
    summarise_detections,
    validate_pixel_count,
    verify_trial044_checkpoint,
)


PACKAGE = Path(__file__).resolve().parents[2]


class Values:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class Boxes:
    xyxy = Values([[10.0, 20.0, 30.0, 50.0], [1.0, 2.0, 8.0, 12.0]])
    conf = Values([0.61, 0.93])
    cls = Values([3.0, 0.0])


class Result:
    boxes = Boxes()


def image_bytes(*, size=(12, 8), mode="RGB", image_format="PNG"):
    image = Image.new(mode, size, 128)
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def test_load_image_bytes_converts_to_rgb():
    image = load_image_bytes(image_bytes(mode="L"))
    assert image.mode == "RGB"
    assert image.size == (12, 8)


def test_load_image_bytes_applies_exif_orientation():
    image = Image.new("RGB", (2, 3), "white")
    exif = Image.Exif()
    exif[274] = 6
    output = io.BytesIO()
    image.save(output, format="JPEG", exif=exif)
    assert load_image_bytes(output.getvalue()).size == (3, 2)


def test_load_image_bytes_rejects_encoded_and_decoded_limits():
    with pytest.raises(UploadValidationError, match="10 MB"):
        load_image_bytes(b"x" * (10 * 1024 * 1024 + 1))
    with pytest.raises(UploadValidationError, match="20 megapixels"):
        validate_pixel_count(5000, 5000)


def test_load_image_bytes_rejects_corrupt_content():
    with pytest.raises(UploadValidationError, match="readable JPG or PNG"):
        load_image_bytes(b"not an image")


def test_load_image_bytes_converts_pillow_decompression_bomb_error(monkeypatch):
    def reject_oversized_image(*_args, **_kwargs):
        raise Image.DecompressionBombError("decoded image is unsafe")

    monkeypatch.setattr(Image, "open", reject_oversized_image)
    with pytest.raises(UploadValidationError, match="20 megapixels"):
        load_image_bytes(b"encoded image")


def test_class_mapping_and_fixed_inference_contract():
    assert EXPECTED_CLASSES == {
        0: "missing_hole",
        1: "mouse_bite",
        2: "open_circuit",
        3: "short",
        4: "spur",
        5: "spurious_copper",
    }
    assert_class_mapping(EXPECTED_CLASSES)
    with pytest.raises(CheckpointVerificationError, match="six-class"):
        assert_class_mapping({0: "short"})
    assert INFERENCE_SETTINGS == {
        "imgsz": 1024,
        "iou": 0.70,
        "max_det": 300,
        "augment": False,
    }


def test_trial044_checkpoint_is_fixed_and_hash_verified(tmp_path):
    assert verify_trial044_checkpoint(PACKAGE) == PACKAGE / "weights" / "trial044_best.pt"
    package = tmp_path / "custom_yolo_pcb"
    (package / "weights").mkdir(parents=True)
    (package / "weights" / "trial044_best.pt").write_bytes(b"tampered")
    with pytest.raises(CheckpointVerificationError, match="hash verification failed"):
        verify_trial044_checkpoint(package)


def test_normalised_rows_are_sorted_and_summarised():
    rows = normalise_detections(Result(), EXPECTED_CLASSES)
    assert rows == [
        {
            "class_id": 0,
            "class_name": "missing_hole",
            "confidence": 0.93,
            "x1": 1.0,
            "y1": 2.0,
            "x2": 8.0,
            "y2": 12.0,
            "width": 7.0,
            "height": 10.0,
        },
        {
            "class_id": 3,
            "class_name": "short",
            "confidence": 0.61,
            "x1": 10.0,
            "y1": 20.0,
            "x2": 30.0,
            "y2": 50.0,
            "width": 20.0,
            "height": 30.0,
        },
    ]
    summary = summarise_detections(rows)
    assert summary == {
        "total_detections": 2,
        "classes_found": 2,
        "mean_confidence": pytest.approx(0.77),
        "top_defect": "missing_hole",
        "per_class_counts": {"missing_hole": 1, "short": 1},
    }


def test_downloads_are_in_memory_png_and_csv():
    rows = normalise_detections(Result(), EXPECTED_CLASSES)
    png = render_annotated_png(Image.new("RGB", (64, 64), "white"), rows)
    with Image.open(io.BytesIO(png)) as image:
        assert image.format == "PNG"
        assert image.size == (64, 64)
    parsed = list(csv.DictReader(io.StringIO(detections_to_csv(rows).decode("utf-8"))))
    assert [row["class_name"] for row in parsed] == ["missing_hole", "short"]


def test_empty_summary_is_safe():
    assert summarise_detections([]) == {
        "total_detections": 0,
        "classes_found": 0,
        "mean_confidence": 0.0,
        "top_defect": None,
        "per_class_counts": {},
    }

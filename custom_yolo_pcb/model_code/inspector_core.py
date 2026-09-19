"""Pure, testable helpers for the in-memory Trial 044 PCB inspector."""

from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
TRIAL044_SHA256 = "4bdde7e5dc60258de06fceb66ed25fd5814eafa3d6108a5dfb0244d59de0274d"

EXPECTED_CLASSES = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper",
}

INFERENCE_SETTINGS = {
    "imgsz": 1024,
    "iou": 0.70,
    "max_det": 300,
    "augment": False,
}

CLASS_COLOURS = {
    "missing_hole": "#0057B8",
    "mouse_bite": "#8A3FFC",
    "open_circuit": "#D1495B",
    "short": "#B45309",
    "spur": "#007A78",
    "spurious_copper": "#52606D",
}

CSV_FIELDS = (
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
)


class UploadValidationError(ValueError):
    """Raised when an uploaded image cannot be processed safely."""


class CheckpointVerificationError(PermissionError):
    """Raised when the fixed Trial 044 checkpoint contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pixel_count(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise UploadValidationError("The uploaded image has invalid dimensions.")
    if width * height > MAX_PIXELS:
        raise UploadValidationError("Image exceeds the 20 megapixels decoded limit.")


def load_image_bytes(data: bytes) -> Image.Image:
    """Decode a JPG or PNG safely without creating an upload file on disk."""
    if not data:
        raise UploadValidationError("Choose a non-empty JPG or PNG image.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadValidationError("Image exceeds the 10 MB upload limit.")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image_format = (opened.format or "").upper()
            if image_format not in {"JPEG", "PNG"}:
                raise UploadValidationError("Only JPG, JPEG, and PNG images are supported.")
            opened.verify()
        with Image.open(io.BytesIO(data)) as opened:
            validate_pixel_count(*opened.size)
            transposed = ImageOps.exif_transpose(opened)
            validate_pixel_count(*transposed.size)
            return transposed.convert("RGB")
    except UploadValidationError:
        raise
    except Image.DecompressionBombError as error:
        raise UploadValidationError(
            "Image exceeds the 20 megapixels decoded limit."
        ) from error
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise UploadValidationError(
            "The uploaded file is not a readable JPG or PNG image."
        ) from error


def assert_class_mapping(names: Mapping[int, str] | Sequence[str]) -> None:
    observed = (
        {int(index): str(name) for index, name in names.items()}
        if isinstance(names, Mapping)
        else {index: str(name) for index, name in enumerate(names)}
    )
    if observed != EXPECTED_CLASSES:
        raise CheckpointVerificationError(
            "The model does not expose the required six-class PCB mapping."
        )


def verify_trial044_checkpoint(package: Path) -> Path:
    package = Path(package).resolve()
    checkpoint = package / "weights" / "trial044_best.pt"
    if not checkpoint.is_file():
        raise CheckpointVerificationError(
            "The fixed Trial 044 checkpoint is missing: weights/trial044_best.pt"
        )
    if sha256_file(checkpoint) != TRIAL044_SHA256:
        raise CheckpointVerificationError(
            "Trial 044 checkpoint hash verification failed. Restore the published file."
        )
    return checkpoint


def normalise_detections(result, names: Mapping[int, str] | Sequence[str]) -> list[dict]:
    assert_class_mapping(names)
    if result.boxes is None:
        return []
    coordinates = result.boxes.xyxy.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    class_ids = result.boxes.cls.cpu().tolist()
    rows = []
    for coordinates_row, confidence, class_id_value in zip(
        coordinates, confidences, class_ids, strict=True
    ):
        class_id = int(class_id_value)
        if class_id not in EXPECTED_CLASSES:
            raise CheckpointVerificationError(
                f"The model returned unexpected class id {class_id}."
            )
        x1, y1, x2, y2 = (float(value) for value in coordinates_row)
        rows.append(
            {
                "class_id": class_id,
                "class_name": EXPECTED_CLASSES[class_id],
                "confidence": float(confidence),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": max(0.0, x2 - x1),
                "height": max(0.0, y2 - y1),
            }
        )
    return sorted(rows, key=lambda row: float(row["confidence"]), reverse=True)


def summarise_detections(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    confidences = [float(row["confidence"]) for row in rows]
    counts = Counter(str(row["class_name"]) for row in rows)
    top_defect = counts.most_common(1)[0][0] if counts else None
    return {
        "total_detections": len(rows),
        "classes_found": len(counts),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "top_defect": top_defect,
        "per_class_counts": dict(sorted(counts.items())),
    }


def detections_to_csv(rows: Sequence[Mapping[str, object]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in CSV_FIELDS})
    return output.getvalue().encode("utf-8")


def _label_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 16)
    except OSError:
        return ImageFont.load_default()


def render_annotated_png(
    image: Image.Image, rows: Sequence[Mapping[str, object]]
) -> bytes:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = _label_font()
    stroke_width = max(2, round(min(annotated.size) / 300))
    for row in rows:
        name = str(row["class_name"])
        colour = CLASS_COLOURS[name]
        x1, y1 = float(row["x1"]), float(row["y1"])
        x2, y2 = float(row["x2"]), float(row["y2"])
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=stroke_width)
        label = f"{name} {float(row['confidence']):.2f}"
        left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
        label_height = bottom - top + 8
        label_y = max(0, y1 - label_height)
        draw.rectangle((x1, label_y, x1 + right - left + 10, y1), fill=colour)
        draw.text((x1 + 5, label_y + 3), label, fill="white", font=font)
    output = io.BytesIO()
    annotated.save(output, format="PNG", optimize=True)
    return output.getvalue()

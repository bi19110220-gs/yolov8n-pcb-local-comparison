"""Temporary MPDIoU training boundary for the validation-only top-five campaign.

The loss exists only while a trial is training.  Checkpoints deliberately retain
the ordinary YOLOv8n deployment graph so they can be evaluated and deployed
without this module.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

from pcb_mpdiou_loss import MPDIoUBboxLoss, install_mpdiou_criterion, mpdiou_criterion_boundary


def _has_mpdiou_criterion(model: object) -> bool:
    """Return whether a model currently owns the temporary MPDIoU criterion."""
    criterion = getattr(model, "criterion", None)
    return isinstance(getattr(criterion, "bbox_loss", None), MPDIoUBboxLoss)


@contextmanager
def mpdiou_serialization_boundary(*models: object) -> Iterator[None]:
    """Hide each installed temporary criterion while a checkpoint is serialized."""
    selected = [model for model in models if model is not None and _has_mpdiou_criterion(model)]
    if not selected:
        raise ValueError("MPDIoU serialization boundary requires an installed temporary criterion.")
    with mpdiou_criterion_boundary(*selected):
        yield


def _state_dict_keys(value: object) -> Iterator[str]:
    """Yield serialized tensor/state keys from an Ultralytics checkpoint section."""
    if isinstance(value, Mapping):
        yield from (str(key) for key in value)
        return
    state_dict = getattr(value, "state_dict", None)
    if callable(state_dict):
        yield from (str(key) for key in state_dict())


def _criterion_state_keys(payload: object) -> list[str]:
    """Find checkpoint state that would leak the temporary loss into deployment."""
    if not isinstance(payload, Mapping):
        return []
    found: list[str] = []
    for section in ("model", "ema"):
        for key in _state_dict_keys(payload.get(section)):
            if key.startswith("criterion.") or ".criterion." in key:
                found.append(f"{section}.{key}")
    return found


def assert_mpdiou_checkpoint_is_clean(checkpoint_path: Path, *, expected_parameters: int) -> None:
    """Reject a checkpoint with loss state, then prove it reloads as normal YOLOv8n."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"MPDIoU checkpoint is missing: {checkpoint_path}")
    payload: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    criterion_keys = _criterion_state_keys(payload)
    if criterion_keys:
        raise ValueError(
            "MPDIoU checkpoint contains temporary criterion state: " + ", ".join(sorted(criterion_keys))
        )
    model = YOLO(str(checkpoint_path)).model
    # Ultralytics checkpoints reload as an unfused graph.  The campaign's
    # recorded YOLOv8n parameter metric is the normal deployment (fused)
    # graph, so compare like with like instead of rejecting every clean
    # checkpoint for the BatchNorm parameters that fusion removes.
    model.fuse()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != expected_parameters:
        raise ValueError(
            f"MPDIoU checkpoint parameter count mismatch: expected {expected_parameters}, got {parameter_count}."
        )


class Top5MPDIoUTrainer(DetectionTrainer):
    """Detection trainer that swaps CIoU for MPDIoU only during this trial."""

    def _build_train_pipeline(self) -> None:
        model = unwrap_model(self.model)
        if not _has_mpdiou_criterion(model):
            install_mpdiou_criterion(model)
        super()._build_train_pipeline()

    def save_model(self) -> bool:
        live_model = unwrap_model(self.model)
        ema_model = unwrap_model(self.ema.ema) if getattr(self, "ema", None) is not None else None
        with mpdiou_serialization_boundary(live_model, ema_model):
            return super().save_model()

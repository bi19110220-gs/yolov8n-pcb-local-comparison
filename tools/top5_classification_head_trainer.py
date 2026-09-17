"""YOLOv8n trainer for a strictly classification-tower-only continuation.

The deployment graph remains the stock YOLOv8n Detect head. During training,
only ``model.22.cv3.*`` is allowed to receive gradients or optimizer state;
the regression tower, backbone, neck, and DFL projection stay frozen.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from ultralytics.utils.torch_utils import unwrap_model

from tools.top5_mpdiou_trainer import Top5MPDIoUTrainer


CLASSIFICATION_PARAMETER_PREFIX = "model.22.cv3."
EXPECTED_CLASSIFICATION_TENSORS = 24
EXPECTED_CLASSIFICATION_PARAMETERS = 370_578


def configure_classification_head_only(
    model: nn.Module,
    *,
    expected_tensors: int = EXPECTED_CLASSIFICATION_TENSORS,
    expected_parameters: int = EXPECTED_CLASSIFICATION_PARAMETERS,
) -> dict[str, Any]:
    """Set and attest the exact stock YOLOv8n classification parameter scope."""
    model = unwrap_model(model)
    named = list(model.named_parameters())
    selected = [
        (name, parameter)
        for name, parameter in named
        if name.startswith(CLASSIFICATION_PARAMETER_PREFIX)
    ]
    if len(selected) != expected_tensors:
        raise PermissionError(
            "YOLOv8n classification scope tensor count changed: "
            f"expected {expected_tensors}, got {len(selected)}"
        )
    parameter_count = sum(parameter.numel() for _, parameter in selected)
    if parameter_count != expected_parameters:
        raise PermissionError(
            "YOLOv8n classification scope parameter count changed: "
            f"expected {expected_parameters}, got {parameter_count}"
        )
    selected_names = tuple(name for name, _ in selected)
    selected_ids = {id(parameter) for _, parameter in selected}
    for _name, parameter in named:
        parameter.requires_grad_(id(parameter) in selected_ids)
    if tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ) != selected_names:
        raise PermissionError("YOLOv8n classification scope did not apply exactly")
    return {
        "schema": "top5_20_percent/classification_scope_v1",
        "parameter_prefix": CLASSIFICATION_PARAMETER_PREFIX,
        "trainable_names": list(selected_names),
        "trainable_tensors": len(selected),
        "trainable_parameters": parameter_count,
        "frozen_parameters": sum(
            parameter.numel() for name, parameter in named if name not in selected_names
        ),
        "optimizer_parameter_ids_bound": False,
    }


def _filter_optimizer_to_trainable(optimizer: torch.optim.Optimizer) -> set[int]:
    """Remove every frozen parameter from the already-constructed optimizer."""
    selected_ids: set[int] = set()
    retained_groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        retained = [parameter for parameter in group["params"] if parameter.requires_grad]
        selected_ids.update(id(parameter) for parameter in retained)
        if retained:
            group["params"] = retained
            retained_groups.append(group)
    optimizer.param_groups[:] = retained_groups
    if not selected_ids:
        raise PermissionError("classification-only optimizer has no trainable parameters")
    return selected_ids


class Top5ClassificationHeadTrainer(Top5MPDIoUTrainer):
    """MPDIoU trainer whose optimizer owns only the stock classification tower."""

    def build_optimizer(
        self,
        model,
        name="auto",
        lr=0.001,
        momentum=0.9,
        decay=1e-5,
        iterations=1e5,
    ):
        scope = configure_classification_head_only(model)
        optimizer = super().build_optimizer(
            model=model,
            name=name,
            lr=lr,
            momentum=momentum,
            decay=decay,
            iterations=iterations,
        )
        optimizer_ids = _filter_optimizer_to_trainable(optimizer)
        expected_ids = {
            id(parameter)
            for name, parameter in unwrap_model(model).named_parameters()
            if name.startswith(CLASSIFICATION_PARAMETER_PREFIX)
        }
        if optimizer_ids != expected_ids:
            raise PermissionError("classification-only optimizer parameter binding changed")
        scope["optimizer_parameter_ids_bound"] = True
        scope["optimizer_parameters"] = len(optimizer_ids)
        self.classification_scope_audit = scope
        return optimizer

    def _model_train(self):
        super()._model_train()
        # Keep frozen BatchNorm statistics unchanged during this continuation.
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                parameters = list(module.parameters(recurse=False))
                if parameters and all(not parameter.requires_grad for parameter in parameters):
                    module.eval()

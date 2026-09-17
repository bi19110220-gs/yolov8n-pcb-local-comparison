"""Original MPDIoU box loss with an exact Ultralytics criterion boundary."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.tal import bbox2dist


MPDIOU_EPSILON = 1e-9


def _validate_boxes(boxes: torch.Tensor, *, name: str) -> None:
    if (
        not isinstance(boxes, torch.Tensor)
        or not boxes.is_floating_point()
        or boxes.ndim != 2
        or int(boxes.shape[-1]) != 4
        or not bool(torch.isfinite(boxes).all())
    ):
        raise ValueError(f"{name} must be a finite floating Nx4 tensor.")
    if bool((boxes[:, 2:] <= boxes[:, :2]).any()):
        raise ValueError(f"{name} must contain positive-area ordered xyxy boxes.")


def _validate_image_size(image_size: torch.Tensor) -> None:
    if (
        not isinstance(image_size, torch.Tensor)
        or not image_size.is_floating_point()
        or image_size.numel() != 2
        or not bool(torch.isfinite(image_size).all())
        or bool((image_size <= 0).any())
    ):
        raise ValueError("MPDIoU image_size must be a finite positive floating two-tensor.")


class MPDIoULoss(nn.Module):
    """Aligned pixel-space MPDIoU loss from arXiv:2307.07662."""

    def __init__(self, *, epsilon: float = MPDIOU_EPSILON) -> None:
        super().__init__()
        if not float(epsilon) > 0.0:
            raise ValueError("MPDIoU epsilon must be positive.")
        self.epsilon = float(epsilon)
        self.last_graph_flags: dict[str, bool] | None = None

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        *,
        image_size: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int]]:
        _validate_boxes(pred_boxes, name="pred_boxes")
        target_boxes = target_boxes.detach()
        _validate_boxes(target_boxes, name="target_boxes")
        _validate_image_size(image_size)
        if pred_boxes.shape != target_boxes.shape:
            raise ValueError("MPDIoU boxes must have identical shapes.")
        image_size = image_size.detach().to(device=pred_boxes.device, dtype=pred_boxes.dtype)
        diagonal_squared = image_size.square().sum().clamp_min(self.epsilon)
        self.last_graph_flags = {
            "target_boxes_detached": not target_boxes.requires_grad,
            "image_diagonal_detached": not diagonal_squared.requires_grad,
        }
        if pred_boxes.shape[0] == 0:
            empty = pred_boxes[:, 0]
            diagnostics = {
                "anchors": 0,
                "mean_iou": 0.0,
                "mean_top_left_penalty": 0.0,
                "mean_bottom_right_penalty": 0.0,
                "mean_mpdiou_loss": 0.0,
            }
            return (empty, diagnostics) if return_diagnostics else empty

        pred_wh = pred_boxes[:, 2:] - pred_boxes[:, :2]
        target_wh = target_boxes[:, 2:] - target_boxes[:, :2]
        intersection_wh = (
            torch.minimum(pred_boxes[:, 2:], target_boxes[:, 2:])
            - torch.maximum(pred_boxes[:, :2], target_boxes[:, :2])
        ).clamp_min(0.0)
        intersection = intersection_wh.prod(dim=-1)
        union = pred_wh.prod(dim=-1) + target_wh.prod(dim=-1) - intersection
        iou = intersection / union.clamp_min(self.epsilon)
        top_left_penalty = (pred_boxes[:, :2] - target_boxes[:, :2]).square().sum(-1) / diagonal_squared
        bottom_right_penalty = (pred_boxes[:, 2:] - target_boxes[:, 2:]).square().sum(-1) / diagonal_squared
        loss = 1.0 - iou + top_left_penalty + bottom_right_penalty
        diagnostics = {
            "anchors": int(loss.numel()),
            "mean_iou": float(iou.detach().mean()),
            "mean_top_left_penalty": float(top_left_penalty.detach().mean()),
            "mean_bottom_right_penalty": float(bottom_right_penalty.detach().mean()),
            "mean_mpdiou_loss": float(loss.detach().mean()),
        }
        return (loss, diagnostics) if return_diagnostics else loss


class MPDIoUBboxLoss(BboxLoss):
    """Ultralytics BboxLoss replacing only CIoU with pixel-space MPDIoU."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__(reg_max=reg_max)
        self.mpdiou = MPDIoULoss()
        self.last_diagnostics: dict[str, float | int] | None = None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1).detach()
        pred_pixels = (pred_bboxes * stride)[fg_mask]
        target_pixels = (target_bboxes * stride)[fg_mask]
        per_anchor, diagnostics = self.mpdiou(
            pred_pixels,
            target_pixels,
            image_size=imgsz,
            return_diagnostics=True,
        )
        loss_iou = (per_anchor.unsqueeze(-1) * weight).sum() / target_scores_sum.detach()
        self.last_diagnostics = {
            **diagnostics,
            "positive_anchors": int(fg_mask.sum()),
            "weighted_mpdiou_loss": float(loss_iou.detach()),
        }

        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points,
                target_bboxes,
                self.dfl_loss.reg_max - 1,
            )
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask],
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            normalized_pred = pred_dist * stride
            normalized_pred[..., 0::2] /= imgsz[1]
            normalized_pred[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(
                    normalized_pred[fg_mask],
                    target_ltrb[fg_mask],
                    reduction="none",
                ).mean(-1, keepdim=True)
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_iou, loss_dfl


def install_mpdiou_criterion(model: nn.Module):
    """Install MPDIoU into one detection criterion without deployment state."""
    criterion = getattr(model, "criterion", None)
    if criterion is None:
        if not hasattr(model, "init_criterion"):
            raise ValueError("MPDIoU installation requires a detection model.")
        criterion = model.init_criterion()
        model.criterion = criterion
    source = getattr(criterion, "bbox_loss", None)
    if isinstance(source, MPDIoUBboxLoss):
        raise ValueError("MPDIoU criterion is already installed.")
    if type(source) is not BboxLoss:
        raise ValueError("MPDIoU installation requires the stock BboxLoss.")
    destination = MPDIoUBboxLoss(int(criterion.reg_max)).to(criterion.device)
    destination.train(source.training)
    criterion.bbox_loss = destination
    return criterion


@contextmanager
def mpdiou_criterion_boundary(*models: nn.Module):
    """Remove temporary MPDIoU criterion objects during serialization."""
    records: list[tuple[nn.Module, object]] = []
    seen: set[int] = set()
    try:
        for model in models:
            if model is None or id(model) in seen:
                continue
            seen.add(id(model))
            criterion = model.__dict__.get("criterion")
            if criterion is None or not isinstance(
                getattr(criterion, "bbox_loss", None), MPDIoUBboxLoss
            ):
                raise ValueError("MPDIoU serialization boundary requires its criterion.")
            records.append((model, criterion))
            delattr(model, "criterion")
        yield
    finally:
        for model, criterion in reversed(records):
            model.criterion = criterion

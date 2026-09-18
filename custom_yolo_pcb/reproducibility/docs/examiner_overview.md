# Examiner Overview

## Research question

This package compares a fresh stock YOLOv8n baseline trained with the recorded
grouped-v1 authority against a classification-head Trial 044 continuation that
uses the recorded OHEM training view and a Trial 035 parent checkpoint.

## What is controlled

- Both are six-class PCB defect detectors based on YOLOv8n.
- Each run uses its exact recorded hyperparameters and validation authority.
- Both publication records are validation-only.
- Reproduction uses one local GPU worker at a time on Windows.

## What is not controlled

The runs intentionally use different recorded training authorities, image
sizes, schedules, and starting checkpoints. They are not a controlled
same-data ablation. Historical Trial 044 used CPU; the new execution is an RTX
3080 adaptation and is not expected to be bit-identical.

## Evidence package

After terminal completion, the package includes exact configurations,
normalized logs, training curves, clean-validation metrics, checkpoint hashes,
and the final original-versus-enhanced comparison. Held-out test evaluation is
outside version 1.0.0.

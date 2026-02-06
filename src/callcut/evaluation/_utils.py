"""Utility functions for evaluation metrics."""

from __future__ import annotations


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 from confusion matrix counts.

    Returns (precision, recall, f1). Returns 0.0 for undefined metrics
    (e.g., precision when tp + fp == 0).
    """
    eps = 1e-12
    precision = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall + eps)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1

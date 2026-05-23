"""Helpers to convert continuous attributes into Boolean literals.

The dual GraphTM consumes node-symbol literals via the
GraphTsetlinMachine library; this module is a small helper for the
small number of numeric attributes (entity-degree, neighbour-count,
relation-frequency) the model could optionally consume.

Kept minimal because the bulk of the binarisation work is symbolic
(entity labels and polarity flags) and handled directly in
:mod:`dual_graph`.
"""
from __future__ import annotations

import numpy as np


def thermometer_encode(x: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Thermometer-encode a scalar array against ascending thresholds.

    For a value v and thresholds (t0 < t1 < ... < t_{k-1}), the output is
    a k-bit Boolean vector ``[v >= t0, v >= t1, ..., v >= t_{k-1}]``.
    """
    if thresholds.ndim != 1:
        raise ValueError("thresholds must be 1-D and ascending")
    if not np.all(np.diff(thresholds) > 0):
        raise ValueError("thresholds must be strictly ascending")
    return (x[..., None] >= thresholds[None, :]).astype(np.uint8)


def percentile_thresholds(x: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Compute ascending percentile thresholds for thermometer encoding."""
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    th = np.unique(np.percentile(x, qs))
    if len(th) < 1:
        return np.array([np.median(x)])
    return th

"""propose: observation metadata extraction -- shape, dtype, range, layout, contiguity.
input: arbitrary observation tensors or arrays.
output: dict of metadata properties."""

from __future__ import annotations

from typing import Any


def shape_of(value: Any) -> tuple[int, ...] | str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return "unknown"
    return tuple(int(x) for x in shape)


def dtype_of(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    return str(dtype) if dtype is not None else type(value).__name__


def min_max(value: Any) -> tuple[float | str, float | str]:
    try:
        return float(value.min()), float(value.max())
    except Exception:
        return "unknown", "unknown"


def infer_layout(shape: tuple[int, ...] | str) -> str:
    if not isinstance(shape, tuple):
        return "unknown"
    if len(shape) == 3:
        if shape[-1] in (1, 3, 4):
            return "HWC"
        if shape[0] in (1, 3, 4):
            return "CHW"
    if len(shape) == 4:
        if shape[1] in (1, 3, 4):
            return "NCHW"
        if shape[-1] in (1, 3, 4):
            return "NHWC"
    return "unknown"


def is_contiguous(value: Any) -> str:
    flags = getattr(value, "flags", None)
    if flags is not None:
        try:
            return str(bool(flags["C_CONTIGUOUS"]))
        except Exception:
            return "unknown"
    try:
        return str(bool(value.is_contiguous()))
    except Exception:
        return "unknown"


def observation_metadata(value: Any) -> dict[str, Any]:
    shape = shape_of(value)
    low, high = min_max(value)
    return {
        "shape": shape,
        "dtype": dtype_of(value),
        "range": [low, high],
        "layout": infer_layout(shape),
        "contiguous": is_contiguous(value),
    }

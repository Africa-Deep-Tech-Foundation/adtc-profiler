"""Read parameter count, context length, and architecture from a GGUF binary header.

Parses only the key-value section of the header — does not load weights.
Returns an empty dict on any parse failure (non-GGUF file, unsupported version).
"""
from __future__ import annotations

import struct
from pathlib import Path

_MAGIC = b"GGUF"

# GGUF value type IDs
_VT_UINT8, _VT_INT8 = 0, 1
_VT_UINT16, _VT_INT16 = 2, 3
_VT_UINT32, _VT_INT32 = 4, 5
_VT_FLOAT32, _VT_BOOL = 6, 7
_VT_STRING = 8
_VT_ARRAY = 9
_VT_UINT64, _VT_INT64, _VT_FLOAT64 = 10, 11, 12

_SCALAR_SIZES = {
    _VT_UINT8: 1, _VT_INT8: 1,
    _VT_UINT16: 2, _VT_INT16: 2,
    _VT_UINT32: 4, _VT_INT32: 4,
    _VT_FLOAT32: 4, _VT_BOOL: 1,
    _VT_UINT64: 8, _VT_INT64: 8, _VT_FLOAT64: 8,
}


def _u32(f) -> int:
    return struct.unpack("<I", f.read(4))[0]


def _u64(f) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def _read_string(f) -> str:
    return f.read(_u64(f)).decode("utf-8", errors="replace")


def _read_value(f, vtype: int):
    """Read one GGUF value and return it (or None for floats/unneeded types)."""
    if vtype in _SCALAR_SIZES:
        raw = f.read(_SCALAR_SIZES[vtype])
        if vtype == _VT_UINT32:
            return struct.unpack("<I", raw)[0]
        if vtype == _VT_INT32:
            return struct.unpack("<i", raw)[0]
        if vtype == _VT_UINT64:
            return struct.unpack("<Q", raw)[0]
        if vtype == _VT_INT64:
            return struct.unpack("<q", raw)[0]
        if vtype == _VT_BOOL:
            return bool(raw[0])
        return None  # float or small int we don't need
    if vtype == _VT_STRING:
        return _read_string(f)
    if vtype == _VT_ARRAY:
        elem_type = _u32(f)
        count = _u64(f)
        if elem_type in _SCALAR_SIZES:
            # Seek past scalar arrays in one hop — element-wise reads on an
            # adversarial count would loop for a very long time.
            f.seek(count * _SCALAR_SIZES[elem_type], 1)
            return None
        for _ in range(count):
            _read_value(f, elem_type)
        return None
    return None  # unknown — will stall; callers should protect with try/except


_MAX_TENSORS = 1 << 16
_MAX_DIMS = 8
_MAX_DIM = 1 << 40


def _sum_tensor_params(f, n_tensors: int) -> int | None:
    """Sum element counts over the tensor-info section (follows the KV section).

    This is the actual parameter count of the model — unlike the optional
    `general.parameter_count` KV, the tensor table is always present.
    """
    if not 0 < n_tensors <= _MAX_TENSORS:
        return None
    total = 0
    for _ in range(n_tensors):
        _read_string(f)                    # tensor name
        n_dims = _u32(f)
        if not 0 < n_dims <= _MAX_DIMS:
            return None
        count = 1
        for _ in range(n_dims):
            dim = _u64(f)
            if not 0 < dim <= _MAX_DIM:
                return None
            count *= dim
        _u32(f)                            # ggml type
        _u64(f)                            # data offset
        total += count
    return total


def extract_metadata(model_path: Path) -> dict:
    """Return GGUF header metadata relevant to fraud detection and run display.

    Keys returned (when available):
      params_count    – actual parameter count from the file header
      context_length  – maximum context window (n_ctx_train)
      architecture    – model architecture string (e.g. "llama", "qwen2")
    """
    try:
        with open(model_path, "rb") as f:
            if f.read(4) != _MAGIC:
                return {}
            version = _u32(f)
            # v1 uses 32-bit counts/lengths — this parser reads the v2/v3
            # 64-bit layout, so accepting v1 would yield misaligned garbage.
            if version not in (2, 3):
                return {}
            n_tensors = _u64(f)
            n_kv = _u64(f)

            result: dict = {}
            for _ in range(n_kv):
                key = _read_string(f)
                vtype = _u32(f)
                value = _read_value(f, vtype)

                if key == "general.parameter_count":
                    result["params_count"] = value
                elif key == "general.architecture":
                    result["architecture"] = value
                elif key.endswith(".context_length"):
                    result["context_length"] = value

            if result.get("params_count") is None:
                # `general.parameter_count` is optional and usually absent
                # (the reference SmolLM2 fixture lacks it). Fall back to
                # summing the tensor table so the fraud check has real data.
                params = _sum_tensor_params(f, n_tensors)
                if params is not None:
                    result["params_count"] = params

        return result
    except Exception:
        return {}


def parse_parameter_estimate(estimate: str) -> int | None:
    """Parse a human-readable parameter estimate ('135M', '1.1B', '7B') to an integer."""
    s = estimate.strip().upper()
    for suffix, mult in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if s.endswith(suffix):
            try:
                return int(float(s[:-1]) * mult)
            except ValueError:
                return None
    try:
        return int(s)
    except ValueError:
        return None


def fraud_check(claimed_estimate: str, actual_params: int | None) -> bool | None:
    """Two-sided check: measured params within ±15% of the claimed estimate.

    Returns None when the check cannot be performed (no measured count, or an
    unparseable claim) — an unknown must not masquerade as a passed check.
    ±15% absorbs labeling conventions ("7B" models range roughly 6.7–7.3B);
    quantization does not change the parameter count.
    """
    if actual_params is None:
        return None
    claimed = parse_parameter_estimate(claimed_estimate)
    if claimed is None or claimed <= 0:
        return None
    return claimed * 0.85 <= actual_params <= claimed * 1.15

"""Unit tests for the GGUF header parser and the fraud check."""
import struct

from adtc_profiler import gguf


def _string(s: bytes) -> bytes:
    return struct.pack("<Q", len(s)) + s


def _kv_u64(key: bytes, value: int) -> bytes:
    return _string(key) + struct.pack("<I", 10) + struct.pack("<Q", value)


def _kv_str(key: bytes, value: bytes) -> bytes:
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _tensor(name: bytes, dims: list[int]) -> bytes:
    out = _string(name) + struct.pack("<I", len(dims))
    for d in dims:
        out += struct.pack("<Q", d)
    out += struct.pack("<I", 0)   # ggml type
    out += struct.pack("<Q", 0)   # offset
    return out


def _gguf(version: int, tensors: list[bytes], kvs: list[bytes]) -> bytes:
    header = b"GGUF" + struct.pack("<I", version)
    header += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kvs))
    return header + b"".join(kvs) + b"".join(tensors)


def test_v1_files_are_rejected(tmp_path):
    # v1 uses 32-bit counts; parsing it with the v2/v3 layout yields garbage,
    # so the parser must refuse rather than mis-read.
    f = tmp_path / "m.gguf"
    f.write_bytes(_gguf(1, [], [_kv_str(b"general.architecture", b"llama")]))
    assert gguf.extract_metadata(f) == {}


def test_params_count_from_explicit_kv(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(_gguf(3, [], [_kv_u64(b"general.parameter_count", 135_000_000)]))
    assert gguf.extract_metadata(f)["params_count"] == 135_000_000


def test_params_count_falls_back_to_tensor_table(tmp_path):
    # Most real GGUFs (including the reference SmolLM2 fixture) omit
    # general.parameter_count — the tensor table is the ground truth.
    f = tmp_path / "m.gguf"
    tensors = [_tensor(b"tok_embd.weight", [576, 49152]), _tensor(b"out.bias", [576])]
    f.write_bytes(_gguf(3, tensors, [_kv_str(b"general.architecture", b"llama")]))
    meta = gguf.extract_metadata(f)
    assert meta["params_count"] == 576 * 49152 + 576


def test_fraud_check_two_sided():
    assert gguf.fraud_check("135M", 135_000_000) is True
    assert gguf.fraud_check("135M", 1_500_000_000) is False  # ships far bigger
    assert gguf.fraud_check("7B", 135_000_000) is False      # ships far smaller
    assert gguf.fraud_check("7B", 6_900_000_000) is True     # labeling slack


def test_fraud_check_unknown_is_none_not_true():
    # An unperformable check must not report success.
    assert gguf.fraud_check("135M", None) is None
    assert gguf.fraud_check("garbage", 135_000_000) is None

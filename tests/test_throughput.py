"""Unit tests for throughput measurement — the 30%-weight leaderboard number."""
import pytest

from adtc_profiler import throughput


def _rows(pp_ts=681.4, tg_ts=390.2, n_prompt=512, n_gen=128):
    return [
        {"n_prompt": n_prompt, "n_gen": 0, "avg_ts": pp_ts},
        {"n_prompt": 0, "n_gen": n_gen, "avg_ts": tg_ts},
    ]


def test_measure_projects_rows(monkeypatch):
    monkeypatch.setattr(throughput, "run_llama_bench", lambda *a, **k: _rows())
    out = throughput.measure(__import__("pathlib").Path("m.gguf"))
    assert out["tokens_per_second_generation"] == 390.2
    # TTFT = time to ingest the whole 512-token prompt at pp rate
    assert out["first_token_latency_ms"] == pytest.approx(512 / 681.4 * 1000, abs=0.01)
    assert out["prompt_tokens"] == 512
    assert out["generated_tokens"] == 128


def test_measure_requires_generation_row(monkeypatch):
    monkeypatch.setattr(
        throughput, "run_llama_bench",
        lambda *a, **k: [{"n_prompt": 512, "n_gen": 0, "avg_ts": 100.0}],
    )
    with pytest.raises(throughput.LlamaBenchError):
        throughput.measure(__import__("pathlib").Path("m.gguf"))


def test_measure_rejects_nonpositive_rate(monkeypatch):
    monkeypatch.setattr(
        throughput, "run_llama_bench",
        lambda *a, **k: [{"n_prompt": 0, "n_gen": 128, "avg_ts": 0.0}],
    )
    with pytest.raises(throughput.LlamaBenchError):
        throughput.measure(__import__("pathlib").Path("m.gguf"))


def test_llama_bench_command_pins_cpu(monkeypatch):
    """GPU offload must be pinned off — participant laptops with Metal/CUDA
    would otherwise record GPU numbers the CPU-only audit can never match."""
    captured = {}

    class _Proc:
        returncode = 0
        stdout = "[]"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(throughput, "_find_llama_bench", lambda: "llama-bench")
    monkeypatch.setattr(throughput.subprocess, "run", fake_run)
    throughput.run_llama_bench(__import__("pathlib").Path("m.gguf"))
    cmd = captured["cmd"]
    assert "-ngl" in cmd
    assert cmd[cmd.index("-ngl") + 1] == "0"


# Regression coverage: measure() used to never pass a thread count at all, leaving llama-bench
# to fall back to its own unconfigured default — on at least one real judging machine that
# landed on something inefficient enough that a routine benchmark didn't finish inside the
# worker's 30-minute subprocess timeout. Every run must now be pinned to this machine's own
# physical core count, not left to chance.

def test_measure_uses_detected_physical_cpu_count(monkeypatch):
    captured = {}

    def fake_run_llama_bench(model_path, **kwargs):
        captured.update(kwargs)
        return _rows()

    monkeypatch.setattr(throughput, "run_llama_bench", fake_run_llama_bench)
    monkeypatch.setattr(throughput, "detect_physical_cpu_count", lambda: 4)
    result = throughput.measure(__import__("pathlib").Path("m.gguf"))
    assert captured["n_threads"] == 4
    assert result["threads_used"] == 4


def test_measure_respects_an_explicit_n_threads_override(monkeypatch):
    captured = {}

    def fake_run_llama_bench(model_path, **kwargs):
        captured.update(kwargs)
        return _rows()

    monkeypatch.setattr(throughput, "run_llama_bench", fake_run_llama_bench)
    monkeypatch.setattr(throughput, "detect_physical_cpu_count", lambda: 4)
    throughput.measure(__import__("pathlib").Path("m.gguf"), n_threads=2)
    assert captured["n_threads"] == 2


def test_detect_physical_cpu_count_falls_back_to_logical_when_psutil_cannot_tell(monkeypatch):
    monkeypatch.delenv("ADTC_CPU_THREADS", raising=False)
    monkeypatch.setattr(throughput.psutil, "cpu_count", lambda logical=True: None)
    monkeypatch.setattr(throughput.os, "cpu_count", lambda: 6)
    assert throughput.detect_physical_cpu_count() == 6


def test_detect_physical_cpu_count_honors_worker_topology(monkeypatch):
    monkeypatch.setenv("ADTC_CPU_THREADS", "8")
    monkeypatch.setattr(throughput.psutil, "cpu_count", lambda logical=True: 2)
    assert throughput.detect_physical_cpu_count() == 8


def test_run_llama_bench_includes_thread_flag_when_given(monkeypatch):
    captured = {}

    class _Proc:
        returncode = 0
        stdout = "[]"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(throughput, "_find_llama_bench", lambda: "llama-bench")
    monkeypatch.setattr(throughput.subprocess, "run", fake_run)
    throughput.run_llama_bench(__import__("pathlib").Path("m.gguf"), n_threads=4)
    cmd = captured["cmd"]
    assert "-t" in cmd
    assert cmd[cmd.index("-t") + 1] == "4"

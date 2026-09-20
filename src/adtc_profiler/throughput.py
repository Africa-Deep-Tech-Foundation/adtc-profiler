"""Wrap llama.cpp's `llama-bench` to produce throughput numbers in the schema's shape.

llama-bench --output json emits a JSON array of run rows. The default `-p 512 -n 128`
gives one prompt-processing row (n_prompt=512, n_gen=0) and one generation row
(n_prompt=0, n_gen=128). We extract:
  - throughput.tokens_per_second_generation -> tg row's avg_ts
  - throughput.first_token_latency_ms       -> approximated from pp row's avg_ts
  - throughput.prompt_tokens / generated_tokens for traceability
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import psutil


class LlamaBenchError(RuntimeError):
    """llama-bench failed to produce parseable output."""


def detect_physical_cpu_count() -> int:
    """Physical core count of the machine actually running this benchmark — not logical/
    hyperthreaded count. Without this, measure() left -t unset, so llama-bench fell back to
    its own internal default thread count with no relationship to the real hardware; on at
    least one real judging machine that landed on something inefficient enough that a
    `-p 512 -n 128` throughput run didn't finish inside the worker's 30-minute subprocess
    timeout. Every machine now benchmarks at its own true physical-core count, so results stay
    meaningful and comparable without hardcoding a single number that's wrong for whatever
    hardware actually executes a given run.

    psutil.cpu_count(logical=False) can return None on some exotic platforms/containers where
    physical topology isn't queryable — os.cpu_count() (logical count) is a safe fallback that's
    still far better than an unconfigured, unpredictable llama-bench default.
    """
    # The private hardware worker computes topology once and passes it through so the profiler,
    # benchmark shim, and Phase 6 inference all use the same host-core decision.
    configured = os.environ.get("ADTC_CPU_THREADS", "").strip()
    if configured.isdigit() and int(configured) > 0:
        return int(configured)

    physical = psutil.cpu_count(logical=False)
    if isinstance(physical, int) and physical > 0:
        return physical
    return max(1, os.cpu_count() or 1)


def _find_llama_bench() -> str:
    """Locate llama-bench binary; fail with actionable error if missing."""
    for name in ("llama-bench", "llama.cpp-llama-bench"):
        path = shutil.which(name)
        if path:
            return path
    raise LlamaBenchError(
        "llama-bench not found on PATH. Install llama.cpp:\n"
        "  brew install llama.cpp        # macOS\n"
        "  apt install llama.cpp         # Debian/Ubuntu (if packaged)\n"
        "  or build from https://github.com/ggerganov/llama.cpp"
    )


def run_llama_bench(
    model_path: Path,
    n_prompt: int = 512,
    n_gen: int = 128,
    n_threads: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Invoke llama-bench and return the parsed JSON array of run rows."""
    binary = _find_llama_bench()
    cmd = [
        binary,
        "-m", str(model_path),
        "-p", str(n_prompt),
        "-n", str(n_gen),
        # Pin to CPU: llama-bench defaults to full GPU offload, so an
        # unpinned participant run on an Apple-Silicon/NVIDIA laptop records
        # GPU-accelerated numbers that can never reconcile with the CPU-only
        # audit VM (measured >10x on prompt processing) — an automatic
        # compare failure for participants who did nothing wrong. The
        # challenge target is commodity CPU.
        "-ngl", "0",
        "--output", "json",
    ]
    if n_threads is not None:
        cmd.extend(["-t", str(n_threads)])
    # Note: llama-bench does not accept --seed; it only matters for sampling, which
    # llama-bench doesn't do. seed is still recorded in reproducibility.random_seed
    # for the participant's full inference run. _ = seed (kept for API stability).
    _ = seed

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise LlamaBenchError(
            f"llama-bench exited {proc.returncode}\nstderr:\n{proc.stderr[:2000]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise LlamaBenchError(
            f"llama-bench output was not valid JSON: {e}\nstdout head:\n{proc.stdout[:500]}"
        ) from e


def measure(model_path: Path, *, seed: int = 42, n_threads: int | None = None) -> dict:
    """Run llama-bench and project to the schema's `throughput` block.

    first_token_latency_ms is approximated as 1000/pp_rate (the time to process
    a single prompt token at the measured prompt-processing rate). This is an
    honest approximation; the spec's ±25% throughput tolerance applies.

    n_threads defaults to this machine's own physical core count (detect_physical_cpu_count())
    rather than leaving it unset — see that function's docstring for why an unconfigured
    llama-bench default is unsafe. Callers can still override explicitly if ever needed.
    """
    if n_threads is None:
        n_threads = detect_physical_cpu_count()
    rows = run_llama_bench(model_path, n_threads=n_threads, seed=seed)
    if not rows:
        raise LlamaBenchError("llama-bench returned no rows")

    pp_row = next((r for r in rows if r.get("n_gen", 0) == 0 and r.get("n_prompt", 0) > 0), None)
    tg_row = next((r for r in rows if r.get("n_gen", 0) > 0), None)
    if tg_row is None:
        raise LlamaBenchError(f"no generation row in llama-bench output: {rows}")

    tg_rate = float(tg_row.get("avg_ts") or 0.0)
    if tg_rate <= 0:
        raise LlamaBenchError(f"non-positive generation rate: {tg_rate}")

    # Time-to-first-token = time to ingest the whole prompt at the prompt-processing
    # rate. (Not 1/pp_rate, which is the time per single prompt token — that
    # would only equal TTFT for a 1-token prompt.) Falls back to 1/tg_rate when
    # we have no pp measurement (vanishingly rare; llama-bench always runs pp by default).
    if pp_row is not None and pp_row.get("avg_ts"):
        pp_rate = float(pp_row["avg_ts"])
        n_prompt = int(pp_row.get("n_prompt", 0)) or 512
        first_token_ms = (n_prompt / pp_rate) * 1000.0 if pp_rate > 0 else 1000.0 / tg_rate
    else:
        first_token_ms = 1000.0 / tg_rate

    return {
        "tokens_per_second_generation": round(tg_rate, 2),
        "first_token_latency_ms": round(first_token_ms, 2),
        "prompt_tokens": int(pp_row.get("n_prompt", 0)) if pp_row else 0,
        "generated_tokens": int(tg_row.get("n_gen", 0)),
        "threads_used": n_threads,
    }

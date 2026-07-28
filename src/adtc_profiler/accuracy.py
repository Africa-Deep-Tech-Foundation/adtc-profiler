"""Run lm-evaluation-harness against the model and project to the schema's accuracy block.

Heavy deps — install via `uv sync --extra accuracy`. If lm-eval is not available
or `--skip-accuracy` is passed, returns an empty list (schema-valid: `accuracy: []`).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


class AccuracyError(RuntimeError):
    """lm-eval-harness failed to produce parseable output."""


def is_available() -> bool:
    """Whether lm_eval is callable in the current environment."""
    return shutil.which("lm_eval") is not None or shutil.which("lm-eval") is not None


def _lm_eval_bin() -> str:
    for name in ("lm_eval", "lm-eval"):
        path = shutil.which(name)
        if path:
            return path
    raise AccuracyError(
        "lm_eval not found. Install with `uv sync --extra accuracy` or "
        "`pip install lm-eval`."
    )


def run_benchmark(
    model_path: Path,
    *,
    task: str = "arc_easy",
    limit: int = 50,
    language: str = "en",
    seed: int = 42,
) -> dict:
    """Run lm_eval on the GGUF model and return one accuracy row.

    Defaults to a small ARC-Easy subset (50 questions) for fast smoke testing.
    Real audits use the full hidden 30% validation subset distributed by judges.
    """
    binary = _lm_eval_bin()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "results.json"
        # lm-eval's `gguf` model type is an HTTP client for a llama.cpp server
        # and cannot load a file; the `hf` backend dequantizes the GGUF locally
        # via transformers, which keeps the eval self-contained.
        cmd = [
            binary,
            "--model", "hf",
            "--model_args",
            f"pretrained={model_path.parent},gguf_file={model_path.name}",
            "--tasks", task,
            "--limit", str(limit),
            "--seed", str(seed),
            "--output_path", str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise AccuracyError(
                f"lm_eval exited {proc.returncode}\nstderr:\n{proc.stderr[:2000]}"
            )
        # lm_eval writes results to <out_path>/<model>/results_*.json
        result_files = list(Path(tmpdir).rglob("results*.json"))
        if not result_files:
            raise AccuracyError(f"no results file produced under {tmpdir}")
        data = json.loads(result_files[0].read_text())
        results = data.get("results", {})
        if task not in results:
            raise AccuracyError(f"task {task!r} not in results: {list(results)}")
        task_results = results[task]
        # Prefer acc_norm, fall back to acc
        score = task_results.get("acc_norm,none") or task_results.get("acc,none") or 0.0
        return {
            "benchmark": task,
            "dataset_version": "lm-eval-harness",
            "language": language,
            "samples": limit,
            "score": round(float(score), 4),
            "metric": "acc_norm" if "acc_norm,none" in task_results else "acc",
        }

"""Run lm-evaluation-harness against the model and project to the schema's accuracy block.

The model is evaluated in its quantized form, in-process, via llama-cpp-python
(same llama.cpp runtime the challenge targets): we tokenize context+continuation,
evaluate once, and read continuation log-probabilities straight from the logits.
lm-eval supplies the datasets, prompting, and metrics.

Why not lm-eval's built-in backends:
- `hf` dequantizes the GGUF to FP32 — a 3B model needs ~12 GB, blowing the 8 GB
  Standard Laptop profile even though its quantized file fits.
- `gguf` is an HTTP client for a llama.cpp server, and llama-cpp-python's echoed
  logprobs are shifted by one position (token i is scored by the distribution
  that predicts token i+1), which makes every ranking near-random.

If the accuracy stack is unavailable or `--skip-accuracy` is passed, callers emit
an empty list (schema-valid: `accuracy: []`).
"""
from __future__ import annotations

from pathlib import Path

_N_CTX = 2048


class AccuracyError(RuntimeError):
    """The accuracy pipeline failed to produce a scoreable result."""


def is_available() -> bool:
    """Whether the accuracy stack (lm-eval + llama-cpp-python) is importable."""
    try:
        import llama_cpp  # noqa: F401
        import lm_eval  # noqa: F401
    except ImportError:
        return False
    return True


def _common_prefix_len(full: list[int], prefix: list[int]) -> int:
    """Length of the shared token prefix, at least 1.

    The continuation is scored from the first token where the tokenizations
    diverge — BPE may merge characters across the context/continuation
    boundary, so slicing by len(prefix) alone would be wrong. At least one
    token must remain as conditioning context.
    """
    n = 0
    while n < min(len(full), len(prefix)) and full[n] == prefix[n]:
        n += 1
    return max(n, 1)


def _sequence_logprob(scores, tokens: list[int], start: int) -> tuple[float, bool]:
    """Sum log P(tokens[start:]) from per-position logits.

    `scores[i]` are the logits produced after evaluating tokens[0..i], i.e. the
    distribution predicting tokens[i+1] — so tokens[pos] is scored against
    scores[pos - 1].
    """
    import numpy as np

    total = 0.0
    greedy = True
    for pos in range(start, len(tokens)):
        row = np.asarray(scores[pos - 1], dtype=np.float64)
        row = row - row.max()
        logprob_row = row - np.log(np.exp(row).sum())
        total += float(logprob_row[tokens[pos]])
        if int(row.argmax()) != tokens[pos]:
            greedy = False
    return total, greedy


def _make_lm(model_path: Path):
    """lm-eval LM adapter that scores continuations via llama_cpp directly."""
    from llama_cpp import Llama
    from lm_eval.api.model import LM

    class _LlamaCppLM(LM):
        def __init__(self) -> None:
            super().__init__()
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=_N_CTX,
                logits_all=True,
                verbose=False,
            )

        def _tokenize(self, text: str) -> list[int]:
            return self._llm.tokenize(text.encode("utf-8"), add_bos=True, special=False)

        def loglikelihood(self, requests, disable_tqdm: bool = False):
            res = []
            for context, continuation in [req.args for req in requests]:
                full = self._tokenize(context + continuation)
                if len(full) > _N_CTX:
                    raise AccuracyError(
                        f"prompt of {len(full)} tokens exceeds n_ctx={_N_CTX}"
                    )
                start = _common_prefix_len(full, self._tokenize(context))
                if start >= len(full):
                    raise AccuracyError(
                        f"continuation {continuation!r} produced no tokens to score"
                    )
                self._llm.reset()
                self._llm.eval(full)
                res.append(_sequence_logprob(self._llm.scores, full, start))
            return res

        def generate_until(self, requests, disable_tqdm: bool = False):
            res = []
            for request in [req.args for req in requests]:
                context, gen_kwargs = request
                until = gen_kwargs.get("until") or []
                out = self._llm.create_completion(
                    prompt=context,
                    max_tokens=gen_kwargs.get("max_gen_toks", 256),
                    temperature=0.0,
                    stop=until,
                )
                res.append(out["choices"][0]["text"])
            return res

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
            raise NotImplementedError(
                "rolling loglikelihood (perplexity tasks) is not supported"
            )

    return _LlamaCppLM()


def run_benchmark(
    model_path: Path,
    *,
    task: str = "arc_easy",
    limit: int = 50,
    language: str = "en",
    seed: int = 42,
) -> dict:
    """Evaluate the GGUF quantized and return one accuracy row.

    Defaults to a small ARC-Easy subset (50 questions) for fast smoke testing.
    Real audits use the full hidden 30% validation subset distributed by judges.
    """
    if not is_available():
        raise AccuracyError(
            "accuracy stack not installed. Reinstall the profiler "
            "(`python3 -m pip install adtc-profiler`)."
        )
    import lm_eval

    lm = _make_lm(model_path)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[task],
        limit=limit,
        random_seed=seed,
        numpy_random_seed=seed,
        fewshot_random_seed=seed,
    )

    task_results = ((results or {}).get("results") or {}).get(task)
    if not task_results:
        raise AccuracyError(f"task {task!r} produced no results")
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

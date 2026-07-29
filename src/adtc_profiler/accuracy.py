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


def _logprob_of(row, token: int) -> tuple[float, bool]:
    """Log-softmax a logits row and return (logprob of token, is-argmax)."""
    import numpy as np

    row = np.asarray(row, dtype=np.float64)
    row = row - row.max()
    logprob_row = row - np.log(np.exp(row).sum())
    return float(logprob_row[token]), int(row.argmax()) == token


def _extract_score(task_results: dict) -> tuple[float, str] | None:
    """Pick (score, metric) from an lm-eval task result dict.

    Keys are "<metric>,<filter>". Prefer acc_norm, then acc, then any other
    numeric metric (exact_match for generation tasks, etc.). Explicit None
    checks — a legitimate score of exactly 0.0 must survive.
    """
    preferred = ("acc_norm,none", "acc,none")
    for key in preferred:
        if key in task_results and isinstance(task_results[key], (int, float)):
            return float(task_results[key]), key.split(",")[0]
    for key, value in task_results.items():
        metric = key.split(",")[0]
        if metric in ("alias",) or metric.endswith("_stderr"):
            continue
        if isinstance(value, (int, float)):
            return float(value), metric
    return None


def _make_lm(model_path: Path):
    """lm-eval LM adapter that scores continuations via llama_cpp directly."""
    from llama_cpp import Llama
    from lm_eval.api.model import LM

    class _LlamaCppLM(LM):
        def __init__(self) -> None:
            super().__init__()
            # No logits_all: prompt logits are read incrementally, one position
            # at a time, so memory stays O(vocab) instead of O(n_ctx * vocab)
            # (~1 GB for 128k-vocab models — fatal on the 8 GB audit VM).
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=_N_CTX,
                verbose=False,
            )

        def _tokenize(self, text: str) -> list[int]:
            return self._llm.tokenize(text.encode("utf-8"), add_bos=True, special=False)

        def _last_logits(self):
            # With logits_all=False, llama-cpp-python never populates
            # Llama.scores (the write branch is stubbed out) — the only source
            # of the last decoded position's logits is the C context buffer,
            # read exactly the way Llama.eval() itself does.
            import numpy as np

            llm = self._llm
            if not hasattr(llm, "_ctx") or not hasattr(llm._ctx, "get_logits"):
                raise AccuracyError(
                    "llama-cpp-python internals changed: cannot read logits "
                    "(expected Llama._ctx.get_logits). Pin llama-cpp-python 0.3.x."
                )
            return np.ctypeslib.as_array(
                llm._ctx.get_logits(), shape=(llm._n_vocab,)
            )

        def loglikelihood(self, requests, disable_tqdm: bool = False):
            res = []
            for context, continuation in [req.args for req in requests]:
                full = self._tokenize(context + continuation)
                start = _common_prefix_len(full, self._tokenize(context))
                if start >= len(full):
                    raise AccuracyError(
                        f"continuation {continuation!r} produced no tokens to score"
                    )
                if len(full) > _N_CTX:
                    # Left-truncate the context (reference lm-eval backends do
                    # the same) — a long document must not abort the whole run.
                    cut = len(full) - _N_CTX
                    if start - cut < 1:
                        raise AccuracyError(
                            f"continuation of {len(full) - start} tokens cannot "
                            f"fit n_ctx={_N_CTX} with any context"
                        )
                    full = full[cut:]
                    start -= cut
                self._llm.reset()
                self._llm.eval(full[:start])
                total = 0.0
                greedy = True
                for pos in range(start, len(full)):
                    logprob, is_argmax = _logprob_of(self._last_logits(), full[pos])
                    total += logprob
                    greedy = greedy and is_argmax
                    if pos + 1 < len(full):
                        self._llm.eval([full[pos]])
                res.append((total, greedy))
            return res

        def generate_until(self, requests, disable_tqdm: bool = False):
            res = []
            for request in [req.args for req in requests]:
                context, gen_kwargs = request
                until = gen_kwargs.get("until") or []
                max_gen = int(gen_kwargs.get("max_gen_toks", 256))
                tokens = self._tokenize(context)
                budget = _N_CTX - max_gen - 1
                if budget < 1:
                    raise AccuracyError(
                        f"max_gen_toks={max_gen} leaves no room in n_ctx={_N_CTX}"
                    )
                if len(tokens) > budget:
                    tokens = tokens[-budget:]  # left-truncate, keep recent context
                    context = self._llm.detokenize(tokens).decode("utf-8", "replace")
                out = self._llm.create_completion(
                    prompt=context,
                    max_tokens=max_gen,
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
    extracted = _extract_score(task_results)
    if extracted is None:
        raise AccuracyError(
            f"task {task!r} produced no numeric metric; keys: {sorted(task_results)}"
        )
    score, metric = extracted
    # Report how many documents were actually evaluated — lm-eval caps the
    # requested limit at the dataset size.
    n_samples = ((results.get("n-samples") or {}).get(task) or {}).get("effective")
    return {
        "benchmark": task,
        "dataset_version": "lm-eval-harness",
        "language": language,
        "samples": int(n_samples) if isinstance(n_samples, int) else limit,
        "score": round(score, 4),
        "metric": metric,
    }

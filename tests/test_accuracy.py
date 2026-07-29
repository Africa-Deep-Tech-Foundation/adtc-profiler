"""Unit tests for the accuracy scoring helpers.

llama.cpp emits, after evaluating tokens[0..i], the logits predicting
tokens[i+1] — the eval loop must therefore read the logits BEFORE feeding the
next token. `_logprob_of` and `_common_prefix_len` carry that math;
`_extract_score` guards the report against silent-zero and mislabeled metrics.
"""
import math

import numpy as np
import pytest

from adtc_profiler.accuracy import _common_prefix_len, _extract_score, _logprob_of


def _one_hot(tok: int, vocab: int = 4, hot: float = 10.0) -> np.ndarray:
    row = np.zeros(vocab)
    row[tok] = hot
    return row


def test_logprob_of_uniform_row():
    logprob, is_argmax = _logprob_of(np.zeros(4), 2)
    assert logprob == pytest.approx(math.log(0.25))
    assert is_argmax is False  # argmax of uniform is index 0, not 2


def test_logprob_of_peaked_row():
    logprob, is_argmax = _logprob_of(_one_hot(2), 2)
    assert logprob == pytest.approx(math.log(math.exp(10) / (math.exp(10) + 3)))
    assert is_argmax is True


def test_logprob_of_survives_extreme_logits():
    row = np.array([1e4, 0.0, -1e4, 0.0])
    logprob, is_argmax = _logprob_of(row, 0)
    assert logprob == pytest.approx(0.0, abs=1e-6)
    assert is_argmax is True


def test_common_prefix_len_basic():
    assert _common_prefix_len([1, 2, 3, 4], [1, 2]) == 2


def test_common_prefix_len_bpe_boundary_merge():
    # Continuation merged into the context's last token: tokenizations diverge
    # before len(prefix); scoring must start at the divergence point.
    assert _common_prefix_len([1, 2, 9], [1, 2, 3]) == 2


def test_common_prefix_len_never_zero():
    # At least one conditioning token must remain even if tokenizations
    # diverge immediately.
    assert _common_prefix_len([9, 2], [1, 2]) == 1


def test_extract_score_prefers_acc_norm():
    score, metric = _extract_score({"acc,none": 0.5, "acc_norm,none": 0.6})
    assert (score, metric) == (0.6, "acc_norm")


def test_extract_score_zero_acc_norm_is_not_replaced_by_acc():
    # Falsy-or-chain regression: a legitimate 0.0 must survive as-is.
    score, metric = _extract_score({"acc_norm,none": 0.0, "acc,none": 0.04})
    assert (score, metric) == (0.0, "acc_norm")


def test_extract_score_generation_task_metric():
    # gsm8k-style results have no acc keys at all.
    score, metric = _extract_score({
        "alias": "gsm8k",
        "exact_match,strict-match": 0.12,
        "exact_match_stderr,strict-match": 0.01,
    })
    assert (score, metric) == (0.12, "exact_match")


def test_extract_score_no_numeric_metric_returns_none():
    assert _extract_score({"alias": "sometask"}) is None

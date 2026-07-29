"""Unit tests for the accuracy scoring math.

llama.cpp's `scores[i]` holds the logits produced *after* token i (predicting
token i+1), so tokens[pos] must be scored against scores[pos - 1] — an
off-by-one here makes every ranking near-random, which is exactly the bug the
llama-cpp-python server's echoed logprobs exhibit.
"""
import math

import numpy as np
import pytest

from adtc_profiler.accuracy import _common_prefix_len, _sequence_logprob

# Toy vocab of 4 tokens. Row i = logits emitted after consuming token i.
# Uniform rows give each token log(1/4) so expected sums are exact.
UNIFORM = np.zeros(4)


def _one_hot(tok: int, hot: float = 10.0) -> np.ndarray:
    row = np.zeros(4)
    row[tok] = hot
    return row


def test_scores_are_read_from_previous_position():
    # scores[0] strongly predicts token 2; sequence is [0, 2].
    # If alignment were off by one, the scored row would be scores[1] (uniform).
    scores = [_one_hot(2), UNIFORM]
    total, greedy = _sequence_logprob(scores, tokens=[0, 2], start=1)
    assert total == pytest.approx(math.log(math.exp(10) / (math.exp(10) + 3)))
    assert greedy is True


def test_uniform_logits_give_log_quarter_per_token():
    scores = [UNIFORM, UNIFORM, UNIFORM]
    total, greedy = _sequence_logprob(scores, tokens=[0, 1, 2], start=1)
    assert total == pytest.approx(2 * math.log(0.25))
    assert greedy is False


def test_greedy_false_when_any_token_not_argmax():
    scores = [_one_hot(1), _one_hot(3), UNIFORM]
    # token at pos 1 is argmax of scores[0], token at pos 2 is not argmax of scores[1]
    total, greedy = _sequence_logprob(scores, tokens=[0, 1, 2], start=1)
    assert greedy is False


def test_start_offset_skips_context_tokens():
    scores = [_one_hot(1), _one_hot(2), _one_hot(3)]
    all_from_1, _ = _sequence_logprob(scores, tokens=[0, 1, 2, 3], start=1)
    only_last, _ = _sequence_logprob(scores, tokens=[0, 1, 2, 3], start=3)
    assert only_last > all_from_1  # fewer summed terms
    assert only_last == pytest.approx(
        math.log(math.exp(10) / (math.exp(10) + 3))
    )


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

"""Tests for the rating-probability combination (behavioral.py).

Regression cover for a bug that was silent for four models: a single-token
rating was credited the full probability of its first token, even when that
token also opens a longer rating ("1" opens 1 and 10-19). The result looked
like a valid rating and only showed up as an implausible spike at rating 1.

Run: python -m pytest mech_interp/test_completion_factors.py
"""

import torch

from mech_interp.behavioral import combine_first_and_second_probs, completion_factors

# Llama-3.x: every rating 0-20 is a single, distinct token. No prefix overlap,
# so nothing should be rescaled.
SINGLE_TOKEN_SEQS = {r: [100 + r] for r in range(21)}

# Gemma/Qwen: 0-9 single tokens, 10-19 are "1"+digit, 20 is "2"+"0". Token 201
# ("1") is shared by rating 1 and ratings 10-19; token 202 ("2") by 2 and 20.
DIGIT = {d: 200 + d for d in range(10)}
SPLIT_SEQS = {r: [DIGIT[r]] for r in range(10)}
SPLIT_SEQS.update({r: [DIGIT[1], DIGIT[r - 10]] for r in range(10, 20)})
SPLIT_SEQS[20] = [DIGIT[2], DIGIT[0]]


def test_single_token_tokenizer_is_untouched():
    """Llama was never affected; the fix must not perturb it."""
    factors = completion_factors(SINGLE_TOKEN_SEQS, {})
    assert set(factors) == set(range(21))
    assert all(f == 1.0 for f in factors.values())


def test_shared_first_token_is_discounted_by_its_extensions():
    """Rating 1 keeps only the mass that does not continue into 10-19."""
    # 80% of the "1" mass continues as a second digit, spread over 10-19.
    second = {r: 0.08 for r in range(10, 20)}
    second[20] = 0.30
    factors = completion_factors(SPLIT_SEQS, second)

    assert factors[1] == pytest_approx(1.0 - 0.8)
    assert factors[2] == pytest_approx(1.0 - 0.30)
    # A rating whose token opens nothing longer is untouched.
    assert factors[5] == 1.0
    # Two-token ratings are weighted by their own conditional, not a factor.
    assert 10 not in factors


def test_factor_is_clamped_when_extensions_exceed_one():
    """Second-token conditionals are independent probabilities and can sum past
    1 under an ill-behaved model; the factor must not go negative."""
    second = {r: 0.5 for r in range(10, 20)}  # sums to 5.0
    assert completion_factors(SPLIT_SEQS, second)[1] == 0.0


def test_the_bug_itself_expected_rating_is_no_longer_dragged_down():
    """The regression: a model that wants to say 14 must not read as ~1.

    All the "1" mass continues to a second digit, and that digit is 4. The
    answer is unambiguously 14, so E[r] must be 14 -- the old code returned a
    value near 1, because rating 1 was credited the whole of P("1").
    """
    first = torch.zeros(300)
    first[DIGIT[1]] = 1.0  # every path starts with "1"
    second = {r: 0.0 for r in range(10, 20)}
    second[14] = 1.0  # ... and continues with "4"

    assert combine_first_and_second_probs(first, SPLIT_SEQS, second) == pytest_approx(14.0)


def test_mass_splits_between_a_short_and_a_long_rating():
    """P("1")=1 split 30/70 between answering "1" and answering "13"."""
    first = torch.zeros(300)
    first[DIGIT[1]] = 1.0
    second = {r: 0.0 for r in range(10, 20)}
    second[13] = 0.7

    # factor for rating 1 is 1 - 0.7 = 0.3, so E[r] = .3*1 + .7*13 = 9.4
    assert combine_first_and_second_probs(first, SPLIT_SEQS, second) == pytest_approx(9.4)


def test_single_token_model_expected_rating_unchanged():
    """Same distribution under a non-splitting tokenizer gives the plain mean."""
    first = torch.zeros(300)
    first[SINGLE_TOKEN_SEQS[8][0]] = 0.5
    first[SINGLE_TOKEN_SEQS[12][0]] = 0.5
    assert combine_first_and_second_probs(first, SINGLE_TOKEN_SEQS, {}) == pytest_approx(10.0)


def pytest_approx(value, tol=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < tol

        def __repr__(self):
            return f"~{value}"

    return _Approx()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")

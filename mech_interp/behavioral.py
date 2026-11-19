"""Step 1: behavioral replication via one forward pass per prompt (no sampling).

Reads the probability distribution over rating tokens "0".."20" at the final
position and computes the expected rating E[r], instead of sampling text.
"""

import argparse

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mech_interp.config import (
    ANSWER_PREFIX_VARIANTS,
    DEBUG_MODEL,
    MODEL_REPO_IDS,
    RATING_MAX,
    RATING_MIN,
    RESULTS_DIR,
    encode_prompt,
    get_device,
    load_model_and_tokenizer,
)
from mech_interp.stimuli import Stimulus, build_stimuli
from mech_interp.stimuli_jury import (
    CONTRASTS as JURY_CONTRASTS,
    CONTROL_CONTRASTS as JURY_CONTROL_CONTRASTS,
    JURY_RATING_MAX,
    JURY_RATING_MIN,
    build_stimuli as jury_stimuli,
)


class PrefixTokenizationError(ValueError):
    """An answer prefix whose tokens merge with the rating's at the boundary.

    Signals that this prefix variant is unusable for this tokenizer, so
    select_answer_prefix() should try the next one rather than abort.
    """


# A prefix is only usable if the model actually wants to emit a number after
# it. Below this share of next-token probability mass on rating tokens, the
# scaffold is being ignored (or mis-tokenized) and E[r] would be computed from
# a renormalized sliver of the distribution -- garbage that looks like a valid
# rating. Fail loudly instead.
MIN_RATING_TOKEN_MASS = 0.5

# A contrast between conditions is only interpretable if the model's rating
# responds to the conditions at all. If E[r] is near-constant across a stimulus
# set, every difference we compute is a fraction of a point of noise on top of a
# fixed prior, and reporting its *sign* is meaningless. Observed: qwen2.5-0.5b
# spans 0.46 rating points across all six jury conditions (humans span 4.8),
# i.e. three pieces of prosecution evidence move it almost not at all.
MIN_DISCRIMINATION_RANGE = 1.0


class LowDiscriminationError(RuntimeError):
    """The model's E[r] barely moves across conditions, so contrasts are noise."""


def check_discrimination(
    values: dict[str, float], label: str, min_range: float = MIN_DISCRIMINATION_RANGE,
    strict: bool = True,
) -> float:
    """Assert E[r] actually varies across conditions; return the observed range.

    Raises by default rather than warning, matching select_answer_prefix: a
    model that ignores the stimuli produces plausible-looking numbers whose
    contrasts are pure noise, and that failure is invisible downstream.
    """
    spread = max(values.values()) - min(values.values())
    lo = min(values, key=values.get)
    hi = max(values, key=values.get)
    message = (
        f"{label}: E[r] spans only {spread:.3f} rating points across "
        f"{len(values)} conditions ({lo}={values[lo]:.3f} .. {hi}={values[hi]:.3f}); "
        f"need {min_range}. The model is not responding to the stimuli, so every "
        f"contrast computed from these values is noise around a fixed prior."
    )
    if spread < min_range:
        if strict:
            raise LowDiscriminationError(message)
        print(f"WARNING: {message}")
    else:
        print(f"{label}: discrimination OK, E[r] spans {spread:.3f} rating points.")
    return spread


def get_rating_token_sequences(
    tokenizer, answer_prefix: str,
    rating_min: int = RATING_MIN, rating_max: int = RATING_MAX,
) -> dict[int, list[int]]:
    """Map rating value -> token id sequence, *as tokenized after `answer_prefix`*.

    Tokenizing "14" standalone can differ from how it tokenizes following
    '{"score": ' (leading-space variants, digit merges), so each rating is
    encoded in context and the prefix's own tokens are stripped back off.
    If the boundary merges -- the prefix's last token absorbing the first
    digit -- the diff is meaningless, so this raises and the caller falls
    back to another prefix variant.

    Most Llama-3.x models tokenize 0-20 as single tokens; Qwen and Gemma split
    some (e.g. "10" -> "1","0"). Both are handled, up to 2 tokens per rating.
    """
    prefix_ids = tokenizer.encode(answer_prefix, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    sequences = {}
    max_len = 0
    for rating in range(rating_min, rating_max + 1):
        full_ids = tokenizer.encode(answer_prefix + str(rating), add_special_tokens=False)
        if full_ids[:n_prefix] != prefix_ids:
            raise PrefixTokenizationError(
                f"Prefix {answer_prefix!r} merges with rating {rating} under "
                f"{tokenizer.name_or_path}; cannot isolate the rating's tokens."
            )
        ids = full_ids[n_prefix:]
        if not ids:
            raise PrefixTokenizationError(
                f"Rating {rating} produced no tokens after prefix {answer_prefix!r}."
            )
        sequences[rating] = ids
        max_len = max(max_len, len(ids))

    if max_len > 2:
        raise ValueError(
            f"Rating tokenizes to {max_len} tokens for {tokenizer.name_or_path}; "
            "expected_rating() only handles up to 2 tokens per rating."
        )
    return sequences


def rating_token_mass(
    model, tokenizer, prompt: str, answer_prefix: str,
    rating_token_seqs: dict[int, list[int]], device: str,
) -> float:
    """Share of next-token probability sitting on any rating's first token."""
    prompt_ids = encode_prompt(tokenizer, prompt, device, answer_prefix)
    with torch.no_grad():
        probs = model(prompt_ids).logits[0, -1].softmax(dim=0)
    first_token_ids = {ids[0] for ids in rating_token_seqs.values()}
    return float(sum(probs[tid].item() for tid in first_token_ids))


def select_answer_prefix(
    model, tokenizer, sample_prompt: str, device: str,
    rating_min: int = RATING_MIN, rating_max: int = RATING_MAX,
) -> tuple[str, dict[int, list[int]]]:
    """Pick the ANSWER_PREFIX_VARIANTS entry this model responds to best.

    Returns (prefix, rating_token_seqs). Raises if no variant gets the model to
    put meaningful mass on rating tokens -- a silent failure here would poison
    every downstream E[r], so it must be loud.

    rating_min/max are parameterized because tasks do not share a scale: the
    similarity task uses 1-20 (Tversky & Gati 1978) and the jury task 0-20
    (Trueblood & Busemeyer 2011).
    """
    candidates = []
    for prefix in ANSWER_PREFIX_VARIANTS:
        try:
            seqs = get_rating_token_sequences(tokenizer, prefix, rating_min, rating_max)
        except PrefixTokenizationError:
            continue
        mass = rating_token_mass(model, tokenizer, sample_prompt, prefix, seqs, device)
        candidates.append((mass, prefix, seqs))

    if not candidates:
        raise RuntimeError(
            f"No usable answer prefix for {tokenizer.name_or_path}: every variant in "
            f"{ANSWER_PREFIX_VARIANTS} merged with the rating tokens."
        )

    mass, prefix, seqs = max(candidates, key=lambda c: c[0])
    if mass < MIN_RATING_TOKEN_MASS:
        raise RuntimeError(
            f"Best answer prefix {prefix!r} puts only {mass:.3f} of next-token mass on "
            f"rating tokens for {tokenizer.name_or_path} (need {MIN_RATING_TOKEN_MASS}). "
            "The model is not answering with a bare number -- inspect the prompt with "
            "sanity_check_top_tokens() before trusting any E[r]."
        )
    print(f"Answer prefix {prefix!r} selected ({mass:.3f} of mass on rating tokens).")
    return prefix, seqs


def sanity_check_top_tokens(
    model, tokenizer, prompt: str, device: str, answer_prefix: str | None = None, n_show: int = 10
) -> None:
    """Log the top-n predicted tokens for one prompt as a sanity check."""
    inputs = encode_prompt(tokenizer, prompt, device, answer_prefix)
    with torch.no_grad():
        logits = model(inputs).logits[0, -1]
    top = torch.topk(logits.softmax(-1), n_show)
    tokens = [tokenizer.decode([tid]) for tid in top.indices.tolist()]
    print(f"Top-{n_show} tokens for prompt: {prompt!r}")
    for tok, prob in zip(tokens, top.values.tolist()):
        print(f"  {tok!r}: {prob:.4f}")


def compute_second_token_probs(
    model, prompt_ids: torch.Tensor, rating_token_seqs: dict[int, list[int]], device: str
) -> dict[int, float]:
    """P(second token | prompt, true first token) for every 2-token rating.

    Always computed via one real, full-depth batched forward pass: a continuation
    token has no intermediate-layer analogue (an intermediate layer only has a
    well-defined next-token distribution for the position right after the
    prompt), so logit_lens.py reuses this same real conditional at every layer
    and only examines the *first* token's decision layer-by-layer.
    """
    two_token_ratings = [r for r, ids in rating_token_seqs.items() if len(ids) == 2]
    if not two_token_ratings:
        return {}
    first_tokens = torch.tensor(
        [[rating_token_seqs[r][0]] for r in two_token_ratings], device=device
    )
    batch_inputs = torch.cat([prompt_ids.repeat(len(two_token_ratings), 1), first_tokens], dim=1)
    with torch.no_grad():
        second_logits = model(batch_inputs).logits[:, -1, :]
    second_probs = second_logits.softmax(dim=-1)
    return {
        r: float(second_probs[i, rating_token_seqs[r][1]].item())
        for i, r in enumerate(two_token_ratings)
    }


def completion_factors(
    rating_token_seqs: dict[int, list[int]], second_token_probs: dict[int, float]
) -> dict[int, float]:
    """P(the answer stops here), per single-token rating.

    A single-token rating is only the model's answer if no digit follows it. The
    token "1" opens rating 1 *and* every rating from 10 to 19, so crediting
    rating 1 the whole of P("1") absorbs all of their mass -- and since the
    10-19 band is exactly where the human ratings sit, that biases E[r] sharply
    downward. Measured on qwen2.5-0.5b: a digit follows "1" 97.3% of the time,
    so 97.3% of the probability previously assigned to rating 1 belonged
    elsewhere. Correcting it raises E[r] by ~2 points and roughly triples the
    measured contrasts.

    The share to subtract is already known -- it is precisely the second-token
    conditionals -- so this costs no extra forward pass:

        P(answer = 1) = P("1") * (1 - sum_d P(d | "1"))

    Ratings whose first token opens nothing longer get a factor of 1. That
    covers every rating on Llama-3.x, which tokenizes 0-20 as single tokens and
    was therefore never affected; only the split-tokenizer models (Gemma, Qwen)
    change.
    """
    extended: dict[int, float] = {}
    for rating, ids in rating_token_seqs.items():
        if len(ids) == 2:
            extended[ids[0]] = extended.get(ids[0], 0.0) + second_token_probs.get(rating, 0.0)
    return {
        rating: max(0.0, 1.0 - extended.get(ids[0], 0.0))
        for rating, ids in rating_token_seqs.items()
        if len(ids) == 1
    }


def combine_first_and_second_probs(
    first_probs: torch.Tensor,
    rating_token_seqs: dict[int, list[int]],
    second_token_probs: dict[int, float],
) -> float:
    """E[r] from a first-token distribution (real or logit-lens-derived) plus
    precomputed second-token conditionals for 2-token ratings.

    Both rating lengths are weighted by the probability that the answer is
    *exactly* that rating: a 2-token rating by its second token's conditional,
    a 1-token rating by its completion factor (see completion_factors)."""
    factors = completion_factors(rating_token_seqs, second_token_probs)
    rating_probs: dict[int, float] = {}
    for rating, ids in rating_token_seqs.items():
        p = float(first_probs[ids[0]].item())
        if len(ids) == 2:
            p *= second_token_probs.get(rating, 0.0)
        else:
            p *= factors.get(rating, 1.0)
        rating_probs[rating] = p
    total = sum(rating_probs.values())
    if total <= 0:
        return float("nan")
    return sum(r * (p / total) for r, p in rating_probs.items())


def rating_first_token_ids(rating_token_seqs: dict[int, list[int]]) -> list[int]:
    """Distinct first tokens across all ratings (several ratings share one:
    "1" begins 1 and 10-19)."""
    return sorted({ids[0] for ids in rating_token_seqs.values()})


def first_token_distribution(probs: torch.Tensor, first_ids: list[int]) -> torch.Tensor:
    """Next-token distribution restricted to rating first-tokens, renormalized."""
    restricted = probs[first_ids]
    total = restricted.sum()
    return restricted / total if total > 0 else restricted


def expected_rating_and_first_probs(
    model, tokenizer, prompt: str, rating_token_seqs: dict[int, list[int]], device: str,
    answer_prefix: str | None = None,
) -> tuple[float, torch.Tensor]:
    """E[r] plus the full next-token distribution it was computed from.

    Returned together because patching needs both and they come from the same
    forward pass -- E[r] alone cannot support a tokenization-independent
    recovery metric (see patching.run_patching_sweep).
    """
    prompt_ids = encode_prompt(tokenizer, prompt, device, answer_prefix)
    with torch.no_grad():
        first_logits = model(prompt_ids).logits[0, -1]
    first_probs = first_logits.softmax(dim=0)
    second_token_probs = compute_second_token_probs(model, prompt_ids, rating_token_seqs, device)
    e_r = combine_first_and_second_probs(first_probs, rating_token_seqs, second_token_probs)
    return e_r, first_probs


def expected_rating(
    model, tokenizer, prompt: str, rating_token_seqs: dict[int, list[int]], device: str,
    answer_prefix: str | None = None,
) -> float:
    """E[r] = sum_r P(token sequence for r | prompt).

    Single-token ratings are scored directly off the prompt's final-position
    logits. Two-token ratings need a second, batched forward pass to score
    the conditional probability of the second token given the first.
    """
    return expected_rating_and_first_probs(
        model, tokenizer, prompt, rating_token_seqs, device, answer_prefix
    )[0]


def run_behavioral(model_key: str, device: str | None = None, sanity_check: bool = True) -> pd.DataFrame:
    device = get_device(device)

    model, tokenizer = load_model_and_tokenizer(model_key, device)

    stimuli = build_stimuli(RATING_MIN, RATING_MAX)

    answer_prefix, rating_token_seqs = select_answer_prefix(
        model, tokenizer, stimuli[0].prompt, device
    )

    if sanity_check:
        for stim in stimuli[:3]:
            sanity_check_top_tokens(model, tokenizer, stim.prompt, device, answer_prefix)

    rows = []
    for stim in stimuli:
        e_r = expected_rating(
            model, tokenizer, stim.prompt, rating_token_seqs, device, answer_prefix
        )
        rows.append(
            {
                "model": model_key,
                "pair": f"{stim.pair[0]}-{stim.pair[1]}",
                "is_control": stim.is_control,
                "template": stim.template_name,
                "task_type": stim.task_type,
                "order": stim.order,
                "expected_rating": e_r,
            }
        )

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"behavioral_{model_key}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    return df


def run_jury(
    model_key: str, device: str | None = None, strict: bool = True
) -> pd.DataFrame:
    """Jury order-effect task (Trueblood & Busemeyer 2011); see stimuli_jury.py.

    Kept separate from run_behavioral rather than merged: the two tasks have
    different rating scales, different stimulus schemas, and only the country
    task has a per-pair structure to aggregate over.
    """
    device = get_device(device)
    model, tokenizer = load_model_and_tokenizer(model_key, device)

    stimuli = jury_stimuli(JURY_RATING_MIN, JURY_RATING_MAX)
    answer_prefix, rating_token_seqs = select_answer_prefix(
        model, tokenizer, stimuli[0].prompt, device, JURY_RATING_MIN, JURY_RATING_MAX
    )

    ratings = {
        stim.condition: expected_rating(
            model, tokenizer, stim.prompt, rating_token_seqs, device, answer_prefix
        )
        for stim in stimuli
    }
    spread = check_discrimination(ratings, f"jury/{model_key}", strict=strict)

    rows = [
        {
            "model": model_key,
            "task": "jury",
            "condition": stim.condition,
            "sequence": "+".join(stim.sequence),
            "experiment": stim.experiment,
            "n_cases": stim.n_cases,
            "expected_rating": ratings[stim.condition],
            "discrimination_range": spread,
        }
        for stim in stimuli
    ]
    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"jury_{model_key}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

    print(f"\n{'contrast':30s} {'model':>9s}")
    for label, cond_a, cond_b in JURY_CONTRASTS + JURY_CONTROL_CONTRASTS:
        print(f"{label:30s} {ratings[cond_a] - ratings[cond_b]:+9.3f}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEBUG_MODEL, choices=list(MODEL_REPO_IDS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--task", default="similarity", choices=["similarity", "jury"])
    parser.add_argument(
        "--allow-low-discrimination", action="store_true",
        help="Downgrade the discrimination check from an error to a warning.",
    )
    args = parser.parse_args()
    if args.task == "jury":
        run_jury(args.model, args.device, strict=not args.allow_low_discrimination)
    else:
        run_behavioral(args.model, args.device)

"""Step 4: activation patching for causal localization, plus
Step 5: direction transplant (inducing the effect where it's absent).

Both operate on the residual stream at the final token position only
(the position the rating is read from) via forward hooks on
`model.model.layers[l]`. Full token-position sweeps (country-name tokens,
comparative phrase) are noted as future work in MECH_INTERP_PLAN.md Step 4
but not implemented here to keep the causal claim simple: does the layer-l
residual, at the position that matters for the answer, carry the asymmetry.

Handles both single- and two-token rating models. The hook edits a real
residual and lets the rest of the network run to completion (unlike
logit_lens.py, this never truncates depth), so for two-token ratings a
second, hooked forward pass with the first digit appended gives a fully
correct teacher-forced continuation probability -- no approximation needed
here, just care that the hook targets the original final-prompt-token
position (not simply "the last position"), which shifts once a token is
appended.
"""

import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mech_interp.behavioral import (
    combine_first_and_second_probs,
    expected_rating,
    expected_rating_and_first_probs,
    first_token_distribution,
    rating_first_token_ids,
    select_answer_prefix,
)
from mech_interp.config import (
    DEBUG_MODEL,
    MODEL_REPO_IDS,
    RATING_MAX,
    RATING_MIN,
    RESULTS_DIR,
    encode_prompt,
    get_device,
    get_num_layers,
    get_text_decoder,
    load_model_and_tokenizer,
)
from mech_interp.stimuli import PROMINENCE_PAIRS, TEMPLATE_TASK_TYPE, TEMPLATES, build_stimuli


def prompt_ids_for(
    tokenizer, prompt: str, device: str, answer_prefix: str | None = None
) -> torch.Tensor:
    return encode_prompt(tokenizer, prompt, device, answer_prefix)


def get_final_residual(
    model, tokenizer, prompt: str, layer_idx: int, device: str,
    answer_prefix: str | None = None,
) -> torch.Tensor:
    """Residual stream at the final token position, after layer `layer_idx`.

    layer_idx=0 is the embedding output (before any transformer block).
    Prefer get_all_final_residuals() when sweeping layers -- one forward pass
    already computes every layer, so calling this in a loop re-runs the whole
    model once per layer.
    """
    return get_all_final_residuals(model, tokenizer, prompt, device, answer_prefix)[layer_idx]


def get_all_final_residuals(
    model, tokenizer, prompt: str, device: str, answer_prefix: str | None = None
) -> list[torch.Tensor]:
    """Final-token residual at *every* layer, from a single forward pass.

    Indexed so that entry `L` is exactly what ResidualEditHook replaces when it
    is registered on layers[L-1] (entry 0 is the embedding output).

    Captured with forward hooks rather than read off `output_hidden_states`:
    HF appends one extra hidden state *after* the final norm, so
    hidden_states[n_layers] is post-norm while the hook edits the layer's
    pre-norm output. Injecting one into the other differs by ~5e4 on Gemma-3
    and corrupts the deepest layer of every recovery curve. Hooking both
    capture and edit keeps them definitionally in sync across architectures.
    """
    layers = get_text_decoder(model).layers
    captured: dict[int, torch.Tensor] = {}

    def make_capture(idx: int):
        def capture(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx + 1] = hidden[0, -1, :].clone()
        return capture

    handles = [layers[i].register_forward_hook(make_capture(i)) for i in range(len(layers))]
    try:
        prompt_ids = prompt_ids_for(tokenizer, prompt, device, answer_prefix)
        with torch.no_grad():
            out = model(prompt_ids, output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()

    captured[0] = out.hidden_states[0][0, -1, :].clone()
    return [captured[i] for i in range(len(layers) + 1)]


class ResidualEditHook:
    """Forward hook on `model.model.layers[layer_idx - 1]` that edits the
    residual stream at `position` (default: last position) after that block runs.

    mode="replace": overwrite with `vector`.
    mode="add": add `vector` (optionally scaled by `alpha`) to the existing residual.

    `position` must be set explicitly (not left at -1) when the hook is reused
    on a sequence one token longer than the one `vector` was captured from --
    e.g. scoring a two-token rating's second digit -- since "last position"
    would otherwise silently shift to the newly appended token.
    """

    def __init__(self, vector: torch.Tensor, mode: str = "replace", alpha: float = 1.0, position: int = -1):
        self.vector = vector
        self.mode = mode
        self.alpha = alpha
        self.position = position

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        # Edit every batch row, not just row 0: the two-token second pass runs a
        # batch of one sequence per 2-token rating, all sharing the same prompt,
        # so patching only row 0 would leave the rest of the rating distribution
        # computed from an *unpatched* model -- which silently reduced the
        # measured effect to ~zero for every model with 2-token ratings
        # (Gemma, Qwen) while leaving single-token models (Llama-3.1) correct.
        if self.mode == "replace":
            hidden[:, self.position, :] = self.vector
        elif self.mode == "add":
            hidden[:, self.position, :] = hidden[:, self.position, :] + self.alpha * self.vector
        else:
            raise ValueError(f"Unknown mode {self.mode!r}")
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden


def run_with_residual_edit(
    model, tokenizer, prompt: str, layer_idx: int, hook: ResidualEditHook, device: str,
    answer_prefix: str | None = None,
) -> torch.Tensor:
    """Run `prompt` with `hook` applied after layer `layer_idx` (1-indexed block)."""
    target_layer = get_text_decoder(model).layers[layer_idx - 1]
    handle = target_layer.register_forward_hook(hook)
    try:
        prompt_ids = prompt_ids_for(tokenizer, prompt, device, answer_prefix)
        with torch.no_grad():
            logits = model(prompt_ids).logits[0, -1]
    finally:
        handle.remove()
    return logits


def expected_rating_with_hook(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    hook: ResidualEditHook,
    rating_token_seqs: dict[int, list[int]],
    device: str,
    answer_prefix: str | None = None,
) -> float:
    """E[r] for `prompt` with `hook` applied at `layer_idx`.

    For two-token ratings, re-applies the same hook (at the now-shifted
    original-final-token position) on a second forward pass with the true
    first digit appended, so the second digit's conditional probability also
    reflects the intervention -- this is exact, not an approximation, because
    the hook always lets the network run to full depth.
    """
    return expected_rating_and_first_probs_with_hook(
        model, tokenizer, prompt, layer_idx, hook, rating_token_seqs, device, answer_prefix
    )[0]


def expected_rating_and_first_probs_with_hook(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    hook: "ResidualEditHook",
    rating_token_seqs: dict[int, list[int]],
    device: str,
    answer_prefix: str | None = None,
) -> tuple[float, torch.Tensor]:
    """E[r] under the hook, plus the next-token distribution behind it."""
    first_logits = run_with_residual_edit(
        model, tokenizer, prompt, layer_idx, hook, device, answer_prefix
    )
    first_probs = first_logits.softmax(dim=0)

    two_token_ratings = [r for r, ids in rating_token_seqs.items() if len(ids) == 2]
    second_token_probs: dict[int, float] = {}
    if two_token_ratings:
        prompt_ids = prompt_ids_for(tokenizer, prompt, device, answer_prefix)
        original_last_pos = prompt_ids.shape[1] - 1
        second_pass_hook = ResidualEditHook(hook.vector, mode=hook.mode, alpha=hook.alpha, position=original_last_pos)
        target_layer = get_text_decoder(model).layers[layer_idx - 1]
        first_tokens = torch.tensor(
            [[rating_token_seqs[r][0]] for r in two_token_ratings], device=device
        )
        batch_inputs = torch.cat([prompt_ids.repeat(len(two_token_ratings), 1), first_tokens], dim=1)
        handle = target_layer.register_forward_hook(second_pass_hook)
        try:
            with torch.no_grad():
                second_logits = model(batch_inputs).logits[:, -1, :]
        finally:
            handle.remove()
        second_probs = second_logits.softmax(dim=-1)
        for i, r in enumerate(two_token_ratings):
            second_token_probs[r] = float(second_probs[i, rating_token_seqs[r][1]].item())

    e_r = combine_first_and_second_probs(first_probs, rating_token_seqs, second_token_probs)
    return e_r, first_probs


def intervention_stability(
    baseline_probs: torch.Tensor,
    hooked_probs: torch.Tensor,
    rating_token_seqs: dict[int, list[int]],
) -> dict[str, float]:
    """Did the intervention *move* the answer, or just break the model?

    A large change in E[r] is only evidence about the mechanism if the model is
    still doing the task afterwards. Overwriting a residual with an unrelated
    vector can shift the rating a long way simply by degrading the next-token
    distribution into noise, and nothing in a recovery number distinguishes
    that from a genuine causal effect.

    Two cheap readouts over distributions we already have, so no extra forward
    pass:

      kl        KL(baseline || hooked) across the whole vocabulary. Near zero
                means the intervention was gentle; a large value means the
                output distribution was substantially rewritten, and any
                rating change should be treated as suspect.
      rating_mass_before/after
                Share of next-token probability still sitting on rating tokens.
                A collapse here is the concrete failure: the model has stopped
                answering with a number, so E[r] is being renormalised out of
                whatever is left.

    Adapted from the stability checks in CBMAS (NeurIPS 2025 CogInterp
    workshop), which pairs alpha-sweeps with KL and fluency measures for this
    reason.
    """
    # float32, not float64: MPS has no double support, and the sum is over a
    # ~150k-entry vocabulary where float32 is ample.
    eps = 1e-12
    p = baseline_probs.float().clamp_min(eps)
    q = hooked_probs.float().clamp_min(eps)
    kl = float((p * (p / q).log()).sum().item())

    first_ids = sorted({ids[0] for ids in rating_token_seqs.values()})
    idx = torch.tensor(first_ids, device=baseline_probs.device)
    return {
        "kl": kl,
        "rating_mass_before": float(baseline_probs[idx].sum().item()),
        "rating_mass_after": float(hooked_probs[idx].sum().item()),
    }


def first_token_recovery(
    patched: torch.Tensor, order1: torch.Tensor, order2: torch.Tensor
) -> float:
    """How far the patch moved the first-token distribution from order 2 to order 1.

    1.0 = the patched run's rating distribution matches order 1's; 0.0 = it is
    unchanged from order 2's. L1 distance over the rating first-tokens.

    This exists because E[r]-based recovery is not comparable across models.
    Where ratings 10-20 are two tokens, a deep patch cannot reach the second
    digit -- it is produced one position later, from layer n's output, with no
    layer left to carry the edit -- so E[r] mixes order 1's first token with
    order 2's second and reads ~0 even when the patch has fully succeeded
    (measured on gemma3-4b: E[r] recovery 0.00 while the first-token
    distribution matched order 1 exactly). For single-token models (Llama-3.x)
    this metric and E[r] recovery coincide, which is what makes them
    comparable at all.
    """
    denominator = (order2 - order1).abs().sum().item()
    if denominator < 1e-9:
        return float("nan")
    return 1.0 - (patched - order1).abs().sum().item() / denominator


# --- Step 4: patch clean (order 1) -> corrupted (order 2) --------------------

def run_patching_sweep(model_key: str, device: str | None = None) -> pd.DataFrame:
    device = get_device(device)
    model, tokenizer = load_model_and_tokenizer(model_key, device)

    stimuli = build_stimuli(RATING_MIN, RATING_MAX)
    answer_prefix, rating_token_seqs = select_answer_prefix(
        model, tokenizer, stimuli[0].prompt, device
    )
    n_layers = get_num_layers(model)
    first_ids = rating_first_token_ids(rating_token_seqs)

    by_key: dict[tuple, dict[int, str]] = {}
    for stim in stimuli:
        by_key.setdefault((stim.pair, stim.template_name, stim.is_control), {})[stim.order] = stim.prompt

    # Validate the mechanism once up front, on a real pair, before spending
    # ~20 GPU-minutes producing numbers whose meaning depends on it.
    first_orders = next(iter(by_key.values()))
    assert_patch_reproduces_order1(
        model, tokenizer, first_orders[1], first_orders[2], n_layers, device, answer_prefix
    )

    rows = []
    for (pair, template, is_control), orders in by_key.items():
        prompt_1, prompt_2 = orders[1], orders[2]

        e_r_1, probs_1 = expected_rating_and_first_probs(
            model, tokenizer, prompt_1, rating_token_seqs, device, answer_prefix
        )
        e_r_2, probs_2 = expected_rating_and_first_probs(
            model, tokenizer, prompt_2, rating_token_seqs, device, answer_prefix
        )
        dist_1 = first_token_distribution(probs_1, first_ids)
        dist_2 = first_token_distribution(probs_2, first_ids)

        # One pass gives every layer's residual; capturing per-layer inside the
        # loop would re-run the whole model n_layers times for the same values.
        clean_resids = get_all_final_residuals(
            model, tokenizer, prompt_1, device, answer_prefix
        )

        for layer_idx in range(1, n_layers + 1):
            hook = ResidualEditHook(clean_resids[layer_idx], mode="replace")
            e_r_patched, probs_patched = expected_rating_and_first_probs_with_hook(
                model, tokenizer, prompt_2, layer_idx, hook, rating_token_seqs, device, answer_prefix
            )

            denom = e_r_1 - e_r_2
            recovery = (e_r_patched - e_r_2) / denom if abs(denom) > 1e-6 else float("nan")
            recovery_ft = first_token_recovery(
                first_token_distribution(probs_patched, first_ids), dist_1, dist_2
            )

            rows.append(
                {
                    "model": model_key,
                    "pair": f"{pair[0]}-{pair[1]}",
                    "is_control": is_control,
                    "template": template,
                    "task_type": TEMPLATE_TASK_TYPE[template],
                    "layer": layer_idx,
                    "layer_frac": layer_idx / n_layers,
                    "e_r_order1": e_r_1,
                    "e_r_order2": e_r_2,
                    "e_r_patched": e_r_patched,
                    "recovery": recovery,
                    "recovery_first_token": recovery_ft,
                }
            )

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"patching_{model_key}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    return df


def assert_patch_reproduces_order1(
    model, tokenizer, prompt_1: str, prompt_2: str, n_layers: int, device: str,
    answer_prefix: str | None = None, tolerance: float = 1e-2,
) -> None:
    """Patching the final layer must make prompt 2 produce prompt 1's *logits*.

    This is the architecture-independent invariant: replacing the last layer's
    residual at the read-out position leaves nothing downstream to undo it, so
    the next-token distribution must become order 1's exactly. It catches a
    hook editing the wrong batch rows, a residual captured at the wrong
    normalization point, and layer off-by-ones.

    Deliberately stated over logits rather than E[r]. For models whose ratings
    10-20 are two tokens, a final-layer patch *cannot* change the second
    digit's probability -- that token sits one position later and reads layer
    n's output, which was computed from layer n-1 before the edit, and there is
    no further layer to propagate it. So E[r] after a final-layer patch mixes
    order 1's first token with order 2's second token, and recovery is
    legitimately below 1.0. An E[r]-based check would flag those models as
    broken when they are working correctly.
    """
    resids = get_all_final_residuals(model, tokenizer, prompt_1, device, answer_prefix)
    hook = ResidualEditHook(resids[n_layers], mode="replace")
    patched = run_with_residual_edit(
        model, tokenizer, prompt_2, n_layers, hook, device, answer_prefix
    )
    ids_1 = prompt_ids_for(tokenizer, prompt_1, device, answer_prefix)
    with torch.no_grad():
        reference = model(ids_1).logits[0, -1]

    max_diff = (patched.float() - reference.float()).abs().max().item()
    if max_diff > tolerance:
        raise RuntimeError(
            f"Final-layer patch did not reproduce order 1's logits "
            f"(max |diff| = {max_diff:.4f} > {tolerance}). The patch is not taking effect as "
            "intended -- check that ResidualEditHook edits every batch row and that the captured "
            "residual matches the hooked layer's output (pre- vs post-final-norm)."
        )
    print(f"Patch validity check passed: final-layer patch reproduces order 1 "
          f"(max |dlogit| = {max_diff:.2e}).")


# --- Step 5: direction transplant --------------------------------------------

def extract_direction(
    model, tokenizer, layer_idx: int, template: str, device: str,
    answer_prefix: str | None = None,
) -> torch.Tensor:
    """v = mean_pairs(resid[order=1] - resid[order=2]) at `layer_idx`, for `template`."""
    diffs = []
    for pair in PROMINENCE_PAIRS:
        x, y = pair
        prompt_1 = TEMPLATES[template].format(x=x, y=y, rmin=RATING_MIN, rmax=RATING_MAX)
        prompt_2 = TEMPLATES[template].format(x=y, y=x, rmin=RATING_MIN, rmax=RATING_MAX)
        r1 = get_final_residual(model, tokenizer, prompt_1, layer_idx, device, answer_prefix)
        r2 = get_final_residual(model, tokenizer, prompt_2, layer_idx, device, answer_prefix)
        diffs.append(r1 - r2)
    return torch.stack(diffs).mean(dim=0)


def run_transplant_sweep(
    model_key: str,
    source_template: str,
    target_template: str,
    layer_idx: int,
    alphas: list[float],
    device: str | None = None,
    n_random_controls: int = 3,
) -> pd.DataFrame:
    """Transplant the order-asymmetry direction learned on `source_template`
    into runs of `target_template` (expected to show ~0 effect there), and
    check whether Delta becomes nonzero -- with a random-direction control.
    """
    device = get_device(device)
    model, tokenizer = load_model_and_tokenizer(model_key, device)

    sample_prompt = TEMPLATES[source_template].format(
        x=PROMINENCE_PAIRS[0][0], y=PROMINENCE_PAIRS[0][1], rmin=RATING_MIN, rmax=RATING_MAX
    )
    answer_prefix, rating_token_seqs = select_answer_prefix(
        model, tokenizer, sample_prompt, device
    )

    direction = extract_direction(
        model, tokenizer, layer_idx, source_template, device, answer_prefix
    )
    direction_norm = direction.norm().item()

    rng = np.random.default_rng(0)
    random_dirs = []
    for _ in range(n_random_controls):
        r = torch.randn_like(direction)
        r = r / r.norm() * direction_norm
        random_dirs.append(r)

    rows = []
    for pair in PROMINENCE_PAIRS:
        x, y = pair
        prompt_1 = TEMPLATES[target_template].format(x=x, y=y, rmin=RATING_MIN, rmax=RATING_MAX)
        prompt_2 = TEMPLATES[target_template].format(x=y, y=x, rmin=RATING_MIN, rmax=RATING_MAX)

        for alpha in alphas:
            hook = ResidualEditHook(direction, mode="add", alpha=alpha)
            e_r_1 = expected_rating_with_hook(
                model, tokenizer, prompt_1, layer_idx, hook, rating_token_seqs, device, answer_prefix
            )
            e_r_2 = expected_rating_with_hook(
                model, tokenizer, prompt_2, layer_idx, hook, rating_token_seqs, device, answer_prefix
            )

            row = {
                "model": model_key,
                "pair": f"{x}-{y}",
                "source_template": source_template,
                "source_task_type": TEMPLATE_TASK_TYPE[source_template],
                "target_template": target_template,
                "target_task_type": TEMPLATE_TASK_TYPE[target_template],
                "layer": layer_idx,
                "alpha": alpha,
                "direction": "learned",
                "delta": e_r_1 - e_r_2,
            }
            rows.append(row)

            for i, rand_dir in enumerate(random_dirs):
                hook_r = ResidualEditHook(rand_dir, mode="add", alpha=alpha)
                e_r_1r = expected_rating_with_hook(
                    model, tokenizer, prompt_1, layer_idx, hook_r, rating_token_seqs, device,
                    answer_prefix,
                )
                e_r_2r = expected_rating_with_hook(
                    model, tokenizer, prompt_2, layer_idx, hook_r, rating_token_seqs, device,
                    answer_prefix,
                )
                rows.append(
                    {
                        "model": model_key,
                        "pair": f"{x}-{y}",
                        "source_template": source_template,
                        "source_task_type": TEMPLATE_TASK_TYPE[source_template],
                        "target_template": target_template,
                        "target_task_type": TEMPLATE_TASK_TYPE[target_template],
                        "layer": layer_idx,
                        "alpha": alpha,
                        "direction": f"random_{i}",
                        "delta": e_r_1r - e_r_2r,
                    }
                )

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"transplant_{model_key}_L{layer_idx}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEBUG_MODEL, choices=list(MODEL_REPO_IDS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--mode", choices=["patch", "transplant"], default="patch")
    parser.add_argument("--layer", type=int, default=None, help="required for --mode transplant")
    # Defaults target the headline result: does a direction learned on the
    # similarity task transplant into (and flip sign in) the difference task,
    # matching Tversky & Gati's human s/d double dissociation?
    parser.add_argument("--source-template", default="similar_and")
    parser.add_argument("--target-template", default="different_and")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0])
    args = parser.parse_args()

    if args.mode == "patch":
        run_patching_sweep(args.model, args.device)
    else:
        if args.layer is None:
            raise SystemExit("--layer is required for --mode transplant")
        run_transplant_sweep(
            args.model,
            args.source_template,
            args.target_template,
            args.layer,
            args.alphas,
            args.device,
        )

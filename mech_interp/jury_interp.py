"""Logit lens and activation patching for the jury task (stimuli_jury.py).

Scoped deliberately. The jury conditions do not all admit the same tools:

  logit lens   -- runs on every condition. E[r] is read at the final position,
                  which exists regardless of how long the prompt is.
  patching     -- runs on SP_WD <-> WD_SP only. Those two prompts contain the
                  same text in opposite order and tokenize to the same length
                  (198 tokens under every tokenizer we run), so position i means
                  the same thing in both and a residual copied from one into the
                  other is a controlled intervention.

The contrast we most care about -- the defence sign reversal, where the same
defence text moves confidence in guilt one way after a strong prosecution and
the other way after a weak one -- is NOT patchable this way. SP_WD and SP differ
by a whole evidence block, so there is no aligned position to copy. That
contrast needs the steering/transplant approach in steering_probe.py, and is out
of scope here rather than silently approximated.

Two-token ratings are handled by the shared machinery: Gemma and Qwen split
10-20 into two tokens, so Delta(L) for them reflects the first-digit decision
only. See logit_lens.py's module docstring.
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch

from mech_interp.behavioral import (
    check_discrimination,
    expected_rating,
    expected_rating_and_first_probs,
    select_answer_prefix,
)
from mech_interp.config import (
    DEBUG_MODEL,
    MODEL_REPO_IDS,
    RESULTS_DIR,
    get_device,
    get_num_layers,
    load_model_and_tokenizer,
)
from mech_interp.logit_lens import logit_lens_expected_ratings
from mech_interp.patching import (
    ResidualEditHook,
    expected_rating_and_first_probs_with_hook,
    first_token_recovery,
    get_all_final_residuals,
    intervention_stability,
    prompt_ids_for,
)
from mech_interp.stimuli_jury import (
    CONTRASTS,
    CONTROL_CONTRASTS,
    JURY_RATING_MAX,
    JURY_RATING_MIN,
    build_stimuli,
)

# The only positionally-aligned pair in the stimulus set: same text, reordered.
PATCHABLE_CONTRAST = ("recency_order", "SP_WD", "WD_SP")


def run_logit_lens(model_key: str, device: str | None = None) -> pd.DataFrame:
    """E[r] per condition per layer, plus every contrast resolved by layer."""
    device = get_device(device)
    model, tokenizer = load_model_and_tokenizer(model_key, device)
    stimuli = build_stimuli(JURY_RATING_MIN, JURY_RATING_MAX)
    prefix, seqs = select_answer_prefix(
        model, tokenizer, stimuli[0].prompt, device, JURY_RATING_MIN, JURY_RATING_MAX
    )
    n_layers = get_num_layers(model)

    per_layer: dict[str, dict[int, float]] = {}
    for stim in stimuli:
        per_layer[stim.condition] = logit_lens_expected_ratings(
            model, tokenizer, stim.prompt, seqs, device, prefix
        )

    final_layer = max(next(iter(per_layer.values())))
    check_discrimination(
        {c: v[final_layer] for c, v in per_layer.items()}, f"jury-lens/{model_key}"
    )

    rows = []
    for condition, layers in per_layer.items():
        for layer, e_r in layers.items():
            rows.append({
                "model": model_key, "condition": condition,
                "layer": layer, "n_layers": n_layers, "expected_rating": e_r,
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"jury_logit_lens_{model_key}.csv", index=False)

    contrast_rows = []
    for label, a, b in CONTRASTS + CONTROL_CONTRASTS:
        for layer in per_layer[a]:
            contrast_rows.append({
                "model": model_key, "contrast": label, "layer": layer,
                "n_layers": n_layers, "value": per_layer[a][layer] - per_layer[b][layer],
            })
    cdf = pd.DataFrame(contrast_rows)
    out = RESULTS_DIR / f"jury_logit_lens_contrasts_{model_key}.csv"
    cdf.to_csv(out, index=False)
    print(f"Saved layer curves to {out}")

    print(f"\nContrast at final layer ({final_layer}):")
    for label, _, _ in CONTRASTS + CONTROL_CONTRASTS:
        v = cdf[(cdf["contrast"] == label) & (cdf["layer"] == final_layer)]["value"].iloc[0]
        print(f"  {label:30s} {v:+8.3f}")
    return cdf


def run_patching(model_key: str, device: str | None = None) -> pd.DataFrame:
    """Patch WD_SP's residual with SP_WD's, layer by layer, at the readout.

    Recovery is (patched - source) / (target - source): 0 means the patch did
    nothing, 1 means it fully moved the model to the other order's answer. Both
    prompts are the same length, so the final position is the same position.
    """
    device = get_device(device)
    model, tokenizer = load_model_and_tokenizer(model_key, device)
    stimuli = {s.condition: s.prompt for s in build_stimuli(JURY_RATING_MIN, JURY_RATING_MAX)}
    label, target_cond, source_cond = PATCHABLE_CONTRAST
    target, source = stimuli[target_cond], stimuli[source_cond]

    prefix, seqs = select_answer_prefix(
        model, tokenizer, target, device, JURY_RATING_MIN, JURY_RATING_MAX
    )

    n_target = prompt_ids_for(tokenizer, target, device, prefix).shape[1]
    n_source = prompt_ids_for(tokenizer, source, device, prefix).shape[1]
    if n_target != n_source:
        raise RuntimeError(
            f"{model_key}: {target_cond} is {n_target} tokens but {source_cond} is "
            f"{n_source}. Patching assumes the two prompts align position-for-position; "
            f"they do not under this tokenizer, so the result would be meaningless."
        )

    e_r_target, probs_target = expected_rating_and_first_probs(
        model, tokenizer, target, seqs, device, prefix
    )
    e_r_source, probs_source = expected_rating_and_first_probs(
        model, tokenizer, source, seqs, device, prefix
    )
    denom = e_r_target - e_r_source
    if abs(denom) < 1e-6:
        raise RuntimeError(
            f"{model_key}: the two orders give the same E[r] ({e_r_target:.4f}), so "
            f"there is no order effect to recover and recovery is undefined."
        )

    target_residuals = get_all_final_residuals(model, tokenizer, target, device, prefix)
    n_layers = get_num_layers(model)

    rows = []
    for layer in range(1, n_layers + 1):
        hook = ResidualEditHook(target_residuals[layer], mode="replace")
        patched, probs_patched = expected_rating_and_first_probs_with_hook(
            model, tokenizer, source, layer, hook, seqs, device, prefix
        )
        stability = intervention_stability(probs_source, probs_patched, seqs)
        rows.append({
            "model": model_key, "contrast": label, "layer": layer, "n_layers": n_layers,
            "e_r_source": e_r_source, "e_r_target": e_r_target,
            "e_r_patched": patched, "recovery": (patched - e_r_source) / denom,
            # E[r] recovery is not comparable across tokenizers: where 10-20 are
            # two tokens a deep patch cannot reach the second digit, so E[r]
            # mixes the patched first token with an unpatched second and reads
            # ~0 even when the patch fully succeeded. first_token_recovery is
            # the tokenizer-independent metric; report both and trust the
            # latter for cross-model claims.
            "first_token_recovery": first_token_recovery(
                probs_patched, probs_target, probs_source
            ),
            **stability,
        })

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / f"jury_patching_{model_key}.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} layers to {out}")

    print(f"\n{source_cond} E[r]={e_r_source:.3f} -> {target_cond} E[r]={e_r_target:.3f} "
          f"(order effect {denom:+.3f})")
    best = df.loc[df["recovery"].idxmax()]
    best_ft = df.loc[df["first_token_recovery"].idxmax()]
    print(f"Peak E[r] recovery      {best['recovery']:.3f} at layer {int(best['layer'])}/{n_layers}")
    print(f"Peak first-tok recovery {best_ft['first_token_recovery']:.3f} at layer "
          f"{int(best_ft['layer'])}/{n_layers}  <- use this across models")
    print(f"Stability at that layer: KL={best_ft['kl']:.4f}, rating mass "
          f"{best_ft['rating_mass_before']:.3f} -> {best_ft['rating_mass_after']:.3f}")
    print(f"\n{'layer':>6s}{'E[r]rec':>10s}{'firstTok':>10s}{'KL':>9s}{'ratingMass':>12s}")
    for _, r in df.iterrows():
        if int(r["layer"]) % max(1, n_layers // 8) == 0:
            print(f"{int(r['layer']):6d}{r['recovery']:+10.3f}{r['first_token_recovery']:+10.3f}"
                  f"{r['kl']:9.4f}{r['rating_mass_after']:12.3f}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEBUG_MODEL, choices=list(MODEL_REPO_IDS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--step", default="logit_lens", choices=["logit_lens", "patching"])
    args = parser.parse_args()
    if args.step == "patching":
        run_patching(args.model, args.device)
    else:
        run_logit_lens(args.model, args.device)

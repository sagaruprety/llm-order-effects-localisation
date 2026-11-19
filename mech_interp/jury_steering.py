"""Steering the jury defence sign reversal by transplanting a prosecution-strength
direction.

The sign reversal -- the same weak defence raises confidence in guilt after a
strong prosecution (SP_WD > SP) but lowers it after a weak one (WP_WD < WP) --
is the jury paradigm's headline effect and the one patching cannot reach: SP_WD
and SP differ by a whole evidence block, so there is no aligned position to copy
(jury_interp.py). Steering supplies the causal test patching cannot.

The design is a *transplant*, not a within-condition patch, and deliberately so:

    v = resid(SP) - resid(WP)      at layer L, the readout position

is extracted from the prosecution-*only* conditions, where no defence is
present. We then add it to the weak-prosecution-plus-defence condition:

    WP_WD  + alpha * v   -->  does the rating move toward SP_WD?

If a direction carrying "the prosecution was strong", learned where no defence
exists, converts a defence-lowers-guilt context into a defence-raises-guilt one,
that is causal evidence the reversal is gated by a transferable prosecution-
strength signal in the residual stream -- not an artefact of the two prompts
differing in length, which is exactly what blocks the patching test.

Rigour, since the jury paradigm has a SINGLE scenario (one warehouse case) and
therefore no population of items to generalise across or leave out -- this is an
existence demonstration on one case, and the controls below carry the weight the
sample size cannot:

  identity   alpha=0 must reproduce the unhooked run, or the hook is misapplied.
  random     norm-matched random directions must NOT reproduce the target, or any
             perturbation of this size would -- the direction is doing nothing
             specific.
  neutral    the same v applied to WP_NE (neutral item in place of defence): if v
             merely adds "more guilt" uniformly it moves this as much as WP_WD;
             the defence-gating reading requires the defence condition to move
             toward its target more than the neutral one moves toward its own.
  stability  KL and rating-token mass, so a rating that moved because the output
             distribution collapsed is not counted as recovery.

Writes results/jury_steering_<model>_L<layer>.csv.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from mech_interp.behavioral import (
    expected_rating_and_first_probs,
    first_token_distribution,
    rating_first_token_ids,
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
from mech_interp.patching import (
    ResidualEditHook,
    expected_rating_and_first_probs_with_hook,
    get_all_final_residuals,
    first_token_recovery,
    intervention_stability,
)
from mech_interp.stimuli_jury import JURY_RATING_MAX, JURY_RATING_MIN, build_prompt

# alpha=0 first: the identity check every other row depends on. alpha is in
# multiples of ||v||, so alpha=1 adds one natural length of the direction.
DEFAULT_ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0]
IDENTITY_TOLERANCE = 1e-3

# Conditions we need residuals and baselines for. Keyed by condition label ->
# evidence sequence, matching stimuli_jury.CONDITIONS.
_SEQUENCES = {
    "SP": ("SP",), "WP": ("WP",),
    "SP_WD": ("SP", "WD"), "WP_WD": ("WP", "WD"),
    "SP_NE": ("SP", "NE"), "WP_NE": ("WP", "NE"),
}

# Each transplant arm: (name, steer_condition, target_condition, direction_sign).
# The direction is always v = resid(SP) - resid(WP); the sign flips it for the
# symmetric downward test.
_ARMS = [
    ("defence_flip_up", "WP_WD", "SP_WD", +1.0),
    ("defence_flip_down", "SP_WD", "WP_WD", -1.0),
    ("neutral_control", "WP_NE", "SP_NE", +1.0),
]


def _prompts() -> dict[str, str]:
    return {c: build_prompt(seq, JURY_RATING_MIN, JURY_RATING_MAX) for c, seq in _SEQUENCES.items()}


def _rescale(vector: torch.Tensor, target_norm: float) -> torch.Tensor:
    return vector / vector.norm() * target_norm


# Depths to steer when --layers is not given. The jury effect is legible only
# late (jury logit lens), so the sweep is weighted toward the second half, where
# a transferable direction is most plausible before the readout commits.
DEFAULT_LAYER_FRACS = [0.4, 0.55, 0.7, 0.85]


def _layers_from_fracs(fracs: list[float], n_layers: int) -> list[int]:
    """Fractional depths -> distinct 1-indexed layer numbers."""
    return sorted({max(1, min(n_layers, round(f * n_layers))) for f in fracs})


def run_steering(
    model_key: str, layers: list[int] | None, alphas: list[float], n_random: int = 3,
    device: str | None = None,
) -> pd.DataFrame:
    """Sweep the prosecution-strength transplant over several layers in one model
    load. Every layer's readout residual comes from the same forward pass, so
    adding layers costs only the hooked passes, not re-running the model.

    `layers=None` resolves to deciles of depth after the model is loaded, so the
    default path costs no extra model load.
    """
    device = get_device(device)
    model, tokenizer = load_model_and_tokenizer(model_key, device)
    n_layers = get_num_layers(model)
    if layers is None:
        layers = _layers_from_fracs(DEFAULT_LAYER_FRACS, n_layers)
        print(f"No layers given; sweeping depths {DEFAULT_LAYER_FRACS} -> {layers}")
    bad = [l for l in layers if not 1 <= l <= n_layers]
    if bad:
        raise SystemExit(f"layers {bad} out of range 1..{n_layers} for {model_key}")

    prompts = _prompts()
    prefix, seqs = select_answer_prefix(
        model, tokenizer, prompts["SP_WD"], device, JURY_RATING_MIN, JURY_RATING_MAX
    )
    first_ids = rating_first_token_ids(seqs)

    # One forward pass per condition captures the readout residual at *every*
    # layer, so the whole layer sweep reuses these six passes.
    all_resids = {
        c: get_all_final_residuals(model, tokenizer, p, device, prefix)
        for c, p in prompts.items()
    }

    # Unhooked baselines are layer-independent -- computed once, reused per layer.
    base: dict[str, tuple[float, torch.Tensor]] = {}
    for c, p in prompts.items():
        e_r, probs = expected_rating_and_first_probs(model, tokenizer, p, seqs, device, prefix)
        base[c] = (e_r, probs)
        print(f"  base E[r] {c:6s} = {e_r:+.3f}")

    # Fixed random unit directions, rescaled to each layer's ||v|| inside the loop
    # so the random arm stays magnitude-matched to the learned one at every depth.
    rng = torch.Generator(device="cpu").manual_seed(0)
    sample_vec = all_resids["SP"][layers[0]]
    random_units = [
        _rescale(torch.randn(sample_vec.shape, generator=rng).to(sample_vec), 1.0)
        for _ in range(n_random)
    ]

    rows: list[dict] = []
    identity_failures: list[str] = []

    for layer in layers:
        direction = all_resids["SP"][layer] - all_resids["WP"][layer]
        v_norm = direction.norm().item()
        resid_norm = float(np.mean([all_resids[c][layer].norm().item() for c in _SEQUENCES]))
        print(f"L{layer}: ||v||={v_norm:.3f}  mean ||resid||={resid_norm:.3f}  "
              f"ratio={v_norm / resid_norm:.4f}")

        named_arms = [(name, s, t, sign, "prosecution_strength", direction)
                      for name, s, t, sign in _ARMS]
        # Random controls run on the primary arm only (WP_WD -> SP_WD).
        named_arms += [
            (f"random_{i}", "WP_WD", "SP_WD", +1.0, f"random_{i}", _rescale(u, v_norm))
            for i, u in enumerate(random_units)
        ]

        for arm, steer_cond, target_cond, sign, dirname, vec in named_arms:
            e_r_base, probs_base = base[steer_cond]
            e_r_target, probs_target = base[target_cond]
            base_dist = first_token_distribution(probs_base, first_ids)
            target_dist = first_token_distribution(probs_target, first_ids)
            denom = e_r_target - e_r_base

            for alpha in alphas:
                hook = ResidualEditHook(sign * vec, mode="add", alpha=alpha)
                e_r_steered, probs_steered = expected_rating_and_first_probs_with_hook(
                    model, tokenizer, prompts[steer_cond], layer, hook, seqs, device, prefix
                )
                if alpha == 0.0 and abs(e_r_steered - e_r_base) > IDENTITY_TOLERANCE:
                    identity_failures.append(
                        f"L{layer} {arm}: alpha=0 gave {e_r_steered:.5f}, unhooked {e_r_base:.5f}"
                    )
                steered_dist = first_token_distribution(probs_steered, first_ids)
                stability = intervention_stability(probs_base, probs_steered, seqs)
                rows.append({
                    "model": model_key, "layer": layer, "n_layers": n_layers,
                    "arm": arm, "direction": dirname,
                    "steer_condition": steer_cond, "target_condition": target_cond,
                    "alpha": alpha, "v_norm": v_norm, "resid_norm": resid_norm,
                    "e_r_base": e_r_base, "e_r_target": e_r_target, "e_r_steered": e_r_steered,
                    "recovery": (e_r_steered - e_r_base) / denom if abs(denom) > 1e-6 else float("nan"),
                    "first_token_recovery": first_token_recovery(steered_dist, target_dist, base_dist),
                    **stability,
                })

    if identity_failures:
        raise RuntimeError(
            "alpha=0 did not reproduce the unhooked run -- the hook is misapplied "
            "and every number here is invalid:\n  " + "\n  ".join(identity_failures)
        )

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / f"jury_steering_{model_key}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")
    _summarise(df)
    return df


def _summarise(df: pd.DataFrame) -> None:
    """Per layer, the peak-alpha reading of each arm -- the numbers that decide
    whether the transplant flips the reversal specifically."""
    print("\n" + "=" * 78)
    print("Steering the defence reversal (recovery toward target; 1.0 = fully flipped).")
    print("Per layer, at the largest alpha that keeps rating mass > 0.5:")
    print(f"{'layer':>6}{'arm':>18}{'alpha':>6}{'recov':>7}{'ftRecov':>8}{'KL':>8}{'mass':>7}")
    for layer in sorted(df.layer.unique()):
        healthy = df[(df.layer == layer) & (df.rating_mass_after > 0.5) & (df.alpha > 0)]
        for arm in df.arm.unique():
            sub = healthy[healthy.arm == arm]
            if sub.empty:
                continue
            r = sub.loc[sub.alpha.idxmax()]
            print(f"{layer:6d}{arm:>18}{r.alpha:6.1f}{r.recovery:7.2f}"
                  f"{r.first_token_recovery:8.2f}{r.kl:8.3f}{r.rating_mass_after:7.2f}")
    print("=" * 78)
    print("Read it like this:")
    print("  * defence_flip_up recovery > 0 means injecting prosecution-strength")
    print("    into WP_WD moved it toward SP_WD -- the reversal is causally gated.")
    print("  * random_* recovery ~ defence_flip_up means any perturbation does it.")
    print("  * neutral_control moving as much says v is generic 'more guilt',")
    print("    not defence-specific integration.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEBUG_MODEL, choices=list(MODEL_REPO_IDS))
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="1-indexed layers to steer; default = deciles of depth")
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--n-random", type=int, default=3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if 0.0 not in args.alphas:
        raise SystemExit("alpha=0 is the identity check and must be included.")
    run_steering(args.model, args.layers, args.alphas, args.n_random, args.device)

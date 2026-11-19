"""Jury steering figure: does a prosecution-strength direction flip the reversal,
and is the flip specific?

One panel per model, first-token recovery (the tokenization-robust metric)
against steering strength alpha, at the layer where the defence arm recovers best
without distorting the output distribution (KL < 0.5). Three lines: the defence
transplant (WP_WD -> SP_WD), the neutral-item control (WP_NE -> SP_NE) steered
with the *same* direction, and the mean of the norm-matched random directions.

The reading is the gap structure, not the height: a direction that flips the
defence but not the neutral (Gemma-3-4B) is specific; one that moves both
(Llama-3.1-8B) is a generic prosecution-strength signal; one that beats neither
random nor neutral is inert. Reads mech_interp/results/jury_steering_<model>.csv.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
MODELS = [
    ("llama3.1-8b", "Llama-3.1-8B"),
    ("gemma3-4b", "Gemma-3-4B"),
    ("qwen3-8b", "Qwen3-8B"),
    ("gemma3-12b", "Gemma-3-12B"),
]
KL_CEILING = 0.5


def best_layer(df: pd.DataFrame) -> int:
    """Layer where the defence arm recovers best without wrecking the output."""
    clean = df[(df.arm == "defence_flip_up") & (df.alpha > 0)
               & (df.rating_mass_after > 0.5) & (df.kl < KL_CEILING)]
    if clean.empty:
        clean = df[(df.arm == "defence_flip_up") & (df.alpha > 0)]
    return int(clean.loc[clean.first_token_recovery.idxmax(), "layer"])


def build_figure(out_path: Path) -> None:
    # 2x2 rather than 1x4 so the figure stays legible at single-column width.
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.0), sharey=True, sharex=True)
    axes = axes.ravel()
    for ax, (model_key, label) in zip(axes, MODELS):
        df = pd.read_csv(RESULTS_DIR / f"jury_steering_{model_key}.csv")
        layer = best_layer(df)
        at = df[df.layer == layer]

        defence = at[at.arm == "defence_flip_up"].sort_values("alpha")
        neutral = at[at.arm == "neutral_control"].sort_values("alpha")
        rnd = (at[at.arm.str.startswith("random")]
               .groupby("alpha").first_token_recovery.mean().reset_index())

        ax.axhline(0, color="0.7", lw=0.8)
        ax.axhline(1, color="0.85", lw=0.8, ls=":")
        ax.plot(defence.alpha, defence.first_token_recovery, "-o", ms=3,
                color="#c44e52", label="defence (WP_WD$\\to$SP_WD)")
        ax.plot(neutral.alpha, neutral.first_token_recovery, "-s", ms=3,
                color="#4c72b0", label="neutral control")
        ax.plot(rnd.alpha, rnd.first_token_recovery, "--^", ms=3,
                color="0.5", label="random (mean)")
        # Hollow ring any point where the intervention distorted the output
        # distribution (KL > 0.5): a recovery there is not a clean flip.
        for arm_df in (defence, neutral):
            bad = arm_df[arm_df.kl > KL_CEILING]
            ax.scatter(bad.alpha, bad.first_token_recovery, s=70,
                       facecolors="none", edgecolors="0.2", linewidths=0.9, zorder=5)
        ax.set_title(f"{label} (L{layer})", fontsize=7.5, pad=2)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-1.5, 1.7)

    for ax in axes[2:]:
        ax.set_xlabel(r"steering $\alpha$", fontsize=8)
    fig.supylabel("First-token recovery (toward target)", fontsize=8, x=0.02)
    # Legend goes in the Gemma-3-12B panel: its curves all sit above the zero
    # line, so the lower-left corner is empty and nothing is occluded.
    axes[3].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build_figure(Path(args.out))

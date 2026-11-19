"""Jury activation-patching recovery and KL diagnostics on the recency
contrast (SP_WD <-> WD_SP).

Companion to Table tab:jury-patching and to the jury logit-lens figure
(plot_jury_logit_lens.py): the same contrast, read causally across depth.
Two panels are needed rather than one because recovery alone cannot
distinguish a genuine late localisation from an intervention that did
nothing -- the paper's own point about Gemma-3-12B. The left panel plots
first-token recovery(L); the right plots the KL divergence between the
hooked and unhooked next-token distributions at the same layer. A model
whose recovery rises while its KL stays pinned near zero (Gemma-3-4B,
Gemma-3-12B) is a null result, not a late localisation; a model whose KL
rises together with recovery (Llama-3.1-8B) is a genuine one.

Panels are stacked vertically, sharing the depth axis, so the figure stays
legible at single-column width.

Reads mech_interp/results/jury_patching_<model>.csv and writes a single
two-panel PDF (normalised depth on x, one line/model per panel).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
CONTRAST = "recency_order"

# Same fixed order + colours as plot_jury_logit_lens.py so the two figures
# read consistently against each other and against the patching table.
MODELS = [
    ("llama3.1-8b", "Llama-3.1-8B", "#1f77b4"),
    ("qwen3-8b", "Qwen3-8B", "#d62728"),
    ("gemma3-4b", "Gemma-3-4B", "#2ca02c"),
    ("gemma3-12b", "Gemma-3-12B", "#9467bd"),
]

# Gemma-3-4B's first-token recovery is numerically unstable in the
# mid-network (swings to roughly -18) precisely because nothing is
# happening there -- the same near-zero-denominator instability the paper
# reports for the E[r] metric elsewhere, here showing up in first-token
# recovery because the first-token distributions are themselves nearly
# degenerate for this model. Clipping the axis does not hide this: the KL
# panel alongside it shows the same layers sitting at floating-point noise,
# which is the diagnostic the paper argues for.
RECOVERY_YLIM = (-1.05, 1.15)


def load_curve(model_key: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"jury_patching_{model_key}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No jury patching data at {path}")
    df = pd.read_csv(path)
    curve = df[df["contrast"] == CONTRAST].copy()
    if curve.empty:
        raise ValueError(f"No {CONTRAST} rows in {path}")
    n_layers = int(curve["n_layers"].iloc[0])
    curve["depth"] = 100.0 * curve["layer"] / n_layers
    return curve.sort_values("layer")


def build_figure(out_path: Path) -> None:
    fig, (ax_rec, ax_kl) = plt.subplots(2, 1, figsize=(3.3, 3.2), sharex=True)

    ax_rec.axhline(0.5, color="0.6", lw=0.8, ls="--", zorder=1)
    ax_kl.axhline(0.0, color="0.6", lw=0.8, zorder=1)

    for model_key, label, colour in MODELS:
        curve = load_curve(model_key)
        ax_rec.plot(
            curve["depth"], curve["first_token_recovery"],
            color=colour, lw=1.8, zorder=2, label=label,
        )
        ax_kl.plot(
            curve["depth"], curve["kl"],
            color=colour, lw=1.8, zorder=2, label=label,
        )

    ax_rec.set_ylabel("First-token recovery", fontsize=8)
    ax_rec.set_xlim(0, 100)
    ax_rec.set_ylim(*RECOVERY_YLIM)
    ax_rec.legend(fontsize=7, frameon=False, loc="lower right")
    ax_rec.tick_params(labelsize=7)
    ax_rec.spines[["top", "right"]].set_visible(False)

    ax_kl.set_xlabel("Network depth (%)", fontsize=8)
    ax_kl.set_ylabel("KL divergence", fontsize=8)
    ax_kl.set_xlim(0, 100)
    ax_kl.tick_params(labelsize=7)
    ax_kl.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", required=True,
        help="Output PDF path (e.g. the Overleaf repo's plots/jury_patching_recency.pdf)",
    )
    args = parser.parse_args()
    build_figure(Path(args.out))

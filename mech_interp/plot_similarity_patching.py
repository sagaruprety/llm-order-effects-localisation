"""Similarity activation-patching recovery: prominence pairs vs equal-prominence controls.

Companion to Table tab:separation and Table tab:patching. Plots median
first-token recovery against normalised depth, one panel per model, with the
prominence curve (solid) read against the equal-prominence control curve
(dashed) -- the gap between them, not the height of the prominence curve, is
the informative quantity.

Restricted to the similarity task and the two directional templates
(similar_degree, similar_to), matching the design reported in the paper's
Stimuli section. The difference task and the nondirectional similar_and
template are not plotted.

Panels are stacked vertically so the figure fits a single ACL column.

Reads mech_interp/results/patching_<model>.csv and writes a PDF.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"

TASK = "similarity"
DIRECTIONAL_TEMPLATES = ("similar_degree", "similar_to")

# Mid-network band over which the paper reports separation; every curve reaches
# 1.0 at 100% depth structurally, so the endpoints carry no information.
MID_LO, MID_HI = 30.0, 90.0

PROMINENCE_COLOUR = "#1f77b4"
CONTROL_COLOUR = "#ff7f0e"

# Fixed model order + colours so the figure is stable across rebuilds and
# reads in the same order as the separation table's rows.
MODELS = [
    ("llama3.1-8b", "Llama-3.1-8B"),
    ("gemma3-4b", "Gemma-3-4B"),
    ("qwen3-8b", "Qwen3-8B"),
]


def load_curves(model_key: str) -> tuple[pd.Series, pd.Series]:
    """Return (prominence, control) median first-token recovery indexed by depth."""
    path = RESULTS_DIR / f"patching_{model_key}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No patching results at {path}")

    df = pd.read_csv(path)
    df = df[
        (df["task_type"] == TASK)
        & (df["template"].isin(DIRECTIONAL_TEMPLATES))
    ]
    if df.empty:
        raise ValueError(f"No {TASK}/directional rows in {path}")

    df = df.copy()
    df["depth"] = 100.0 * df["layer"] / df["layer"].max()

    def median_by_depth(subset: pd.DataFrame) -> pd.Series:
        return subset.groupby("depth")["recovery_first_token"].median().sort_index()

    return median_by_depth(df[~df["is_control"]]), median_by_depth(df[df["is_control"]])


def build_figure(out_path: Path) -> None:
    fig, axes = plt.subplots(
        len(MODELS), 1, figsize=(3.3, 3.5), sharex=True, sharey=True,
    )

    for ax, (model_key, label) in zip(axes, MODELS):
        prominence, control = load_curves(model_key)

        ax.axvspan(MID_LO, MID_HI, color="0.92", zorder=0)
        ax.axhline(0.5, color="0.6", lw=0.7, ls=":", zorder=1)
        ax.plot(
            prominence.index, prominence.values,
            color=PROMINENCE_COLOUR, lw=1.6, zorder=3, label="Prominence pairs",
        )
        ax.plot(
            control.index, control.values,
            color=CONTROL_COLOUR, lw=1.6, ls="--", zorder=2, label="Control pairs",
        )

        ax.set_title(label, fontsize=8, pad=3)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 100)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    axes[-1].set_xlabel("Network depth (%)", fontsize=8)
    fig.supylabel("Median first-token recovery", fontsize=8, x=0.02)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", required=True,
        help="Output PDF path (e.g. the Overleaf repo's plots/patching_prominence_vs_control.pdf)",
    )
    args = parser.parse_args()
    build_figure(Path(args.out))

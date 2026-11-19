"""Similarity logit lens: order asymmetry Delta(L) read off each layer.

Companion to the patching figure (plot_similarity_patching.py). For every layer
L and every (pair, order, template) the residual stream at the final token
position is pushed through the model's own final norm and unembedding and
restricted to the rating tokens, giving

    Delta(L) = E[r | order 1, L] - E[r | order 2, L]

the rating the model would give using only that layer's representation. Zero
means the two orders are indistinguishable at that depth.

One panel per model, prominence pairs (solid) read against equal-prominence
controls (dashed). As with patching, the informative quantity is the gap
between the two curves, not the height of the prominence curve: a model whose
controls move as much as its prominence pairs has a generic order sensitivity,
not a prominence-specific one.

Restricted to the similarity task and the two directional templates
(similar_degree, similar_to), matching the design reported in the paper's
Stimuli section.

Reads mech_interp/results/logit_lens_<model>.csv and writes a PDF.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"

TASK = "similarity"
DIRECTIONAL_TEMPLATES = ("similar_degree", "similar_to")

PROMINENCE_COLOUR = "#1f77b4"
CONTROL_COLOUR = "#ff7f0e"

# Fixed model order so the figure is stable across rebuilds and reads in the
# same order as the behavioural and patching tables.
MODELS = [
    ("llama3.1-8b", "Llama-3.1-8B"),
    ("gemma3-4b", "Gemma-3-4B"),
    ("qwen3-8b", "Qwen3-8B"),
]


def load_delta(model_key: str) -> tuple[pd.Series, pd.Series]:
    """Return (prominence, control) mean Delta(L) indexed by normalised depth."""
    path = RESULTS_DIR / f"logit_lens_{model_key}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No logit lens results at {path}")

    df = pd.read_csv(path)
    df = df[
        (df["task_type"] == TASK)
        & (df["template"].isin(DIRECTIONAL_TEMPLATES))
    ]
    if df.empty:
        raise ValueError(f"No {TASK}/directional rows in {path}")

    # Delta is a within-(pair, template, layer) contrast between the two orders,
    # so pivot the orders into columns before differencing. Averaging the two
    # orders separately first would silently tolerate a missing order.
    wide = df.pivot_table(
        index=["pair", "is_control", "template", "layer", "layer_frac"],
        columns="order",
        values="expected_rating",
    ).reset_index()
    if 1 not in wide.columns or 2 not in wide.columns:
        raise ValueError(f"Both orders are required, got {list(wide.columns)}")
    wide["delta"] = wide[1] - wide[2]

    wide["depth"] = 100.0 * wide["layer_frac"]

    def mean_by_depth(subset: pd.DataFrame) -> pd.Series:
        return subset.groupby("depth")["delta"].mean().sort_index()

    return mean_by_depth(wide[~wide["is_control"]]), mean_by_depth(wide[wide["is_control"]])


def build_figure(out_path: Path) -> None:
    fig, axes = plt.subplots(
        1, len(MODELS), figsize=(7.0, 2.0), sharey=True,
    )

    for ax, (model_key, label) in zip(axes, MODELS):
        prominence, control = load_delta(model_key)

        ax.axhline(0.0, color="0.85", lw=0.8, zorder=1)
        ax.plot(
            prominence.index, prominence.values,
            color=PROMINENCE_COLOUR, lw=1.4, zorder=3, label="Prominence pairs",
        )
        ax.plot(
            control.index, control.values,
            color=CONTROL_COLOUR, lw=1.4, ls="--", zorder=2, label="Control pairs",
        )

        ax.set_title(label, fontsize=9, pad=4)
        ax.set_xlabel("network depth (%)", fontsize=8)
        ax.set_xlim(0, 100)
        ax.grid(True, color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel(r"$\Delta(L)$  (rating points)", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", required=True,
        help="Output PDF path (e.g. the paper repo's plots/logit_lens_delta_new.pdf)",
    )
    args = parser.parse_args()
    build_figure(Path(args.out))

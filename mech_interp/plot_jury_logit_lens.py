"""Jury logit-lens Delta(L) on the recency contrast (SP_WD <-> WD_SP).

Companion to Table tab:jury-patching: the same contrast, read correlationally
by the logit lens across depth. The final-layer endpoint of each curve
reproduces the behavioural recency effect (the table's "Effect" column), which
is the pipeline correctness check, exactly as in the similarity lens figure.

Reads mech_interp/results/jury_logit_lens_contrasts_<model>.csv and writes a
single-panel PDF (normalised depth on x, rating points on y, one line/model).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
CONTRAST = "recency_order"

# Fixed model order + colours so the figure is stable across rebuilds and
# reads consistently against the patching table's row order.
MODELS = [
    ("llama3.1-8b", "Llama-3.1-8B", "#1f77b4"),
    ("qwen3-8b", "Qwen3-8B", "#d62728"),
    ("gemma3-4b", "Gemma-3-4B", "#2ca02c"),
    ("gemma3-12b", "Gemma-3-12B", "#9467bd"),
]


def load_recency(model_key: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"jury_logit_lens_contrasts_{model_key}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No jury logit-lens contrasts at {path}")
    df = pd.read_csv(path)
    curve = df[df["contrast"] == CONTRAST].dropna(subset=["value"]).copy()
    if curve.empty:
        raise ValueError(f"No {CONTRAST} rows with values in {path}")
    n_layers = int(curve["n_layers"].iloc[0])
    curve["depth"] = 100.0 * curve["layer"] / n_layers
    return curve.sort_values("layer")


def build_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.axhline(0.0, color="0.6", lw=0.8, zorder=1)

    for model_key, label, colour in MODELS:
        curve = load_recency(model_key)
        endpoint = curve["value"].iloc[-1]
        ax.plot(
            curve["depth"], curve["value"],
            color=colour, lw=1.8, zorder=2,
            label=f"{label} ({endpoint:+.2f})",
        )

    ax.set_xlabel("Network depth (%)", fontsize=8)
    ax.set_ylabel(r"Logit-lens $\Delta(L)$ (rating pts)", fontsize=8)
    ax.set_xlim(0, 100)
    ax.tick_params(labelsize=7)
    ax.legend(title="Model (final-layer $\\Delta$)", fontsize=6.5, title_fontsize=7,
              frameon=False, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", required=True,
        help="Output PDF path (e.g. the Overleaf repo's plots/jury_logit_lens_recency.pdf)",
    )
    args = parser.parse_args()
    build_figure(Path(args.out))

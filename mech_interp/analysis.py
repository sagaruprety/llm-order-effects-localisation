"""Stats and plots over results/*.csv produced by behavioral.py / logit_lens.py.

Step 1 deliverable: per-pair asymmetry Delta, tested with Wilcoxon signed-rank,
a sign test, and a label-permutation test; reported per-template and pooled,
split into prominence pairs vs. controls.

Step 2 deliverable: Delta(L) emergence curves (one per model), overlaying
control pairs, from logit_lens.py output.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from mech_interp.config import RESULTS_DIR
from mech_interp.stimuli import TEMPLATE_TASK_TYPE

N_PERMUTATIONS = 10_000
RNG_SEED = 0

# results_human.csv lives at the repo root, not mech_interp/results/ (that dir
# holds only pipeline-generated CSVs). It's Tversky & Gati's ground truth
# (1978, Table 4.2): columns p, q, P, s(pq)/s(qp) (similarity, both orders),
# d(pq)/d(qp) (difference, both orders).
#
# Do NOT reinstate a Delta ~ P regression. P is not a prominence magnitude: it
# is the percentage of a *separate* group of 69 subjects who preferred the
# phrasing "q is similar to p" (p.86). Prominence itself was validated
# ordinally by a different group of 68 and its per-pair magnitudes appear
# nowhere in the table. The regression was also empirically dead on Tversky's
# own data (r=-0.23, p=.32 similarity; r=+0.01, p=.96 difference), P being
# ceiling-restricted (median 94.2, 10 of 21 pairs above 95). Dropped
# deliberately -- the graded prediction is not assessable with these materials.
# Per-pair model-vs-human agreement (delta_vs_human_delta_regression) is the
# valid test and stays.
HUMAN_DATA_PATH = RESULTS_DIR.parent.parent / "results" / "results_human.csv"


def load_behavioral(model_key: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"behavioral_{model_key}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No behavioral results at {path}; run behavioral.py first.")
    return pd.read_csv(path)


def load_human_ground_truth() -> pd.DataFrame:
    """Per-pair human data with Delta computed in the same order1-order2
    convention as pair_deltas: order1 = p (prominent) as subject."""
    df = pd.read_csv(HUMAN_DATA_PATH)
    df["pair"] = df["p"] + "-" + df["q"]
    df["delta_s_human"] = df["s(pq)"] - df["s(qp)"]
    df["delta_d_human"] = df["d(pq)"] - df["d(qp)"]
    return df


def pair_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (pair, template, is_control) with delta = order1 - order2."""
    from mech_interp.stimuli import TEMPLATE_TASK_TYPE

    wide = df.pivot_table(
        index=["pair", "template", "is_control"], columns="order", values="expected_rating"
    ).reset_index()
    wide = wide.rename(columns={1: "e_r_order1", 2: "e_r_order2"})
    wide["delta"] = wide["e_r_order1"] - wide["e_r_order2"]
    wide["task_type"] = wide["template"].map(TEMPLATE_TASK_TYPE)
    return wide


def delta_vs_human_delta_regression(df: pd.DataFrame, human_df: pd.DataFrame) -> pd.DataFrame:
    """Per-pair agreement between model and human asymmetry.

    Better specified than Delta~P for the alignment question: it asks whether
    the model is asymmetric on the *same pairs* humans are, in the same
    direction, with no assumption that asymmetry scales linearly with a
    ceiling-restricted directional-preference percentage.
    """
    deltas = pair_deltas(df)
    deltas = deltas[~deltas["is_control"]]

    human_long = pd.concat([
        human_df[["pair", "delta_s_human"]].rename(columns={"delta_s_human": "delta_human"})
        .assign(task_type="similarity"),
        human_df[["pair", "delta_d_human"]].rename(columns={"delta_d_human": "delta_human"})
        .assign(task_type="difference"),
    ])

    rows = []
    for task_type, group in deltas.groupby("task_type"):
        pooled = group.groupby("pair")["delta"].mean().reset_index()
        merged = pooled.merge(
            human_long[human_long["task_type"] == task_type][["pair", "delta_human"]],
            on="pair", how="inner",
        )
        if len(merged) < 3:
            rows.append({"task_type": task_type, "n": len(merged), "r": np.nan, "p_value": np.nan})
            continue
        result = stats.linregress(merged["delta_human"], merged["delta"])
        rows.append({
            "task_type": task_type,
            "n": len(merged),
            "slope": result.slope,
            "r": result.rvalue,
            "p_value": result.pvalue,
        })
    return pd.DataFrame(rows)


def human_model_comparison_table(df: pd.DataFrame, human_df: pd.DataFrame) -> pd.DataFrame:
    """Per-pair, per-task_type: model Delta (pooled across templates) vs. the
    matching human Delta (delta_s_human for similarity, delta_d_human for
    difference), for the double-dissociation check."""
    deltas = pair_deltas(df)
    deltas = deltas[~deltas["is_control"]]
    pooled = deltas.groupby(["pair", "task_type"])["delta"].mean().reset_index()

    human_long = pd.concat([
        human_df[["pair", "P", "delta_s_human"]].rename(columns={"delta_s_human": "delta_human"})
        .assign(task_type="similarity"),
        human_df[["pair", "P", "delta_d_human"]].rename(columns={"delta_d_human": "delta_human"})
        .assign(task_type="difference"),
    ])
    merged = pooled.merge(human_long, on=["pair", "task_type"], how="inner")
    merged["same_sign_as_human"] = np.sign(merged["delta"]) == np.sign(merged["delta_human"])
    return merged.sort_values(["task_type", "pair"]).reset_index(drop=True)


def permutation_test(deltas: np.ndarray, n_permutations: int = N_PERMUTATIONS) -> float:
    """Two-sided test: shuffle each pair's order label independently, compare |mean delta|."""
    rng = np.random.default_rng(RNG_SEED)
    observed = np.abs(deltas.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(deltas)))
    permuted_means = np.abs((signs * deltas).mean(axis=1))
    return float((permuted_means >= observed).mean())


def summarize_group(deltas: pd.Series) -> dict:
    deltas = deltas.dropna().to_numpy()
    if len(deltas) < 2:
        return {"n": len(deltas), "mean_delta": np.nan, "wilcoxon_p": np.nan,
                "sign_test_p": np.nan, "permutation_p": np.nan}

    n = len(deltas)
    mean_delta = float(deltas.mean())

    try:
        _, wilcoxon_p = stats.wilcoxon(deltas)
    except ValueError:
        wilcoxon_p = np.nan

    n_pos = int((deltas > 0).sum())
    n_nonzero = int((deltas != 0).sum())
    sign_p = stats.binomtest(n_pos, n_nonzero, 0.5).pvalue if n_nonzero else np.nan

    perm_p = permutation_test(deltas)

    return {
        "n": n,
        "mean_delta": mean_delta,
        "wilcoxon_p": wilcoxon_p,
        "sign_test_p": sign_p,
        "permutation_p": perm_p,
    }


def behavioral_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (template, pair_group), plus a per-task_type pooled row.

    Pooling is *within* task_type, never across: Tversky's account predicts
    opposite-signed asymmetries for similarity and difference judgements on the
    same pairs, so averaging the two together makes a real effect cancel into
    an apparent null.
    """
    deltas = pair_deltas(df)
    rows = []
    for (template, is_control), group in deltas.groupby(["template", "is_control"]):
        summary = summarize_group(group["delta"])
        summary.update(
            template=template,
            task_type=TEMPLATE_TASK_TYPE[template],
            pair_group="control" if is_control else "prominence",
        )
        rows.append(summary)

    # Pooled rows average a pair's templates together *before* testing, so the
    # test runs over independent pairs. Treating (pair, template) rows as
    # independent triples the apparent n -- the same paraphrase of the same
    # question to the same model is not a new observation -- and inflates
    # significance accordingly (llama3.1-8b similarity: p=.019 at n=63 vs
    # p=.076 at n=21). Per-template rows above are already one-per-pair and
    # need no such correction.
    for (task_type, is_control), group in deltas.groupby(["task_type", "is_control"]):
        per_pair = group.groupby("pair")["delta"].mean()
        summary = summarize_group(per_pair)
        summary.update(
            template=f"pooled_{task_type}",
            task_type=task_type,
            pair_group="control" if is_control else "prominence",
        )
        rows.append(summary)

    cols = ["template", "task_type", "pair_group", "n", "mean_delta",
            "wilcoxon_p", "sign_test_p", "permutation_p"]
    return (
        pd.DataFrame(rows)[cols]
        .sort_values(["pair_group", "task_type", "template"])
        .reset_index(drop=True)
    )


def human_direction_check(df: pd.DataFrame, prominence_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Which order each model favors, vs. Tversky's human direction (prominent as referent).

    Humans rate similarity higher when the *less prominent* country is the
    subject (order 2 in our convention: order 2 = pair[1] first = less-prominent
    subject). So the human-consistent sign is delta = e_r_order1 - e_r_order2 < 0.
    """
    deltas = pair_deltas(df)
    prom_pair_keys = {f"{p[0]}-{p[1]}" for p in prominence_pairs}
    deltas = deltas[deltas["pair"].isin(prom_pair_keys)].copy()
    deltas["favors_human_direction"] = deltas["delta"] < 0
    return deltas[["pair", "template", "delta", "favors_human_direction"]]


def run_analysis(model_key: str) -> pd.DataFrame:
    from mech_interp.stimuli import PROMINENCE_PAIRS

    df = load_behavioral(model_key)
    summary = behavioral_summary_table(df)
    out_path = RESULTS_DIR / f"behavioral_summary_{model_key}.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved summary to {out_path}")
    print(summary.to_string(index=False))

    direction = human_direction_check(df, PROMINENCE_PAIRS)
    direction_path = RESULTS_DIR / f"human_direction_{model_key}.csv"
    direction.to_csv(direction_path, index=False)
    frac = direction["favors_human_direction"].mean()
    print(f"\nFraction of (pair, template) rows matching human direction: {frac:.2f}")

    if HUMAN_DATA_PATH.exists():
        human_df = load_human_ground_truth()

        comparison = human_model_comparison_table(df, human_df)
        comparison_path = RESULTS_DIR / f"human_model_comparison_{model_key}.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"\nSaved human/model comparison to {comparison_path}")
        for task_type, group in comparison.groupby("task_type"):
            print(f"  {task_type}: {group['same_sign_as_human'].mean():.2f} same-sign-as-human "
                  f"(n={len(group)})")

        human_delta_reg = delta_vs_human_delta_regression(df, human_df)
        human_delta_path = RESULTS_DIR / f"delta_vs_human_delta_{model_key}.csv"
        human_delta_reg.to_csv(human_delta_path, index=False)
        print(f"\nSaved per-pair model-vs-human Delta agreement to {human_delta_path}")
        print(human_delta_reg.to_string(index=False))
    else:
        print(f"\nNo human ground truth at {HUMAN_DATA_PATH}, skipping human comparison.")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    run_analysis(args.model)

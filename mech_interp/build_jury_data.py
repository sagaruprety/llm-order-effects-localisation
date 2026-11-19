"""Build results/results_jury.csv from Trueblood & Busemeyer (2011) Table 3.

Human means are published as probabilities of guilt; the experiment collected
them on the same 21-point (0-20) scale the model pipeline uses, so both are
recorded and the scale conversion is done here rather than in analysis.

Run from the repo root:  python -m mech_interp.build_jury_data
"""

import csv
from pathlib import Path

from mech_interp.stimuli_jury import (
    CONDITIONS,
    CONTRASTS,
    CONTROL_CONDITIONS,
    EXPERIMENT,
    JURY_RATING_MAX as RATING_MAX,
    JURY_RATING_MIN as RATING_MIN,
)

SOURCE = "Trueblood & Busemeyer 2011 Cognitive Science 35, Table 3"

# Mean probability of guilt per condition (Table 3). SP and SP_WD are the
# average of Experiment 1's prosecution-first group and Experiment 2's
# strong-prosecution group, as reported in the paper.
HUMAN_PR_GUILT = {
    "SP": .672,
    "SP_WD": .719,
    "WD": .510,
    "WD_SP": .750,
    "WP": .600,
    "WP_WD": .525,
}

OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "results_jury.csv"

FIELDS = [
    "task", "condition", "experiment", "sequence", "n_cases",
    "pr_guilt_human", "rating_human", "rating_min", "rating_max", "source",
]


def to_rating(p: float) -> float:
    """Probability -> the 0-20 confidence scale the jurors actually used."""
    return round(RATING_MIN + p * (RATING_MAX - RATING_MIN), 3)


def build_rows() -> list[dict]:
    return [
        {
            "task": "jury",
            "condition": condition,
            "experiment": EXPERIMENT[condition],
            "sequence": "+".join(sequence),
            "n_cases": len(sequence),
            "pr_guilt_human": HUMAN_PR_GUILT[condition],
            "rating_human": to_rating(HUMAN_PR_GUILT[condition]),
            "rating_min": RATING_MIN,
            "rating_max": RATING_MAX,
            "source": SOURCE,
        }
        for condition, sequence in CONDITIONS.items()
        if condition not in CONTROL_CONDITIONS
    ]


def report_contrasts() -> None:
    """Print the effects the model pipeline will be scored against."""
    print("Human contrasts (rating units, 0-20):")
    for label, cond_a, cond_b in CONTRASTS:
        delta = to_rating(HUMAN_PR_GUILT[cond_a]) - to_rating(HUMAN_PR_GUILT[cond_b])
        print(f"  {label:30s} {cond_a:6s} - {cond_b:6s} = {delta:+.3f}")

    after_strong = HUMAN_PR_GUILT["SP_WD"] - HUMAN_PR_GUILT["SP"]
    after_weak = HUMAN_PR_GUILT["WP_WD"] - HUMAN_PR_GUILT["WP"]
    assert after_strong > 0 > after_weak, (
        "the weak-defense sign reversal is the headline effect and must hold: "
        f"after_strong={after_strong:+.3f}, after_weak={after_weak:+.3f}"
    )
    print(
        f"\nSign reversal confirmed: the same weak defense moves Pr(guilt) "
        f"{after_strong:+.3f} after a strong prosecution but {after_weak:+.3f} "
        f"after a weak one."
    )


def main() -> None:
    rows = build_rows()
    # Control conditions are ours; McKenzie et al. never ran them, so they are
    # legitimately absent here and must not be fabricated.
    missing = set(CONDITIONS) - CONTROL_CONDITIONS - set(HUMAN_PR_GUILT)
    assert not missing, f"no human value for conditions: {sorted(missing)}"

    OUT_PATH.parent.mkdir(exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report_contrasts()
    print(f"\nwrote {OUT_PATH} ({len(rows)} conditions)")


if __name__ == "__main__":
    main()

"""Jury decision-making stimuli (Trueblood & Busemeyer 2011, Cognitive Science 35).

Order-effect paradigm complementary to the country-similarity task in stimuli.py.
Where that task varies the order of two *entities* within one question, this one
varies the order of two *pieces of evidence* presented across turns, and reads the
same 0-20 confidence rating at the end.

Two manipulations, both from McKenzie et al. (2002) as re-analyzed by Trueblood &
Busemeyer:

  Experiment 1 (order):     strong prosecution + weak defense, in both orders.
  Experiment 2 (strength):  weak defense presented after a strong vs. weak
                            prosecution -- the same defense text raises confidence
                            in guilt in the first case and lowers it in the second.

The second manipulation is the sharper mechanistic target: identical evidence
text, opposite sign of effect, determined only by the preceding context.

All evidence is supplied in the prompt and the case is fictitious, so unlike
survey-based order-effect data (see results/results_qq.csv) there is no world
knowledge for a model to draw on and therefore no background text for us to
author -- the stimulus *is* the context, and it is identical across orders.

Human means for every condition are in results/results_jury.csv.
"""

from dataclasses import dataclass

# This task carries its own scale and does NOT import config.RATING_MIN/MAX.
# Trueblood & Busemeyer p.1537 specifies a 21-point scale, "0 = certain of
# innocence, 10 = guilt as likely as innocence, 20 = certain of guilt" -- the
# midpoint is load-bearing, so the endpoints cannot be shifted. The similarity
# task's scale is 1-20 (Tversky & Gati 1978 p.87), a different range, and
# coupling the two would silently re-render these prompts if that one changes.
JURY_RATING_MIN = 0
JURY_RATING_MAX = 20

# Verbatim from Trueblood & Busemeyer (2011), pp. 1537-1538.
CASE_BACKGROUND = (
    "A warehouse was burglarized in the middle of the night. The next day, the "
    "defendant was arrested after the police received an anonymous tip."
)

STRONG_PROSECUTION = (
    "The prosecution presents the following case:\n"
    "- A witness identified the defendant in a police lineup.\n"
    "- The type of burglary (commercial) is the same as that of the defendant's "
    "past convictions.\n"
    "- The method of entry into the warehouse was unusual and is identical to "
    "that of the defendant's past convictions."
)

# Exp. 2's weak prosecution is the strong one's third bullet alone.
WEAK_PROSECUTION = (
    "The prosecution presents the following case:\n"
    "- The method of entry into the warehouse was unusual and is identical to "
    "that of the defendant's past convictions."
)

WEAK_DEFENSE = (
    "The defense presents the following case:\n"
    "- A friend of the defendant's testified that he was with the defendant at "
    "his house at the time of the burglary."
)

# Our control, not from the source. A second item matched to WEAK_DEFENSE in
# form (one bullet, comparable length) but carrying no probative value about
# guilt. Deliberately *source*-neutral -- attributed to neither side -- because
# a neutral fact presented by the defense would still carry defense framing and
# confound the thing this is meant to isolate.
NEUTRAL_ITEM = (
    "The following additional information was entered into the record:\n"
    "- The court clerk confirmed that the warehouse is located in the eastern "
    "district of the city and was built in 1974."
)

EVIDENCE = {
    "SP": STRONG_PROSECUTION,
    "WP": WEAK_PROSECUTION,
    "WD": WEAK_DEFENSE,
    "NE": NEUTRAL_ITEM,
}

# Each condition is the sequence of evidence seen so far, keyed by the label used
# in Table 3. Both a first-case-only and a both-cases reading exist per order so
# the within-order *change* is measurable, not just the terminal value.
CONDITIONS = {
    "SP": ("SP",),
    "SP_WD": ("SP", "WD"),
    "WD": ("WD",),
    "WD_SP": ("WD", "SP"),
    "WP": ("WP",),
    "WP_WD": ("WP", "WD"),
    # Control conditions (ours, no human counterpart) -- see CONTROL_CONTRASTS.
    "NE": ("NE",),
    "SP_NE": ("SP", "NE"),
    "WP_NE": ("WP", "NE"),
}

# Which experiment each condition belongs to, for grouping in analysis. SP and
# SP_WD appear in both experiments; Trueblood & Busemeyer average them across the
# two, so they are tagged "both". "control" marks conditions we added, which have
# no value in results/results_jury.csv.
EXPERIMENT = {
    "SP": "both", "SP_WD": "both",
    "WD": "order", "WD_SP": "order",
    "WP": "strength", "WP_WD": "strength",
    "NE": "control", "SP_NE": "control", "WP_NE": "control",
}

CONTROL_CONDITIONS = {c for c, e in EXPERIMENT.items() if e == "control"}

INSTRUCTION = (
    "You are a juror in a criminal trial. Read the case below and rate your "
    "confidence that the defendant is guilty."
)

SCALE = (
    "On a scale of {rmin} to {rmax}, where {rmin} means certain of innocence, "
    "{mid} means guilt is as likely as innocence, and {rmax} means certain of "
    "guilt, how confident are you that the defendant is guilty? "
    "Return only a number."
)


@dataclass(frozen=True)
class JuryStimulus:
    condition: str          # key into CONDITIONS, e.g. "SP_WD"
    sequence: tuple[str, ...]  # evidence labels in presentation order
    experiment: str         # "order", "strength", or "both"
    n_cases: int            # how many pieces of evidence have been presented
    prompt: str


def build_prompt(
    sequence: tuple[str, ...],
    rating_min: int = JURY_RATING_MIN,
    rating_max: int = JURY_RATING_MAX,
) -> str:
    """Juror framing + background + evidence so far + the rating question."""
    mid = (rating_min + rating_max) // 2
    parts = [INSTRUCTION, CASE_BACKGROUND]
    parts.extend(EVIDENCE[label] for label in sequence)
    parts.append(SCALE.format(rmin=rating_min, rmax=rating_max, mid=mid))
    return "\n\n".join(parts)


def build_stimuli(
    rating_min: int = JURY_RATING_MIN, rating_max: int = JURY_RATING_MAX
) -> list[JuryStimulus]:
    """One stimulus per condition in Table 3."""
    return [
        JuryStimulus(
            condition=condition,
            sequence=sequence,
            experiment=EXPERIMENT[condition],
            n_cases=len(sequence),
            prompt=build_prompt(sequence, rating_min, rating_max),
        )
        for condition, sequence in CONDITIONS.items()
    ]


# --- Effects of interest -----------------------------------------------------
# Each is (label, condition_a, condition_b); the effect is E[r](a) - E[r](b) on
# the model's rating scale. Human counterparts are in results/results_jury.csv.

ORDER_EFFECT = ("recency_order", "SP_WD", "WD_SP")

# The sign reversal: the same weak defense moves confidence in opposite
# directions depending on the strength of the prosecution that preceded it.
DEFENSE_AFTER_STRONG = ("defense_effect_after_strong", "SP_WD", "SP")
DEFENSE_AFTER_WEAK = ("defense_effect_after_weak", "WP_WD", "WP")

CONTRASTS = [ORDER_EFFECT, DEFENSE_AFTER_STRONG, DEFENSE_AFTER_WEAK]

# Control for the deflationary account of the defense sign reversal: if adding
# *any* second item pulls the rating toward a central prior, then opposite-signed
# defense effects fall out mechanically, with no context-sensitivity involved.
# These substitute a non-probative item for the defense, holding position and
# roughly holding length.
#
# Read them as: if the neutral effects are ~0 while the defense effects are
# opposite-signed, the reversal is real. If the neutral effects reproduce the
# same opposite-signed pattern, it is regression toward the prior.
NEUTRAL_AFTER_STRONG = ("neutral_effect_after_strong", "SP_NE", "SP")
NEUTRAL_AFTER_WEAK = ("neutral_effect_after_weak", "WP_NE", "WP")

CONTROL_CONTRASTS = [NEUTRAL_AFTER_STRONG, NEUTRAL_AFTER_WEAK]

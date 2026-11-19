"""Country pairs, prompt templates, and prompt construction for both orders.

Two task types, each with three phrasing styles:
  - similarity ("similar_to"/"similar_and"/"similar_degree"): ported verbatim
    (question wording only) from the published study's real prompts in
    similarity_effect_single_prompt.py -- NOT the same wording as an earlier
    version of this file, which had invented placeholder templates.
  - difference ("different_to"/"different_and"/"different_degree"): authored
    fresh, parallel construction to the similarity styles, since no
    difference-judgment LLM prompts exist anywhere in this repo. This is what
    lets Delta be compared against results/results_human.csv's d(pq)/d(qp)
    columns (Tversky & Gati's human difference judgments), alongside the
    existing s(pq)/s(qp) similarity comparison.

Not to be confused with this repo's DST/DSA/DSD result files (similarity_effect_
dual_prompt.ipynb): "Dual" there means both orders scored in one combined
prompt/response, a different *elicitation format* for the same similarity
question -- unrelated to the similarity-vs-difference *task type* distinction
here. Naming below avoids the DST/DSA/DSD letters entirely to prevent
re-confusing the two.

The "Return only a number" ending replaces the original study's "Shape:
score: int" JSON instruction: mech_interp reads ratings directly off next-
token logits (see behavioral.py), not parsed JSON, so the question wording is
preserved from the paper but the output-format instruction is adapted to
this measurement method.
"""

from dataclasses import dataclass

# Tversky & Gati pairs, kept identical to the 2024 paper (src/similarity_effect.py)
# for comparability. Each tuple is (prominent, less_prominent).
PROMINENCE_PAIRS = [
    ("U.S.A.", "Mexico"),
    ("U.S.S.R.", "Poland"),
    ("China", "Albania"),
    ("U.S.A.", "Israel"),
    ("Japan", "Philippines"),
    ("U.S.A.", "Canada"),
    ("U.S.S.R.", "Israel"),
    ("England", "Ireland"),
    ("Germany", "Austria"),
    ("U.S.S.R.", "France"),
    ("Belgium", "Luxembourg"),
    ("U.S.A.", "U.S.S.R."),
    ("China", "North Korea"),
    ("India", "Sri Lanka"),
    ("U.S.A.", "France"),
    ("U.S.S.R.", "Cuba"),
    ("England", "Jordan"),
    ("France", "Israel"),
    ("U.S.A.", "Germany"),
    ("U.S.S.R.", "Syria"),
    ("France", "Algeria"),
]

# Roughly-equal prominence: minimal asymmetry predicted, used as a sanity check.
# Equal-prominence pairs: Tversky's account predicts no order asymmetry here,
# so these are what distinguish "the model encodes the prominence asymmetry"
# from "the model commits to whichever entity it read first".
#
# There must be at least ~8. A two-sided Wilcoxon over n=5 has a minimum
# attainable p of 0.0625 no matter how consistent the data -- the control
# condition could never reach significance, making the specificity check
# unfalsifiable rather than merely negative. n=10 floors at p=0.002.
CONTROL_PAIRS = [
    ("Norway", "Sweden"),
    ("Spain", "Portugal"),
    ("Belgium", "Netherlands"),
    ("Argentina", "Chile"),
    ("Kenya", "Tanzania"),
    ("Austria", "Switzerland"),
    ("Denmark", "Finland"),
    ("Peru", "Ecuador"),
    ("Malaysia", "Thailand"),
    ("Ghana", "Senegal"),
]

ALL_PAIRS = PROMINENCE_PAIRS + CONTROL_PAIRS

SIMILARITY_TEMPLATES = {
    "similar_to": (
        "On a scale of {rmin} to {rmax}, where {rmin} means no similarity and "
        "{rmax} means complete similarity, is {x} similar to {y}? Return only a number"
    ),
    "similar_and": (
        "On a scale of {rmin} to {rmax}, where {rmin} means no similarity and "
        "{rmax} means complete similarity, how similar are {x} and {y}? Return only a number"
    ),
    "similar_degree": (
        "On a scale of {rmin} to {rmax}, where {rmin} means no similarity and "
        "{rmax} means complete similarity, assess the degree to which {x} similar "
        "to {y}? Return only a number"
    ),
}

DIFFERENCE_TEMPLATES = {
    "different_to": (
        "On a scale of {rmin} to {rmax}, where {rmin} means no difference and "
        "{rmax} means completely different, is {x} different to {y}? Return only a number"
    ),
    "different_and": (
        "On a scale of {rmin} to {rmax}, where {rmin} means no difference and "
        "{rmax} means completely different, how different are {x} and {y}? Return only a number"
    ),
    "different_degree": (
        "On a scale of {rmin} to {rmax}, where {rmin} means no difference and "
        "{rmax} means completely different, assess the degree to which {x} differs "
        "from {y}? Return only a number"
    ),
}

TEMPLATES = {**SIMILARITY_TEMPLATES, **DIFFERENCE_TEMPLATES}

TEMPLATE_TASK_TYPE = {name: "similarity" for name in SIMILARITY_TEMPLATES}
TEMPLATE_TASK_TYPE.update({name: "difference" for name in DIFFERENCE_TEMPLATES})


@dataclass(frozen=True)
class Stimulus:
    pair: tuple[str, str]  # (prominent, less_prominent), or arbitrary for controls
    is_control: bool
    template_name: str
    task_type: str  # "similarity" or "difference"
    order: int  # 1 = pair[0] first, 2 = pair[1] first
    prompt: str


def build_stimuli(rating_min: int, rating_max: int) -> list[Stimulus]:
    """Construct both-order prompts for every (pair, template) combination."""
    stimuli = []
    for pair in ALL_PAIRS:
        is_control = pair in CONTROL_PAIRS
        x, y = pair
        for template_name, template in TEMPLATES.items():
            for order, (first, second) in enumerate([(x, y), (y, x)], start=1):
                prompt = template.format(
                    x=first, y=second, rmin=rating_min, rmax=rating_max
                )
                stimuli.append(
                    Stimulus(
                        pair=pair,
                        is_control=is_control,
                        template_name=template_name,
                        task_type=TEMPLATE_TASK_TYPE[template_name],
                        order=order,
                        prompt=prompt,
                    )
                )
    return stimuli

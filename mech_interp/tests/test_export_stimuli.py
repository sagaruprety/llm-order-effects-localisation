"""The committed data/ exports must match the stimuli defined in code.

stimuli.py and stimuli_jury.py are authoritative; data/ is a convenience copy.
A convenience copy that silently goes stale is worse than none, because a
reader inspecting data/ would be looking at stimuli the experiments never ran.
This test fails if anyone edits the stimulus modules without re-running

    python mech_interp/export_stimuli.py
"""

import csv
from pathlib import Path

import pytest

from mech_interp import export_stimuli

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

CASES = [
    ("similarity_pairs.csv", export_stimuli.similarity_pairs),
    ("similarity_templates.csv", export_stimuli.similarity_templates),
    ("similarity_prompts.csv", export_stimuli.similarity_prompts),
    ("jury_evidence.csv", export_stimuli.jury_evidence),
    ("jury_conditions.csv", export_stimuli.jury_conditions),
    ("jury_prompts.csv", export_stimuli.jury_prompts),
]


def read_rows(name: str) -> list[dict]:
    with (DATA_DIR / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("name,builder", CASES, ids=[c[0] for c in CASES])
def test_export_matches_source(name, builder):
    expected = builder()
    actual = read_rows(name)

    assert len(actual) == len(expected), (
        f"{name} has {len(actual)} rows, the stimulus modules give "
        f"{len(expected)}. Re-run: python mech_interp/export_stimuli.py"
    )

    # CSV round-trips everything as text, so compare stringified values rather
    # than trying to re-infer bool/int types on the way back in.
    for i, (got, want) in enumerate(zip(actual, expected)):
        want_str = {k: str(v) for k, v in want.items()}
        assert got == want_str, (
            f"{name} row {i} is stale.\n  file:   {got}\n  source: {want_str}\n"
            f"Re-run: python mech_interp/export_stimuli.py"
        )


def test_pair_counts_match_the_paper():
    """21 prominence pairs and 10 equal-prominence controls, as reported."""
    rows = read_rows("similarity_pairs.csv")
    assert sum(r["role"] == "prominence" for r in rows) == 21
    assert sum(r["role"] == "control" for r in rows) == 10


def test_every_similarity_pair_appears_in_both_orders():
    """The order effect is a within-pair contrast, so a missing order would
    silently drop a pair from the analysis rather than raise."""
    rows = read_rows("similarity_prompts.csv")
    seen = {}
    for r in rows:
        seen.setdefault((r["first"], r["second"], r["template"]), set()).add(r["order"])
    missing = {k: v for k, v in seen.items() if v != {"1", "2"}}
    assert not missing, f"These (pair, template) combinations lack both orders: {missing}"


def test_jury_conditions_cover_the_nine_reported():
    rows = read_rows("jury_conditions.csv")
    assert {r["condition"] for r in rows} == {
        "SP", "SP_WD", "WD", "WD_SP", "WP", "WP_WD", "NE", "SP_NE", "WP_NE",
    }
    assert sum(r["is_control"] == "True" for r in rows) == 3

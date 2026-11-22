"""Dump every stimulus to data/ as CSV and JSON, for inspection without reading Python.

The stimuli are *defined* in stimuli.py and stimuli_jury.py, and those modules
remain the authoritative source: the experiment scripts build their prompts from
them directly, never from these exports. This script exists so a reader can see
the 21 prominence pairs, the 10 equal-prominence controls, the jury evidence
blocks and every rendered prompt without running anything.

Regenerate after any change to the stimuli:

    python mech_interp/export_stimuli.py

Writes, relative to the repository root:

    data/similarity_pairs.csv     the country pairs, with the control flag
    data/similarity_templates.csv the six prompt templates
    data/similarity_prompts.csv   every rendered (pair, template, order) prompt
    data/jury_evidence.csv        the four evidence blocks, verbatim
    data/jury_conditions.csv      the nine conditions and their evidence order
    data/jury_prompts.csv         every rendered condition prompt
    data/stimuli.json             all of the above in one nested document
"""

import csv
import json
from pathlib import Path

from mech_interp.config import RATING_MAX, RATING_MIN
from mech_interp.stimuli import (
    ALL_PAIRS,
    CONTROL_PAIRS,
    PROMINENCE_PAIRS,
    TEMPLATE_TASK_TYPE,
    TEMPLATES,
    build_stimuli,
)
from mech_interp.stimuli_jury import (
    CONDITIONS,
    EVIDENCE,
    EXPERIMENT,
    JURY_RATING_MAX,
    JURY_RATING_MIN,
    build_prompt,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
    path = DATA_DIR / name
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.relative_to(DATA_DIR.parent)}  ({len(rows)} rows)")


def similarity_pairs() -> list[dict]:
    """The country pairs. For prominence pairs the first element is the more
    prominent country; for controls the order is arbitrary by construction."""
    return [
        {
            "first": a,
            "second": b,
            "is_control": pair in CONTROL_PAIRS,
            "role": "control" if pair in CONTROL_PAIRS else "prominence",
        }
        for pair in ALL_PAIRS
        for a, b in [pair]
    ]


def similarity_templates() -> list[dict]:
    return [
        {"template": name, "task_type": TEMPLATE_TASK_TYPE[name], "text": text}
        for name, text in TEMPLATES.items()
    ]


def similarity_prompts() -> list[dict]:
    return [
        {
            "first": s.pair[0],
            "second": s.pair[1],
            "is_control": s.is_control,
            "template": s.template_name,
            "task_type": s.task_type,
            "order": s.order,
            "prompt": s.prompt,
        }
        for s in build_stimuli(RATING_MIN, RATING_MAX)
    ]


def jury_evidence() -> list[dict]:
    return [{"label": label, "text": text} for label, text in EVIDENCE.items()]


def jury_conditions() -> list[dict]:
    return [
        {
            "condition": name,
            "sequence": "+".join(sequence),
            "n_blocks": len(sequence),
            "experiment": EXPERIMENT[name],
            "is_control": EXPERIMENT[name] == "control",
        }
        for name, sequence in CONDITIONS.items()
    ]


def jury_prompts() -> list[dict]:
    return [
        {
            "condition": name,
            "sequence": "+".join(sequence),
            "experiment": EXPERIMENT[name],
            "prompt": build_prompt(sequence, JURY_RATING_MIN, JURY_RATING_MAX),
        }
        for name, sequence in CONDITIONS.items()
    ]


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Writing stimulus exports to {DATA_DIR}")

    pairs = similarity_pairs()
    templates = similarity_templates()
    prompts = similarity_prompts()
    evidence = jury_evidence()
    conditions = jury_conditions()
    jprompts = jury_prompts()

    write_csv("similarity_pairs.csv", ["first", "second", "is_control", "role"], pairs)
    write_csv("similarity_templates.csv", ["template", "task_type", "text"], templates)
    write_csv(
        "similarity_prompts.csv",
        ["first", "second", "is_control", "template", "task_type", "order", "prompt"],
        prompts,
    )
    write_csv("jury_evidence.csv", ["label", "text"], evidence)
    write_csv(
        "jury_conditions.csv",
        ["condition", "sequence", "n_blocks", "experiment", "is_control"],
        conditions,
    )
    write_csv("jury_prompts.csv", ["condition", "sequence", "experiment", "prompt"], jprompts)

    bundle = {
        "similarity": {
            "rating_scale": {"min": RATING_MIN, "max": RATING_MAX},
            "n_prominence_pairs": len(PROMINENCE_PAIRS),
            "n_control_pairs": len(CONTROL_PAIRS),
            "pairs": pairs,
            "templates": templates,
            "prompts": prompts,
        },
        "jury": {
            "rating_scale": {"min": JURY_RATING_MIN, "max": JURY_RATING_MAX},
            "evidence": evidence,
            "conditions": conditions,
            "prompts": jprompts,
        },
    }
    json_path = DATA_DIR / "stimuli.json"
    json_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    print(f"  {json_path.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()

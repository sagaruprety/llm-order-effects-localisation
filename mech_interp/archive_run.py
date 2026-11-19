"""Snapshot mech_interp/results/ into a tracked, timestamped run directory.

results/ is gitignored and every sweep overwrites it in place, so a result a
paper claim rests on can silently disappear when the next run starts. This
copies the current results into mech_interp/runs/<label>/ (tracked in git)
together with a manifest recording the commit, host, and library versions the
numbers were produced under.

    python -m mech_interp.archive_run --label 2026-07-20-corrected-templates
"""

import argparse
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from mech_interp.config import RESULTS_DIR

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _library_versions() -> dict[str, str]:
    """Versions that can change results without any code change -- notably
    transformers, whose per-architecture forward/hook internals differ across
    releases."""
    versions = {}
    for name in ("torch", "transformers", "numpy", "pandas", "scipy"):
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = "not installed"
    return versions


def build_manifest(label: str, files: list[Path], note: str | None = None) -> dict:
    return {
        "label": label,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        # These describe the machine running *this script*, which is not
        # necessarily where the CSVs were computed -- results are typically
        # produced on a GPU VM and archived after download. Use --note to
        # record the producing environment; a manifest that silently implies
        # the wrong transformers version is worse than no manifest, since
        # per-architecture forward/hook behaviour changes across releases.
        "archiving_host": platform.node(),
        "archiving_platform": platform.platform(),
        "archiving_libraries": _library_versions(),
        "producing_environment": note or "UNRECORDED -- pass --note next time",
        "files": sorted(f.name for f in files),
    }


def archive_run(label: str | None = None, overwrite: bool = False, note: str | None = None) -> Path:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"No results directory at {RESULTS_DIR}.")

    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs in {RESULTS_DIR}; nothing to archive.")

    label = label or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    dest = RUNS_DIR / label
    if dest.exists() and not overwrite:
        raise FileExistsError(
            f"{dest} already exists. Pass --overwrite to replace it, or pick a new --label. "
            "Refusing to clobber an archived run."
        )

    dest.mkdir(parents=True, exist_ok=True)
    for f in csv_files:
        shutil.copy2(f, dest / f.name)

    manifest = build_manifest(label, csv_files, note)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Archived {len(csv_files)} CSVs to {dest}")
    print(f"  commit={manifest['git_commit'][:12]}  produced_on={manifest['producing_environment']}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default=None, help="Run directory name (default: UTC timestamp).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--note", default=None,
                        help="Where these numbers were actually computed, e.g. 'GCP L4 VM, transformers 5.14.1'.")
    args = parser.parse_args()
    archive_run(args.label, args.overwrite, args.note)

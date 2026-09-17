"""Run every computation used by the current article draft.

The order follows the manuscript.  Arguments are passed to every producer, so
``python generate_all.py --force`` rebuilds the caches used by the corpus
audits.  The final step checks the versioned JSON exports against TeX/main.tex.
"""

from pathlib import Path
import subprocess
import sys
import time


SCRIPT_ROOT = Path(__file__).resolve().parent
ARTICLE_ROOT = SCRIPT_ROOT.parent

# Keep this explicit: adding a producer must be a reviewed change to the
# reproduction pipeline, not something an incidental filename can trigger.
SCRIPTS = [
    "ART-the-corpus.py",
    "ART-a-vocabulary-fixed-by-rule.py",
    "ART-the-system-used-below.py",
    "ART-what-the-two-readings-cost.py",
    "ART-what-the-reading-separates.py",
    "ART-the-order-used-below.py",
    "FIG-the-order-used-below.py",
    "ART-reading-from-the-tonic.py",
    "ART-the-degrees-of-a-key.py",
    "FIG-the-degrees-of-a-key.py",
    "ART-borrowed-degrees.py",
    # Cached key diagnostics consumed by the work- and corpus-level producers.
    "key_audit.py",
    "common_practice_key_audit.py",
    "common_practice_audit.py",
    "ART-the-representation-of-a-work.py",
    "ART-the-two-repertoires.py",
    "FIG-the-two-repertoires.py",
    "ART-the-brightness-contrast.py",
    "ROBUST-p-sensitivity.py",
    "ART-the-p-sensitivity.py",
    "ART-what-separates-them.py",
    "publish_article_outputs.py",
    "record_run_manifest.py",
    "check_article_numbers.py",
]


def revision(repository: Path) -> str:
    """Return a compact, auditable Git revision for a sibling repository."""
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ))
    return commit + ("+dirty" if dirty else "")


def main() -> None:
    print("source revisions", flush=True)
    for name in ("TeX", "jazz-harmony-experiments", "leadsheetanalyser"):
        print(f"  {name:26s} {revision(ARTICLE_ROOT / name)}", flush=True)

    durations = []
    started = time.perf_counter()
    for script_name in SCRIPTS:
        print(f"\n==> {script_name}", flush=True)
        clock = time.perf_counter()
        subprocess.run(
            [sys.executable, str(SCRIPT_ROOT / script_name), *sys.argv[1:]],
            cwd=SCRIPT_ROOT,
            check=True,
        )
        durations.append((time.perf_counter() - clock, script_name))

    total = time.perf_counter() - started
    print(f"\n{'':=<58}")
    print(f"{'timing, slowest first':38s}{'seconds':>10s}{'share':>10s}")
    for seconds, script_name in sorted(durations, reverse=True):
        print(f"{script_name:38s}{seconds:10.1f}{seconds / total:10.1%}")
    print(f"{'total':38s}{total:10.1f}")


if __name__ == "__main__":
    main()

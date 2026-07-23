"""Run every computation used by the current article draft.

Arguments are passed through to every script, so

    python generate_all.py --force

rebuilds the cached audits instead of loading them.  A per-script timing summary
is printed at the end: without it we were guessing at where the minutes went.
"""

from pathlib import Path
import subprocess
import sys
import time

from article_setup import ARTICLE_ROOT


SCRIPT_ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "injectivity_check.py",
    "modal_affinities.py",
    "realbook_family_frequencies.py",
    "kind_set_contrast.py",
    "affinity_figure_variants.py",
    "rooted_chord_diagnostics.py",
    "key_audit.py",
    "funky_dissimilarity_matrix.py",
    "common_practice_audit.py",
    "mixed_corpus_mds.py",
    "robustness_checks.py",
    "messiaen_affinities.py",
]

PASS_THROUGH = sys.argv[1:]

durations = []
started = time.perf_counter()
for script_name in SCRIPTS:
    print(f"\n==> {script_name}", flush=True)
    clock = time.perf_counter()
    subprocess.run(
        [sys.executable, str(SCRIPT_ROOT / script_name), *PASS_THROUGH],
        cwd=ARTICLE_ROOT,
        check=True,
    )
    durations.append((time.perf_counter() - clock, script_name))

total = time.perf_counter() - started
print(f"\n{'':=<58}")
print(f"{'timing, slowest first':38s}{'seconds':>10s}{'share':>10s}")
for seconds, script_name in sorted(durations, reverse=True):
    print(f"{script_name:38s}{seconds:10.1f}{seconds / total:10.1%}")
print(f"{'total':38s}{total:10.1f}")

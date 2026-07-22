"""Shared paths for the reproducible computations of the article."""

from pathlib import Path
import sys


ARTICLE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ARTICLE_ROOT / "leadsheetanalyser"
FIGURE_ROOT = ARTICLE_ROOT / "fig"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def cache_is_fresh(target, *sources) -> bool:
    """True when `target` exists and postdates every source it derives from.

    The audit scripts spend minutes running a key-finding analysis per song and
    then write the result to a CSV.  Nothing about that computation changes
    between runs unless its data or its own code changes, so the CSV is a
    legitimate cache -- the same discipline the distance matrices already
    follow.  The calling script must pass its own path among the sources, or
    editing the analysis would leave a stale CSV in place.

    Pass --force on the command line to ignore any cache.
    """
    if "--force" in sys.argv:
        return False
    target = Path(target)
    if not target.exists():
        return False
    stamp = target.stat().st_mtime
    return all(Path(s).exists() and Path(s).stat().st_mtime <= stamp
               for s in sources)


def output_directory() -> Path:
    """Return --out DIR when provided, otherwise the article's fig directory."""
    if "--out" in sys.argv:
        directory = Path(sys.argv[sys.argv.index("--out") + 1]).resolve()
    else:
        directory = FIGURE_ROOT
    directory.mkdir(parents=True, exist_ok=True)
    return directory

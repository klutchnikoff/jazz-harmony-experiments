"""Shared paths for the reproducible computations of the article.

Expected layout -- the two repositories cloned side by side in any parent
directory, which is all a reader needs:

    <parent>/
        leadsheetanalyser/          # cloned, supplies data/
        jazz-harmony-experiments/   # this repository
        results/                    # created on first run: figures and tables

`results/` deliberately sits outside both repositories.  The figures the
manuscript uses are build products, not sources, and a reader who has only the
two public repositories must be able to produce them without owning the
(private) manuscript repository.

Which `leadsheetanalyser` is imported
------------------------------------
By default, the one pinned in `pyproject.toml` and installed by `pdm install`.
That is the release the article cites, so it is what a reader gets.

Set `LSA_LOCAL=1` to import the sibling source tree instead.  Use it only while
editing the package -- the results then come from an unpublished working tree.
Every run prints which of the two it resolved, because the previous version of
this file injected the source tree unconditionally and nothing said so.
"""

from pathlib import Path
import os
import sys


# <parent>/ : the two repositories' common parent, whatever it is called.
ARTICLE_ROOT = Path(__file__).resolve().parents[1]

PACKAGE_ROOT = ARTICLE_ROOT / "leadsheetanalyser"
DATA_ROOT = PACKAGE_ROOT / "data"

RESULTS_ROOT = ARTICLE_ROOT / "results"
CACHE_ROOT = RESULTS_ROOT / "cache"

USE_LOCAL_PACKAGE = os.environ.get("LSA_LOCAL") == "1"

if USE_LOCAL_PACKAGE and str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _announce_package() -> None:
    """State which leadsheetanalyser this run uses, and where its data is.

    Silence here is what let an unpublished working tree masquerade as the
    pinned release: both resolve, and only the import order decided.
    """
    try:
        import leadsheetanalyser
    except ImportError:
        sys.exit(
            "leadsheetanalyser is not importable.\n"
            "  Install the pinned release:  pdm install\n"
            "  Or use a sibling clone:      LSA_LOCAL=1 pdm run python <script>.py"
        )
    origin = "local source tree" if USE_LOCAL_PACKAGE else "pinned release"
    version = getattr(leadsheetanalyser, "__version__", "unknown")
    print(f"[setup] leadsheetanalyser {version} ({origin})", file=sys.stderr)
    print(f"[setup] data    {DATA_ROOT}", file=sys.stderr)
    print(f"[setup] results {RESULTS_ROOT}", file=sys.stderr)


def require_data(*names) -> None:
    """Fail early, and legibly, when the corpus files are absent.

    `data/` is not distributed with either the repository or the wheel: the
    large files are gitignored and the wheel ships only the package directory.
    Without this check the first script dies on a bare FileNotFoundError deep
    inside a reader, which says nothing about what to do next.
    """
    missing = [n for n in names if not (DATA_ROOT / n).exists()]
    if missing:
        sys.exit(
            f"missing corpus files in {DATA_ROOT}:\n"
            + "".join(f"  - {n}\n" for n in missing)
            + "See leadsheetanalyser/README for how to obtain them."
        )


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
    """Return --out DIR when provided, otherwise the shared results directory."""
    if "--out" in sys.argv:
        directory = Path(sys.argv[sys.argv.index("--out") + 1]).resolve()
    else:
        directory = RESULTS_ROOT
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def cache_directory() -> Path:
    """Regenerable intermediates: audits, distance matrices."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT


_announce_package()

"""Record exact provenance for the most recent complete article run.

The manifest lives in the shared, unversioned ``results`` directory.  Keeping
it outside the three repositories avoids a self-referential commit hash while
still allowing a release archive to include it alongside the computed output.
"""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys


SCRIPT_ROOT = Path(__file__).resolve().parent
ARTICLE_ROOT = SCRIPT_ROOT.parent
RESULTS_ROOT = ARTICLE_ROOT / "results"
MANIFEST = RESULTS_ROOT / "run-manifest.json"
REPOSITORIES = ("TeX", "jazz-harmony-experiments", "leadsheetanalyser")
PACKAGES = (
    "leadsheetanalyser",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "seaborn",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    """Digest relative names and contents of every regular file below root."""
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


def _worktree_sha256(repository: Path) -> str | None:
    """Digest tracked diffs and untracked file contents, or None when clean."""
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain=v1", "-z"]
    )
    if not status:
        return None

    digest = hashlib.sha256()
    digest.update(subprocess.check_output(
        ["git", "-C", str(repository), "diff", "--binary", "HEAD"]
    ))
    untracked = subprocess.check_output(
        ["git", "-C", str(repository), "ls-files", "--others",
         "--exclude-standard", "-z"]
    ).split(b"\0")
    for encoded in sorted(item for item in untracked if item):
        path = repository / encoded.decode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _repository_record(repository: Path) -> dict:
    worktree = _worktree_sha256(repository)
    tags = _git(repository, "tag", "--points-at", "HEAD").splitlines()
    return {
        "branch": _git(repository, "branch", "--show-current"),
        "commit": _git(repository, "rev-parse", "HEAD"),
        "dirty": worktree is not None,
        "tags_at_commit": sorted(tag for tag in tags if tag),
        "worktree_sha256": worktree,
    }


def _file_record(path: Path) -> dict:
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _files(patterns: tuple[Path, ...]) -> dict:
    paths = sorted({path for pattern in patterns for path in pattern.parent.glob(pattern.name)})
    return {
        path.relative_to(ARTICLE_ROOT).as_posix(): _file_record(path)
        for path in paths if path.is_file()
    }


def main() -> None:
    data_root = ARTICLE_ROOT / "leadsheetanalyser" / "data"
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repositories": {
            name: _repository_record(ARTICLE_ROOT / name)
            for name in REPOSITORIES
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": {name: version(name) for name in PACKAGES},
        },
        "source_data": {
            "leadsheetanalyser/data/meta.csv": _file_record(data_root / "meta.csv"),
            "leadsheetanalyser/data/music_realbook.pkl": _file_record(
                data_root / "music_realbook.pkl"
            ),
            "leadsheetanalyser/data/jams_files": {
                "sha256_tree": _tree_sha256(data_root / "jams_files")
            },
        },
        "computed_outputs": _files((
            SCRIPT_ROOT / "article-data" / "*.json",
            SCRIPT_ROOT / "analysis-data" / "*.json",
            RESULTS_ROOT / "cache" / "*.csv",
            RESULTS_ROOT / "figures" / "*.pdf",
            RESULTS_ROOT / "figures" / "*.png",
            ARTICLE_ROOT / "TeX" / "generated" / "*.tex",
        )),
    }
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_name(f".{MANIFEST.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(MANIFEST)
    print(f"recorded run provenance in {MANIFEST}")


if __name__ == "__main__":
    sys.exit(main())

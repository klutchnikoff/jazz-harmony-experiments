"""Publish completed figure builds into the optional manuscript checkout.

The figure producers always write their canonical build products to
``../results/figures``.  This explicit final step is the only code that copies
those files into ``TeX/fig``; a reader can therefore reproduce the analysis
without possessing the private manuscript repository.
"""

from pathlib import Path
import shutil

SCRIPT_ROOT = Path(__file__).resolve().parent
ARTICLE_ROOT = SCRIPT_ROOT.parent
FIGURE_ROOT = ARTICLE_ROOT / "results" / "figures"
STEMS = (
    "order-used-below",
    "degrees-of-a-key",
    "borrowed-degrees",
    "two-repertoires",
)
FORMATS = ("pdf", "png")
TEX_ROOT = ARTICLE_ROOT / "TeX"
TEX_FIGURE_ROOT = TEX_ROOT / "fig"


def _atomic_copy(source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def main() -> None:
    if not (TEX_ROOT / "main.tex").exists():
        print("manuscript checkout absent; figures remain in results/figures")
        return

    missing = [
        FIGURE_ROOT / f"{stem}.{suffix}"
        for stem in STEMS
        for suffix in FORMATS
        if not (FIGURE_ROOT / f"{stem}.{suffix}").exists()
    ]
    if missing:
        names = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"cannot publish missing figure builds:\n{names}")

    TEX_FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    for stem in STEMS:
        for suffix in FORMATS:
            source = FIGURE_ROOT / f"{stem}.{suffix}"
            target = TEX_FIGURE_ROOT / source.name
            _atomic_copy(source, target)
    print(f"published {len(STEMS) * len(FORMATS)} figure files to {TEX_FIGURE_ROOT}")


if __name__ == "__main__":
    main()

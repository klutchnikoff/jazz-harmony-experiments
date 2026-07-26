"""Exporting the article's numbers, and checking the manuscript still states them.

Every script named ART-*.py produces the figures of one subsection and calls
`export` with them.  The rule that makes this worth anything: a script exports
exactly the numbers the subsection states, no more.  ART-check.py can then read
the manuscript and verify that each exported value is there, so that a figure
recomputed after a change of rule cannot quietly stay stale in the text.

Values land in article-data/<name>.json, versioned with the scripts.
"""
import json
from pathlib import Path

from article_setup import ARTICLE_ROOT

# Versioned with the scripts, not with the results: a diff on this directory is
# the history of every number the article states.
DATA_DIR = Path(__file__).resolve().parent / "article-data"
MANUSCRIPT = ARTICLE_ROOT / "TeX" / "main.tex"


def export(name, values):
    """Write the values of one subsection, and say where they went."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.json"
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    print(f"exported {len(values)} values to {path.parent.name}/{path.name}")
    return path


def load_all():
    """Every exported subsection, as {name: {key: value}}."""
    if not DATA_DIR.exists():
        return {}
    return {p.stem: json.loads(p.read_text()) for p in sorted(DATA_DIR.glob("*.json"))}


def manuscript_text():
    """The manuscript with LaTeX digit grouping removed, ready for searching."""
    text = MANUSCRIPT.read_text()
    return text.replace("{,}", "").replace("\\,", "").replace(",", "")


def occurrences(value, text):
    """Does this value appear in the manuscript, as a standalone number?"""
    import re
    literal = f"{value}"
    return bool(re.search(rf"(?<![\d.]){re.escape(literal)}(?![\d.])", text))

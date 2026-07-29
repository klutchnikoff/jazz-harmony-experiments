"""Exporting the article's numbers, and checking the manuscript still states them.

Every script named ART-<subsection title>.py produces the figures of exactly one
subsection and calls `export` with them.  Nothing else carries that prefix, so
listing ART-* shows the article's structure at a glance.

The rule that makes this worth anything: a script exports exactly the numbers its
subsection states, no more.  check_article_numbers.py then reads the manuscript
and verifies that each exported value is there, so that a figure recomputed after
a change of rule cannot quietly stay stale in the text.

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
    """The manuscript with LaTeX digit grouping removed, ready for searching.

    Inline math delimiters go too, so that a count written \\(32\\) kinds can be
    checked as the phrase "32 kinds": a small integer on its own occurs in too
    many places for finding it to prove anything.

    Only the grouping is stripped, not every comma: the manuscript writes large
    numbers as 172{,}783, so plain commas belong to vectors, and removing them
    would make a weight such as (0,1,0,2,0,3,2,0,1,0,1) unsearchable.  Runs of
    whitespace are collapsed so a phrase remains searchable across TeX source
    line breaks.
    """
    text = MANUSCRIPT.read_text()
    for old, new in (("{,}", ""), ("\\,", ""), ("\\(", ""), ("\\)", "")):
        text = text.replace(old, new)
    return " ".join(text.split())


def occurrences(value, text):
    """Does this value appear in the manuscript, as a standalone number?"""
    import re
    literal = f"{value}"
    # A digit at either end must not abut another digit, nor a decimal point or
    # comma that carries one: without that, 28 matches inside 2829 and a
    # truncated vector matches as a prefix of the full one.  What the separator
    # is followed by settles it, since prose puts a comma after a number too:
    # "0.19," ends the number, "0.19,5" would not.
    before = r"(?<!\d)(?<!\d[.,])" if literal[:1].isdigit() else ""
    after = r"(?!\d)(?![.,]\d)" if literal[-1:].isdigit() else ""
    return bool(re.search(before + re.escape(literal) + after, text))

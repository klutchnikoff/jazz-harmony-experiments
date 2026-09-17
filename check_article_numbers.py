"""Check the generated LaTeX snapshot against the versioned article data.

Each ART-*.py script refreshes both its JSON and the generated macro file. This
check has no scientific-package dependencies: it verifies that the macro file
is byte-for-byte current, that every exported key is referenced by the
manuscript, and that the scientific literals already migrated to macros have
not been reintroduced by hand.

Run: python check_article_numbers.py
"""

import re
import sys

from article_data import MANUSCRIPT, TEX_VALUES, load_all, render_tex_values


# Contextual spellings formerly present in main.tex.  This is deliberately not
# a ban on every numeral: equation indices, definitions, years, and bibliographic
# metadata remain legitimate LaTeX source.  These patterns guard the results and
# protocol constants that have specifically been connected to article-data.
LEGACY_SCIENTIFIC_LITERALS = (
    r"\bseven diatonic (?:modes|coordinates)\b",
    r"\bnine-(?:mode|coordinate)",
    r"\b(?:seven|nine) modes\b",
    r"\ball six cut points\b",
    r"\bThe four largest contributions\b",
    r"\bat least three parsable chords\b",
    r"\b(?:Twenty|Twelve) kinds\b",
    r"\bthirty-two kinds\b",
    r"\\\(B=20\{,\}000\\\)",
    r"\\\(p=0\.(?:15|18|19|3)\\\)",
    r"\\\((?:63\.05|25\.57|1\.57|5\.00|1\.77)\\\)",
    r"\\\(2,5,7,11\\\)",
    r"\\\((?:95|90)\\%\\\)",
)


def main():
    exported = load_all()
    if not exported:
        print("no exported values yet; run the ART-*.py scripts first")
        return 1
    if not MANUSCRIPT.exists():
        print(f"manuscript not present: {MANUSCRIPT}; JSON exports are available")
        return 0
    if not TEX_VALUES.exists():
        print(f"missing generated macro file: {TEX_VALUES}")
        return 1

    expected = render_tex_values(exported)
    if TEX_VALUES.read_text() != expected:
        print(f"stale generated macro file: {TEX_VALUES}")
        print("run any ART-*.py producer, or article_data.write_tex_values()")
        return 1

    manuscript = MANUSCRIPT.read_text()
    references = set(re.findall(
        r"\\ArticleValue(?:Thin|Word|WordCap)?\{([^{}]+)\}\{([^{}]+)\}",
        manuscript,
    ))
    available = {
        (section, key)
        for section, values in exported.items()
        for key in values
    }
    missing = sorted(available - references)
    unknown = sorted(references - available)
    legacy = [pattern for pattern in LEGACY_SCIENTIFIC_LITERALS
              if re.search(pattern, manuscript)]
    total = len(available)

    if missing:
        print(f"{len(missing)} of {total} exported values are not referenced:")
        for section, key in missing:
            print(f"  {section}/{key}")
    if unknown:
        print(f"{len(unknown)} manuscript references have no exported value:")
        for section, key in unknown:
            print(f"  {section}/{key}")
    if legacy:
        print("scientific literals already migrated to macros were found:")
        for pattern in legacy:
            print(f"  {pattern}")
    if missing or unknown or legacy:
        return 1

    print(f"generated macros are current; all {total} values are referenced")
    return 0


if __name__ == "__main__":
    sys.exit(main())

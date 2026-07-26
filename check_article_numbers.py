"""Check that the manuscript still states the numbers the scripts produce.

Each ART-*.py script exports the figures of one subsection.  This reads them all
and looks for each one in TeX/main.tex, digit grouping removed.  A value that
cannot be found is a number the pipeline computes and the text does not state --
almost always a figure recomputed after a change of rule and never carried over.

Run:  LSA_LOCAL=1 .venv/bin/python check_article_numbers.py
"""
import sys

from article_data import load_all, manuscript_text, occurrences


def main():
    exported = load_all()
    if not exported:
        print("no exported values yet; run the ART-*.py scripts first")
        return 0

    text = manuscript_text()
    missing = []
    for name, values in exported.items():
        print(f"\n{name}")
        for key, value in sorted(values.items()):
            found = occurrences(value, text)
            print(f"  {'ok ' if found else 'MISSING'}  {key:24s} {value}")
            if not found:
                missing.append((name, key, value))

    total = sum(len(v) for v in exported.values())
    if missing:
        print(f"\n{len(missing)} of {total} values are not in the manuscript:")
        for name, key, value in missing:
            print(f"  {name}: {key} = {value}")
        return 1
    print(f"\nall {total} values appear in the manuscript")
    return 0


if __name__ == "__main__":
    sys.exit(main())

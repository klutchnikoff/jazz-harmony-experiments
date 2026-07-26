"""Article data for Section 3.5, "The system used below".

The table of that subsection gives each mode's weights as an integer vector over
its total.  This exports those vectors, so that a weight edited in the manuscript
without being changed in the package -- or the reverse -- is caught.

Vectors are exported in the manuscript's own form, comma-separated and without
spaces, which is why check_article_numbers.py strips only LaTeX digit grouping
and leaves plain commas alone.

Run:  LSA_LOCAL=1 .venv/bin/python ART-the-system-used-below.py
"""
from fractions import Fraction

import numpy as np

from article_data import export
from leadsheetanalyser.constants import (
    W_DIATONIC, W_MESSIAEN, DIATONIC_MODE_NAMES, MESSIAEN_MODE_NAMES)

BRIGHT = ["Lydian", "Ionian", "Mixolydian", "Dorian", "Aeolian", "Phrygian",
          "Locrian"]


def integer_form(row):
    """(denominator, comma-separated numerators) for a row of weights."""
    fractions = [Fraction(x).limit_denominator(10 ** 6) for x in row]
    denominator = 1
    for f in fractions:
        denominator = denominator * f.denominator // np.gcd(denominator,
                                                            f.denominator)
    numerators = [int(f * denominator) for f in fractions]
    common = int(np.gcd.reduce([n for n in numerators if n]))
    if common > 1:
        numerators = [n // common for n in numerators]
        denominator //= common
    return denominator, ",".join(str(n) for n in numerators)


def main():
    order = [DIATONIC_MODE_NAMES.index(m) for m in BRIGHT]
    rows = [(m, np.asarray(W_DIATONIC, float)[j])
            for m, j in zip(BRIGHT, order)]
    rows.append(("Whole-tone", np.asarray(W_MESSIAEN, float)[0]))
    rows.append(("Octatonic", np.asarray(W_MESSIAEN, float)[1]))

    words = {7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven"}
    values = {"system_size": f"{words[len(rows)]} modes"}
    print(f"\n{'mode':12s} {'1/d':>6s}  weights")
    for name, row in rows:
        d, vector = integer_form(row)
        assert abs(row.sum() - 1) < 1e-12, f"{name} does not sum to one"
        values[f"weights_{name.lower().replace('-', '_')}"] = vector
        print(f"{name:12s} {'1/' + str(d):>6s}  ({vector})")

    export("the-system-used-below", values)


if __name__ == "__main__":
    main()

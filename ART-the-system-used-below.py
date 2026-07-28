"""Article data for Section 3.4, "The system used below".

The table of that subsection gives each mode's weights as an integer vector over
its total.  This exports those vectors, so that a weight edited in the manuscript
without being changed in the package -- or the reverse -- is caught.

The script also reconstructs the diatonic rows from the rule in Section 3.2 and
checks the support and transposition claims made for the two symmetric rows.

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
CENTRES = ["F", "C", "G", "D", "A", "E", "B"]
PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5,
                 "G": 7, "A": 9, "B": 11}
C_DIATONIC = set(PITCH_CLASSES.values())


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


def assert_brightness_order(rows):
    """Each diatonic support is the one above it with a single degree lowered.

    Section 3.2 defines brighter and darker by that ordering and by nothing else,
    so the table's order is a claim and not a convenience.
    """
    supports = [{i + 1 for i, x in enumerate(row) if x > 0} for _, row in rows[:7]]
    for above, below in zip(supports, supports[1:]):
        gone, came = above - below, below - above
        assert len(gone) == len(came) == 1, (
            f"the supports differ by {len(gone)} degrees, not one")
        assert came.pop() == gone.pop() - 1, (
            "the degree that changes is not lowered by a semitone")


def assert_diatonic_construction(rows):
    """Reconstruct the first seven rows from the formula in Section 3.2."""
    for (name, row), centre_name in zip(rows[:7], CENTRES):
        centre = PITCH_CLASSES[centre_name]
        S = sorted((pitch - centre) % 12 for pitch in C_DIATONIC)
        assert len(S) == 7 and S[0] == 0

        # The third and fifth are the third and fifth scale degrees.
        t, f = S[2], S[4]
        T = {i for i in S if (i + 6) % 12 in S}
        assert len(T) == 2, f"{name} does not contain a unique tritone pair"
        tau = T - {0}
        assert len(tau) in (1, 2)

        numerators = np.array([
            int(i in S)
            + int(i in {t, f})
            + (2 / len(tau)) * int(i in tau)
            for i in range(1, 12)
        ])
        assert numerators.sum() == 10
        assert np.allclose(row, numerators / 10), (
            f"{name} does not follow the diatonic weighting formula")


def assert_symmetric_collections(rows):
    """Check the supports, uniform weights, and transposition counts."""
    expected = [
        ({0, 2, 4, 6, 8, 10}, 5, 2),
        ({0, 1, 3, 4, 6, 7, 9, 10}, 7, 3),
    ]
    for (name, row), (support, non_tonic_size, transpositions) in zip(
            rows[7:], expected):
        actual = {0} | {i + 1 for i, weight in enumerate(row) if weight > 0}
        assert actual == support, f"unexpected support for {name}"
        assert np.count_nonzero(row) == non_tonic_size
        assert np.allclose(row[row > 0], 1 / non_tonic_size)
        orbit = {
            tuple(sorted((pitch + shift) % 12 for pitch in actual))
            for shift in range(12)
        }
        assert len(orbit) == transpositions, (
            f"unexpected number of transpositions for {name}")


def main():
    order = [DIATONIC_MODE_NAMES.index(m) for m in BRIGHT]
    rows = [(m, np.asarray(W_DIATONIC, float)[j])
            for m, j in zip(BRIGHT, order)]
    rows.append(("Whole-tone", np.asarray(W_MESSIAEN, float)[0]))
    rows.append(("Octatonic", np.asarray(W_MESSIAEN, float)[1]))

    assert_brightness_order(rows)
    assert_diatonic_construction(rows)
    assert_symmetric_collections(rows)

    words = {7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven"}
    values = {
        "diatonic_centres": "F, C, G, D, A, E, and B",
        "diatonic_modes_first": "Lydian, Ionian, Mixolydian, Dorian",
        "diatonic_modes_last": "Aeolian, Phrygian",
        "system_size": f"{words[len(rows)]} modes",
    }
    print(f"\n{'mode':12s} {'1/d':>6s}  weights")
    for name, row in rows:
        d, vector = integer_form(row)
        assert abs(row.sum() - 1) < 1e-12, f"{name} does not sum to one"
        values[f"weights_{name.lower().replace('-', '_')}"] = vector
        print(f"{name:12s} {'1/' + str(d):>6s}  ({vector})")

    export("the-system-used-below", values)


if __name__ == "__main__":
    main()

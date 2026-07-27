"""Article data for Section 5.3, "Borrowed degrees".

Figure 2 draws a dot on the mode harmony predicts for each of the eight
chromatic degrees, and the subsection says the readings fall there.  Nothing
tested that until this script: a figure that asserts without the pipeline
checking it is exactly what the rest of the article refuses.

The predictions owe nothing to the modal system.  A major key borrows from its
parallel minor, whose mode is Aeolian.  The flattened supertonic is the degree
Phrygian is known by.  A tonic carrying a flattened seventh is what Mixolydian
is.  The subdominant seventh brings the minor third to a key that keeps its
major sixth, which is Dorian.  They were written down after the readings were
computed, though, so they corroborate where the pairings of Section 4 test.

Exports only the two ends of the Aeolian run, the five degrees between them
being covered by the assertion rather than by a number in the text.

Run:  LSA_LOCAL=1 .venv/bin/python ART-borrowed-degrees.py
"""
import numpy as np

from article_data import export
from chord_scale import SYSTEM, MODES

ORDER = 0.15
MAJ7, MI7, DOM, HALF = (4, 7, 11), (3, 7, 10), (4, 7, 10), (3, 6, 10)

# (name, semitones above the tonic, kind, the mode harmony predicts)
BORROWED = [
    ("half-diminished supertonic", 2, HALF, "Aeolian"),
    ("flattened mediant", 3, MAJ7, "Aeolian"),
    ("minor subdominant", 5, MI7, "Aeolian"),
    ("flattened submediant", 8, MAJ7, "Aeolian"),
    ("flattened leading note", 10, DOM, "Aeolian"),
]
ELSEWHERE = [
    ("flattened supertonic", 1, DOM, "Phrygian"),
    ("tonic seventh", 0, DOM, "Mixolydian"),
    ("subdominant seventh", 5, DOM, "Dorian"),
]


def reading(offset, kind):
    content = {(offset + i) % 12 for i in (0,) + tuple(kind)} | {0}
    intervals = [i - 1 for i in range(1, 12) if i in content]
    m = np.mean(SYSTEM[:, intervals] ** ORDER, axis=1) ** (1 / ORDER)
    return m / m.sum()


def main():
    shares = {}
    print(f"\n{'degree':30s} {'predicted':11s} {'read':11s} share")
    for name, offset, kind, want in BORROWED + ELSEWHERE:
        p = reading(offset, kind)
        top = [MODES[j] for j in range(9) if p[j] >= p.max() - 1e-12]
        print(f"{name:30s} {want:11s} {'='.join(top):11s} {p.max():.3f}")
        assert top == [want], (
            f"the {name} reads {top} and no longer {want} outright, "
            "so Figure 2 draws a dot outside its box")
        shares[name] = p.max()

    run = [shares[n] for n, _, _, _ in BORROWED]
    lo, hi = min(run), max(run)
    assert shares["minor subdominant"] == lo and \
        shares["half-diminished supertonic"] == hi, \
        "the ends of the Aeolian run have moved, and Section 5.3 names them"
    print(f"\nthe five borrowed degrees read Aeolian from {lo:.3f} to {hi:.3f}")

    export("borrowed-degrees", {
        "aeolian_lowest": f"{lo:.2f}",
        "aeolian_highest": f"{hi:.2f}",
    })


if __name__ == "__main__":
    main()

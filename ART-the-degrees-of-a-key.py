"""Article data for Section 5.2, "The degrees of a key".

The subsection makes four counts and quotes one number.  The counts are small
integers that occur all over the manuscript, so they are asserted here rather
than exported, and only the supertonic's tie carries a figure worth searching
for.

The choice of sevenths is the repertoire's before it is ours, and the share of
symbols carrying one is exported for both corpora: a majority of the jazz
symbols, a quarter of the common-practice ones, which is the difference between
a repertoire that writes ii7 and one that writes ii.

  major sevenths   five of the seven diatonic degrees read Ionian outright
  minor sevenths   five of the seven read Aeolian outright
  triads           three of seven in either key
  sixths           three in the major, two in the minor

"Outright" means the mode is the sole maximum.  The supertonic of a major key
divides its largest share evenly between Ionian and Dorian, and is not counted;
counting a tie as a hit is what made an earlier tally of the chord-scale
pairings read 15 of 16 when it was 13.

Run:  LSA_LOCAL=1 .venv/bin/python ART-the-degrees-of-a-key.py
"""
import numpy as np

from article_data import export
from chord_scale import SYSTEM, MODES
from vocabulary import corpus_counts

ORDER = 0.15
MAJ7, MI7, DOM, HALF = (4, 7, 11), (3, 7, 10), (4, 7, 10), (3, 6, 10)

# (label, semitones above the tonic, quality) for the diatonic degrees
MAJOR = [("I", 0, "maj"), ("ii", 2, "min"), ("iii", 4, "min"), ("IV", 5, "maj"),
         ("V", 7, "dom"), ("vi", 9, "min"), ("vii", 11, "dim")]
MINOR = [("i", 0, "min"), ("ii", 2, "dim"), ("III", 3, "maj"), ("iv", 5, "min"),
         ("v", 7, "min"), ("VI", 8, "maj"), ("VII", 10, "dom")]

FORMS = {
    "triad":   {"maj": (4, 7), "min": (3, 7), "dom": (4, 7), "dim": (3, 6)},
    "sixth":   {"maj": (4, 7, 9), "min": (3, 7, 9), "dom": (4, 7, 9),
                "dim": (3, 6, 9)},
    "seventh": {"maj": MAJ7, "min": MI7, "dom": DOM, "dim": HALF},
}


def reading(offset, kind):
    content = {(offset + i) % 12 for i in (0,) + tuple(kind)} | {0}
    intervals = [i - 1 for i in range(1, 12) if i in content]
    m = np.mean(SYSTEM[:, intervals] ** ORDER, axis=1) ** (1 / ORDER)
    return m / m.sum()


def top_modes(offset, kind):
    p = reading(offset, kind)
    return [MODES[j] for j in range(9) if p[j] >= p.max() - 1e-12], p.max()


def tally(degrees, form, home):
    """How many degrees have `home` as their sole largest share."""
    n = 0
    for _, offset, quality in degrees:
        modes, _ = top_modes(offset, FORMS[form][quality])
        n += modes == [home]
    return n


def main():
    print(f"\n{'':9s} {'triad':>7s} {'sixth':>7s} {'seventh':>8s}")
    counts = {}
    for label, degrees, home in (("C major", MAJOR, "Ionian"),
                                 ("A minor", MINOR, "Aeolian")):
        row = {f: tally(degrees, f, home) for f in FORMS}
        counts[label] = row
        print(f"{label:9s} " + " ".join(f"{row[f]:>7d}"
                                        for f in ("triad", "sixth", "seventh")))

    assert counts["C major"]["seventh"] == 5, "the major sevenths no longer read 5 of 7"
    assert counts["A minor"]["seventh"] == 5, "the minor sevenths no longer read 5 of 7"
    assert counts["C major"]["triad"] == 3 and counts["A minor"]["triad"] == 3
    assert counts["C major"]["sixth"] == 3 and counts["A minor"]["sixth"] == 2

    modes, share = top_modes(2, MI7)          # the supertonic of a major key
    assert modes == ["Ionian", "Dorian"], f"the supertonic now reads {modes}"
    print(f"\nthe supertonic seventh divides {share:.4f} between "
          f"{' and '.join(modes)}")

    tokens = corpus_counts()[0]
    seventh = {}
    for corpus in ("jazz", "cp"):
        counts = tokens[corpus]
        total = sum(counts.values())
        with_seventh = sum(n for k, n in counts.items() if k[9] or k[10])
        seventh[corpus] = 100 * with_seventh / total
        print(f"  {corpus:4s} {with_seventh:7,d} of {total:7,d} symbols carry a "
              f"seventh, {seventh[corpus]:.1f}%")
    assert seventh["jazz"] > 50 > seventh["cp"], (
        "the sevenths are no longer the jazz norm and the triads the "
        "common-practice one, which is what Section 5.2 rests on")

    export("the-degrees-of-a-key", {
        "supertonic_tie": f"{share:.2f}",
        "jazz_sevenths": f"{seventh['jazz']:.1f}",
        "cp_sevenths": f"{seventh['cp']:.1f}",
    })


if __name__ == "__main__":
    main()

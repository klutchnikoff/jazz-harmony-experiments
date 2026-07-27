"""Article data for Section 5.1, "Reading from the tonic".

The subsection defines the degree of a chord in a key,

    C_q(c) = C(r,k) + {q},    k^q_i = 1  iff  q + i in C(r,k),

and states three properties of it before giving one example.  The properties are
asserted here rather than exported, having no number in the manuscript to match:

  transposition   the degree depends on r - q and k alone, so the same chord a
                  fixed interval above two different tonics gives one degree.
  extension       q = r returns the kind itself, so Section 4 is the case of a
                  chord sitting on the tonic.
  the lost root   Ami7 and C6 in C share their pitch classes, so they share a
                  degree.  The cost of passing through the content, and the
                  ambiguity Section 2.1 already named.

Only the example carries figures: C7 read from F against C7 read from C, which
is the whole point of the section in two numbers.

Run:  LSA_LOCAL=1 .venv/bin/python ART-reading-from-the-tonic.py
"""
import itertools

import numpy as np

from article_data import export
from chord_scale import SYSTEM, MODES

ORDER = 0.15
NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# (name, root, intervals of the kind)
C7 = ("C7", 0, (4, 7, 10))
AMI7 = ("Ami7", 9, (3, 7, 10))
C6 = ("C6", 0, (4, 7, 9))


def content(root, kind):
    return {(root + i) % 12 for i in (0,) + tuple(kind)}


def degree(root, kind, tonic):
    """The kind rooted on the tonic whose content is C(r,k) together with q."""
    whole = content(root, kind) | {tonic}
    return tuple(1 if (tonic + i) % 12 in whole else 0 for i in range(1, 12))


def reading(deg, p=ORDER):
    intervals = [i for i in range(11) if deg[i]]
    m = np.mean(SYSTEM[:, intervals] ** p, axis=1) ** (1 / p)
    return m / m.sum()


def main():
    # transposition: the degree turns on r - q, not on the key
    for shift in range(12):
        assert degree((C7[1] + shift) % 12, C7[2], (5 + shift) % 12) \
            == degree(C7[1], C7[2], 5), "the degree depends on the absolute key"

    # extension: a chord on the tonic gives back its own kind
    for root, kind in ((C7[1], C7[2]), (AMI7[1], AMI7[2]), (C6[1], C6[2])):
        expected = tuple(1 if i in kind else 0 for i in range(1, 12))
        assert degree(root, kind, root) == expected, "q = r does not return k"

    # the lost root, in C
    assert degree(*AMI7[1:], 0) == degree(*C6[1:], 0), \
        "Ami7 and C6 no longer share a degree in C"

    values = {}
    print(f"\n{'chord':6s} {'in':3s}  {'content with the tonic':26s} "
          f"{'intervals':16s} reading")
    for tonic in (5, 0):                       # F, then C
        deg = degree(C7[1], C7[2], tonic)
        pr = reading(deg)
        top = int(np.argmax(pr))
        assert sum(1 for x in pr if x >= pr.max() - 1e-12) == 1, \
            "the example has no single top mode, which the text assumes"
        whole = sorted(content(C7[1], C7[2]) | {tonic})
        intervals = [i + 1 for i in range(11) if deg[i]]
        print(f"{C7[0]:6s} {NOTES[tonic]:3s}  "
              f"{'{' + ','.join(NOTES[c] for c in whole) + '}':26s} "
              f"{str(intervals):16s} {MODES[top]} {pr[top]:.3f}")
        values[f"c7_in_{NOTES[tonic].lower()}"] = f"{MODES[top]} at {pr[top]:.2f}"

    export("reading-from-the-tonic", values)


if __name__ == "__main__":
    main()

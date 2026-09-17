"""The pairings of chords with scales that the reading is tested against.

Two tiers, and no third.

  Named.       The pairings jazz pedagogy states for a chord type: Ionian with
               the major seventh, Mixolydian with the dominant seventh and with
               the suspended dominant, Dorian with the minor seventh and the
               minor sixth, Locrian with the half-diminished and the diminished
               triad, the octatonic scale with the altered dominants, and the
               whole-tone scale with the augmented dominant (Levine 1995).

  Ninth rule.  A kind obtained by adding the ninth to one of those inherits its
               pairing, the ninth being a degree of the paired mode in each case.
               The rule admits no latitude and reaches six further kinds.

Everything else is left unpaired.  Removing a seventh is not covered: without it
a dominant is no longer a dominant, and C(add9) is not a Mixolydian chord because
C9 is.  Extensions beyond the ninth are not covered either.  Assigning those by
ear would be this table's author deciding what the test should conclude.

The pairings are fixed here, before any profile is computed.

Run:  LSA_LOCAL=1 .venv/bin/python chord_scale.py
"""
import numpy as np

import article_setup  # noqa: F401  (resolves leadsheetanalyser before it is used)
from leadsheetanalyser.chord_dissimilarities import modal_profile
from leadsheetanalyser.constants import (
    W_DIATONIC, W_MESSIAEN, DIATONIC_MODE_NAMES)
from vocabulary import build, name

BRIGHT = ["Lydian", "Ionian", "Mixolydian", "Dorian", "Aeolian", "Phrygian",
          "Locrian"]
MODES = BRIGHT + ["whole-tone", "octatonic"]
SYSTEM = np.vstack([np.asarray(W_DIATONIC, float)[[DIATONIC_MODE_NAMES.index(m)
                                                   for m in BRIGHT]],
                    np.asarray(W_MESSIAEN, float)[:2]])

NAMED = {
    "maj7": "Ionian",
    "7": "Mixolydian",
    "7sus4": "Mixolydian",
    "min7": "Dorian",
    "min6": "Dorian",
    "min7b5": "Locrian",
    "dim": "Locrian",
    "7b9": "octatonic",
    "7#9": "octatonic",
    "aug7": "whole-tone",
}

NINTH_RULE = {
    "maj9": "maj7",
    "9": "7",
    "min9": "min7",
    "min69": "min6",
    "9#5": "aug7",
    "7sus2sus4": "7sus4",
}

PAIRING = dict(NAMED)
PAIRING.update({k: NAMED[base] for k, base in NINTH_RULE.items()})


def rank_of(kind, expected, p):
    profile = modal_profile(np.array(kind, int), SYSTEM, p)
    order = np.argsort(profile)[::-1]
    return int(np.where(order == MODES.index(expected))[0][0]) + 1, profile


def main():
    vocab = build()[0]
    tested = [k for k in vocab if name(k) in PAIRING]
    unpaired = [k for k in vocab if name(k) not in PAIRING]
    print(f"\n{len(tested)} of the {len(vocab)} kinds carry a pairing "
          f"({len(NAMED)} named, {len(NINTH_RULE)} by the ninth rule); "
          f"{len(unpaired)} do not")
    print("unpaired: " + ", ".join(sorted(name(k) for k in unpaired)))

    for p in (1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05):
        ranks = [rank_of(k, PAIRING[name(k)], p)[0] for k in tested]
        print(f"\n  p = {p:4.2f}   first {sum(r == 1 for r in ranks):2d}/"
              f"{len(ranks)}   in the first two "
              f"{sum(r <= 2 for r in ranks):2d}/{len(ranks)}   mean rank "
              f"{np.mean(ranks):.2f}")
        if p == 0.15:
            for k in tested:
                r, prof = rank_of(k, PAIRING[name(k)], p)
                top = int(np.argmax(prof))
                flag = "  " if r == 1 else ("<-" if r == 2 else "<<")
                print(f"     {flag} {name(k):12s} wants {PAIRING[name(k)]:11s}"
                      f" rank {r}   reads {MODES[top]:11s} {prof[top]:.3f}")


if __name__ == "__main__":
    main()

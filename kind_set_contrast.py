"""Why Figure 1 shows the twenty kinds and not five.

Section 7.2 rests on the count of kinds keeping a unique most-affine mode: 14 of
20 under W_D against 6 under uniform weights.  That contrast depends on WHICH
kinds are displayed, and the figure was designed on the strength of this table.
It is kept in the pipeline because the design decision is otherwise invisible.

The contrast is large over the diagnostic vocabulary and over the twenty
commonest kinds, but nearly vanishes on any five-kind subset: the five kinds of
"Funky" leave only one kind between the two systems, the five foundational kinds
of the earlier figure only two.  Bare triads are the informative case, being
modally impoverished, which is why a five-kind figure had to be foundational --
and why showing all twenty says more than either.

Run:  python py-code/kind_set_contrast.py
"""
from collections import Counter

import numpy as np
import pandas as pd

import article_setup  # noqa: F401
from article_setup import DATA_ROOT
from diagnostic_vocabulary import VOCABULARY
from leadsheetanalyser.constants import W_DIATONIC, DIATONIC_MODE_NAMES

WD = np.asarray(W_DIATONIC, float)
SUPPORT = (WD > 0).astype(float)
W_UNIFORM = SUPPORT / SUPPORT.sum(axis=1, keepdims=True)
BRIGHT = ["Lydian", "Ionian", "Mixolydian", "Dorian", "Aeolian", "Phrygian", "Locrian"]
order = [DIATONIC_MODE_NAMES.index(m) for m in BRIGHT]

VOCAB_NAME = {tuple(k): n for n, k, _ in VOCABULARY}
FUNKY = {"Cmaj7", "C7", "Cmi7", "Cmi7b5", "Cmi6"}


def maxima(row, tol=1e-9):
    return np.flatnonzero(row >= row.max() - tol)


# --- 2. does the tie-breaking contrast survive on various kind sets? -------
def contrast(kinds, label):
    rows_wd = np.asarray([WD @ np.asarray(k, float) for k in kinds])
    rows_un = np.asarray([W_UNIFORM @ np.asarray(k, float) for k in kinds])
    u_wd = sum(1 for r in rows_wd if len(maxima(r)) == 1)
    u_un = sum(1 for r in rows_un if len(maxima(r)) == 1)
    print(f"  {label:38s} W_D {u_wd:2d}/{len(kinds)}   uniform {u_un:2d}/{len(kinds)}")


df = pd.read_pickle(DATA_ROOT / "music_realbook.pkl")
counts = Counter()
for prog in df["chord_progression"]:
    for c in prog:
        if c is None or c[0] is None or any(x is None for x in c[1:12]):
            continue
        counts[tuple(int(x) for x in c[1:12])] += 1
top20 = counts.most_common(20)

print("kinds with a UNIQUE most-affine mode")
contrast([k for _, k, _ in VOCABULARY], "diagnostic vocabulary (20)")
contrast([k for k, _ in top20], "20 most frequent kinds")
contrast([k for _, k, _ in VOCABULARY if VOCAB_NAME[tuple(k)] in FUNKY],
         "the five kinds of Funky")
FOUNDATIONAL = {"C", "Cmi", "Cdim", "C7", "Caug"}
contrast([k for _, k, _ in VOCABULARY if VOCAB_NAME[tuple(k)] in FOUNDATIONAL],
         "the five foundational kinds (Figure 1)")


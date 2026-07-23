"""
Verification of the complementarity claim of Section 9.2 (modes of limited
transposition).  Companion to modal_affinities.py, which handles the diatonic
system W_D.

The augmented triad and the diminished seventh are the two foundational kinds
that W_D leaves without a home (Figure 1).  Each is itself a mode of limited
transposition -- the augmented triad invariant under transposition by a major
third, the diminished seventh by a minor third -- so under the Messiaen system
W_L each saturates its own collection at the maximal affinity 1.  This script
checks both facts numerically; modal_affinities.py plots the saturation as the
third column of Figure 1.

Run:  python messiaen_affinities.py
"""
import numpy as np

import article_setup  # noqa: F401
from leadsheetanalyser.constants import (
    W_MESSIAEN, MESSIAEN_MODE_NAMES, W_DIATONIC, DIATONIC_MODE_NAMES,
)

WM = np.asarray(W_MESSIAEN, float)
WD = np.asarray(W_DIATONIC, float)

# (name, interval vector over semitones 1..11, transposition period in semitones)
KINDS = [
    ("augmented triad",   (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0), 4),
    ("diminished seventh", (0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0), 3),
]

print("complementarity of W_L and W_D (Section 9.2)\n")
print(f"  {'kind':20s}{'invariant':>12s}{'W_L max':>10s}{'W_D max':>10s}")
for name, kind, period in KINDS:
    k = np.asarray(kind, float)
    pcs = {0, *(i + 1 for i in range(11) if kind[i])}
    invariant = {(p + period) % 12 for p in pcs} == pcs
    wl, wd = WM @ k, WD @ k
    assert invariant, f"{name} is not invariant under +{period}"
    assert abs(wl.max() - 1.0) < 1e-9, f"{name} does not reach affinity 1 under W_L"
    print(f"  {name:20s}{f'+{period}':>12s}{wl.max():10.3f}{wd.max():10.3f}")

print(f"\n  W_L home of the augmented triad:   "
      f"{MESSIAEN_MODE_NAMES[int(np.argmax(WM @ np.asarray(KINDS[0][1], float)))]}")
print(f"  W_L home of the diminished seventh: "
      f"{MESSIAEN_MODE_NAMES[int(np.argmax(WM @ np.asarray(KINDS[1][1], float)))]}")
print(f"  under W_D their best diatonic modes are only "
      f"{DIATONIC_MODE_NAMES[int(np.argmax(WD @ np.asarray(KINDS[0][1], float)))]} "
      f"and {DIATONIC_MODE_NAMES[int(np.argmax(WD @ np.asarray(KINDS[1][1], float)))]}, "
      f"at affinities well below 1.")

"""
Reproducible figures behind the "Which dissimilarity to use?" paragraph.

For the 12 x 20 = 240 rooted chords formed from our kinds, in the diatonic
system W_D, compares D_min, D_max and D_sum on three criteria:

  1. discrimination  - does the measure vanish only on chords that truly share
                       a pitch-class set, or also on genuinely distinct ones?
  2. near-metricity  - fraction of triples violating the triangle inequality;
  3. centrality      - Spearman rank correlation between the three measures.

Run:  python py-code/rooted_chord_diagnostics.py
"""
import numpy as np
from itertools import combinations
from math import comb
from scipy.stats import spearmanr

import article_setup
from diagnostic_vocabulary import VOCABULARY
from leadsheetanalyser.constants import W_DIATONIC
from leadsheetanalyser.chord_dissimilarities import reinterpret_chord, modal_embedding
from leadsheetanalyser.chords import chord_to_pitch_classes

WD = np.asarray(W_DIATONIC, float)

def dissimilarities(chord1, chord2, W):
    r1, k1 = chord1
    r2, k2 = chord2
    k1_r2 = reinterpret_chord(k1, r1, r2)
    k2_r1 = reinterpret_chord(k2, r2, r1)
    d1 = np.linalg.norm(modal_embedding(k1_r2, W) - modal_embedding(k2, W))
    d2 = np.linalg.norm(modal_embedding(k2_r1, W) - modal_embedding(k1, W))
    return min(d1, d2), max(d1, d2), d1 + d2


# 240 rooted chords: (root, kind vector), plus each one's pitch-class set.
chords, pcs = [], []
for root in range(12):
    for _, kind, _ in VOCABULARY:
        k = kind
        kv = np.array(k, int)
        chords.append((root, kv))
        pcs.append(frozenset(chord_to_pitch_classes(root, kv)))
n = len(chords)

# Pairwise dissimilarity matrices.
Dmin = np.zeros((n, n)); Dmax = np.zeros((n, n)); Dsum = np.zeros((n, n))
same_pcs = np.zeros((n, n), bool)
for i, j in combinations(range(n), 2):
    dmn, dmx, dsm = dissimilarities(chords[i], chords[j], WD)
    Dmin[i, j] = Dmin[j, i] = dmn
    Dmax[i, j] = Dmax[j, i] = dmx
    Dsum[i, j] = Dsum[j, i] = dsm
    same_pcs[i, j] = same_pcs[j, i] = (pcs[i] == pcs[j])

TOL = 1e-9
measures = {"D_min": Dmin, "D_max": Dmax, "D_sum": Dsum}


def zero_pairs(D):
    """(# zero-dissimilarity pairs, # of those with DIFFERENT pitch-class sets)."""
    z = sum((D[i, j] < TOL) for i, j in combinations(range(n), 2))
    spurious = sum((D[i, j] < TOL) and not same_pcs[i, j] for i, j in combinations(range(n), 2))
    return z, spurious


def unordered_triangle_violations(D):
    """Count three-element sets on which any triangle inequality fails."""
    oriented_count = 0
    for y in range(n):                       # y is the intermediate vertex
        S = D[:, y][:, None] + D[y, :][None, :]   # S[x, z] = D(x, y) + D(y, z)
        V = D > S + TOL
        V[y, :] = False; V[:, y] = False; np.fill_diagonal(V, False)
        oriented_count += int(V.sum())
    # A failed inequality is counted twice above, once for each orientation
    # of its two endpoints.  At most one inequality can fail on a given
    # three-element set, so division by two gives the unordered count.
    assert oriented_count % 2 == 0
    return oriented_count // 2


def upper(D):
    return np.array([D[i, j] for i, j in combinations(range(n), 2)])


n_same = sum(same_pcs[i, j] for i, j in combinations(range(n), 2))
pair_total = n * (n - 1) // 2
triple_total = comb(n, 3)
print(f"n = {n} chords, {pair_total} pairs; {n_same} pairs share a pitch-class set")
print(f"unordered triples of distinct chords: {triple_total}\n")
print(
    f"{'measure':8} "
    f"{'zero pairs':>18} "
    f"{'spurious zeros':>20} "
    f"{'triangle violations':>27}"
)
for name, D in measures.items():
    z, sp = zero_pairs(D)
    tv = unordered_triangle_violations(D)
    print(
        f"{name:8} "
        f"{z:>8} ({100*z/pair_total:>5.2f}%) "
        f"{sp:>10} ({100*sp/pair_total:>5.2f}%) "
        f"{tv:>15} ({100*tv/triple_total:>5.2f}%)"
    )

um, ux, us = upper(Dmin), upper(Dmax), upper(Dsum)
print("\nSpearman rank correlations:")
print(f"  D_sum vs D_min : {spearmanr(us, um).statistic:.3f}")
print(f"  D_sum vs D_max : {spearmanr(us, ux).statistic:.3f}")
print(f"  D_min vs D_max : {spearmanr(um, ux).statistic:.3f}")


# ---------------------------------------------------------------------------
# corpus_distances.chord_cost_matrix computes the same ground cost as the
# package, vectorised over the whole vocabulary instead of pair by pair.  The
# corpus matrices cached in tmp/ were built before that rewrite, so the two must
# agree exactly or those caches would silently stop matching the code.
from corpus_distances import chord_cost_matrix, VARIANTS  # noqa: E402

rooted_chords = [(root, tuple(int(x) for x in kind))
                 for root in range(12) for _, kind, _ in VOCABULARY]
songs = [[(c, 1.0)] for c in rooted_chords]          # one chord per "song"
index, C = chord_cost_matrix(songs, *VARIANTS["duration"][:2])
reference = np.zeros_like(C)
for (r1, k1), i in index.items():
    for (r2, k2), j in index.items():
        if i < j:
            reference[i, j] = reference[j, i] = dissimilarities(
                (r1, np.asarray(k1)), (r2, np.asarray(k2)), WD)[2]
deviation = np.abs(C - reference).max()
print(f"\nvectorised ground cost vs package reference: "
      f"max |difference| = {deviation:.2e} over {len(index)} rooted chords")
assert deviation < 1e-9, "vectorised cost disagrees with the package"

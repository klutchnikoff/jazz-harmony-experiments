"""
Pairwise D_sum dissimilarities between the six distinct chords of Kenny
Burrell's "Funky" in the diatonic modal system W_D (Section 7.3).

The piece is not chosen for its fame.  Section 7.1 checks the modes that
Phi_{W_D} assigns to the five chord kinds that chord-scale theory names
(maj7, 7, mi7, mi7b5, mi6), and we wanted a piece of the corpus whose harmony
consists of those kinds and no others, with few enough distinct chords for the
matrix to be read directly.  The selection block below performs that search over
the key-corroborated Real Book and shows that "Funky" is the answer: the only
song whose duration lies entirely on the five kinds, with six distinct rooted
chords.  Its harmony in B-flat is I, ii7, ii7b5, IVmaj7, iv6, V7.

Two properties of D are visible on it, and the script prints both.

  Three pairs lie two pitch classes apart yet span D = 0.53 to 1.30.  D is
  therefore not counting shared notes here, unlike on the example this section
  used previously ("Cry Of The Wild Goose", where the rank correlation between D
  and the pitch-class symmetric difference was 0.99).

  D(Cmi7b5, Ebmi6) = 0, the two chords being the same pitch-class set read from
  two different roots.  This is one of the 66 pairs of Section 7.2, met in a real
  piece: the half-diminished supertonic and the minor subdominant with added
  sixth are the same four notes.

Run:  python py-code/funky_dissimilarity_matrix.py
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
from itertools import combinations
from scipy.stats import spearmanr

import article_setup  # noqa: F401
from article_setup import output_directory
from figure_style import (
    HEATMAP_ASPECT,
    HEATMAP_CMAP,
    annotated_heatmap_axes,
    save_article_figure,
)
from leadsheetanalyser.constants import W_DIATONIC
from leadsheetanalyser.chord_dissimilarities import reinterpret_chord, modal_embedding

def dissimilarities(chord1, chord2, W):
    r1, k1 = chord1
    r2, k2 = chord2
    k1_r2 = reinterpret_chord(k1, r1, r2)
    k2_r1 = reinterpret_chord(k2, r2, r1)
    d1 = np.linalg.norm(modal_embedding(k1_r2, W) - modal_embedding(k2, W))
    d2 = np.linalg.norm(modal_embedding(k2_r1, W) - modal_embedding(k1, W))
    return min(d1, d2), max(d1, d2), d1 + d2

OUT = output_directory()
WD = np.asarray(W_DIATONIC, dtype=float)

# "Funky" in its annotated key of B-flat major, in functional order.
CHORDS = [
    ("B♭maj7",  10, (0,0,0,1,0,0,1,0,0,0,1)),   # I
    ("Cmi7",     0, (0,0,1,0,0,0,1,0,0,1,0)),   # ii
    ("Cmi7♭5",   0, (0,0,1,0,0,1,0,0,0,1,0)),   # ii∅
    ("E♭maj7",   3, (0,0,0,1,0,0,1,0,0,0,1)),   # IV
    ("E♭mi6",    3, (0,0,1,0,0,0,1,0,1,0,0)),   # iv
    ("F7",       5, (0,0,0,1,0,0,1,0,0,1,0)),   # V
]

labels = [label for label, _, _ in CHORDS]
rooted = [(root, np.asarray(kind, dtype=int)) for _, root, kind in CHORDS]
n = len(rooted)
D = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        D[i, j] = D[j, i] = dissimilarities(rooted[i], rooted[j], WD)[2]

fig, ax, cax = annotated_heatmap_axes(n, n)
image = ax.imshow(D, cmap=HEATMAP_CMAP, vmin=0, aspect=HEATMAP_ASPECT)
ax.set_xticks(range(n), labels=labels, rotation=35, ha="right")
ax.set_yticks(range(n), labels=labels)
for i in range(n):
    for j in range(n):
        colour = "white" if D[i, j] > 0.55 * D.max() else "black"
        ax.text(j, i, f"{D[i, j]:.2f}", ha="center", va="center", color=colour)
fig.colorbar(image, cax=cax, label=r"$D_{\mathrm{sum}}$")
save_article_figure(fig, OUT, "funky_dissimilarity_matrix")
print("saved funky_dissimilarity_matrix.png and funky_dissimilarity_matrix.pdf")


def pitch_classes(root, kind):
    return {root % 12} | {(root + i) % 12 for i in range(1, 12) if kind[i - 1]}


pairs = list(combinations(range(n), 2))
sym = [len(pitch_classes(*rooted[a]) ^ pitch_classes(*rooted[b])) for a, b in pairs]
mod = [D[a, b] for a, b in pairs]
rho, p = spearmanr(sym, mod)

print(f"\npitch-class symmetric difference vs D: Spearman rho = {rho:.3f} (p = {p:.3f})")
print("\npairs two pitch classes apart:")
for (a, b), s, d in zip(pairs, sym, mod):
    if s == 2:
        print(f"  {labels[a]:>8s} - {labels[b]:<8s}  D = {d:.2f}")
print("\nvanishing pairs (identical pitch-class content):")
for (a, b), s, d in zip(pairs, sym, mod):
    if d == 0:
        print(f"  {labels[a]:>8s} - {labels[b]:<8s}  D = {d:.2f}, symmetric difference {s}")

# ---------------------------------------------------------------------------
# Why this piece: the selection is reproducible, not a lucky find.
from collections import defaultdict
from corpus_distances import load_real_book, key_reliable
from diagnostic_vocabulary import VOCABULARY

KIND_NAME = {tuple(k): name for name, k, _ in VOCABULARY}
# the five kinds chord-scale theory names, each with its added-ninth variant
FAMILY = {"Cmaj7": "maj7", "Cmaj9": "maj7", "C7": "7", "C9": "7",
          "Cmi7": "mi7", "Cmi9": "mi7", "Cmi7b5": "mi7b5",
          "Cmi6": "mi6", "Cmi69": "mi6"}

print("\n-- selection of the piece")
songs, titles, song_ids = load_real_book()
reliable = key_reliable(song_ids)
found = []
for song, title, rel in zip(songs, titles, reliable):
    total = sum(d for _, d in song)
    if not rel or total <= 0:
        continue
    families, covered = set(), 0.0
    for (_, kind), d in song:
        family = FAMILY.get(KIND_NAME.get(tuple(kind)))
        if family:
            families.add(family)
            covered += d
    if len(families) == 5 and covered / total >= 0.85:
        found.append((covered / total, len({c for c, _ in song}), title))

found.sort(key=lambda r: (-r[0], r[1]))
print(f"  songs using all five kinds for at least 85% of their duration: {len(found)}")
print(f"  {'share':>7s}{'chords':>8s}  title")
for share, distinct, title in found[:5]:
    print(f"  {share:7.0%}{distinct:8d}  {title}")

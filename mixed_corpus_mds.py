"""Exploratory MDS map of a mixed jazz / common-practice corpus (Section 8.2).

The Real Book is placed alongside the tonal works of the When-in-Rome corpus
(baroque, classical, and romantic), reading the relevant block of the union
distance matrix built by build_union_distances.py.

Duration weighting matters here more than anywhere else.  The two corpora are
annotated at different rhythmic granularities -- the Real Book on a coarse chart
grid, When-in-Rome by fine-grained harmonic analysis -- so counting chord tokens
would let a difference of editorial convention pass for a difference of harmonic
style.  Weighting by the time a chord occupies removes that confound.

Collection normalisation is likewise applied identically to both corpora (see
corpus_distances.py): every minor key is mapped to its relative major before
transposition.  The Real Book annotates only about 6% of its songs in minor,
against 61% of the baroque works; without a shared convention those minor works
would sit a minor third away from the jazz songs for a purely notational reason.

The jazz side is restricted to the songs whose annotated key is corroborated
(Section 8.1): 2,534 of 2,829.  Restricting is exact rather than approximate,
being a submatrix of a matrix whose entries never depended on the other songs.

Deliberately clustering-free: the geometry is anchored by descriptive statistics
only.

Run:  python mixed_corpus_mds.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from article_setup import output_directory
from figure_style import FIGURE_PAD, save_article_figure
from corpus_distances import (
    load_corpus, distance_matrix, descriptive_statistics, classical_mds,
    key_reliable,
)

OUT = output_directory()
STYLE_ORDER = ["jazz", "baroque", "classical", "romantic"]
GROUP_COLOUR = {"common-practice": "#ff7f0e", "jazz": "#1f77b4"}

# Every jazz song is shown, so the two groups differ in size by a factor of six.
# The jazz cloud is therefore drawn sheer: individual songs recede and the cloud
# reads as a density, against which the common-practice works stand out as
# distinct points.  Jazz goes underneath.
#
# 0.15 is a deliberate compromise.  Above it the dense core saturates to full
# colour and the cloud loses its internal density gradient; below it the sparse
# jazz songs at the edges -- the ones carrying the point that jazz reaches
# beyond the common-practice region -- become invisible.
JAZZ_ALPHA = float(os.environ.get("JAZZ_ALPHA", 0.15))
CP_ALPHA = 0.60

songs, titles, song_ids, styles, n_jazz = load_corpus()
D_all = distance_matrix(songs, "duration")
_, _, _, size_all = descriptive_statistics(songs, "duration")

# The key filter of Section 8.1.  Restricting is a submatrix operation and
# nothing is recomputed: D(i,j) never consults a song other than i and j, so
# every surviving distance is unchanged by the removal of the others.
reliable = np.concatenate([key_reliable(song_ids[:n_jazz]),
                           np.ones(len(songs) - n_jazz, bool)])
print(f"\nkey filter: {reliable[:n_jazz].sum()} of {n_jazz} jazz songs retained "
      f"({reliable[:n_jazz].mean():.1%}); all common-practice works kept")
keep = np.flatnonzero(reliable)
D_all = D_all[np.ix_(keep, keep)]
styles = styles[keep]
titles = [titles[i] for i in keep]
song_ids = [song_ids[i] for i in keep]
songs = [songs[i] for i in keep]
size_all = size_all[keep]
print(f"retained jazz corpus: "
      f"{len({c for i in np.flatnonzero(styles == 'jazz') for c, _ in songs[i]})} "
      f"distinct rooted chords after normalisation")

present = [s for s in STYLE_ORDER if s in set(styles)]
group = {s: np.where(styles == s)[0] for s in present}
jz, savant = group["jazz"], np.concatenate([group[s] for s in present if s != "jazz"])

print(f"\ncorpus: { {s: len(group[s]) for s in present} }  (total {len(songs)})")

print("\nmean song distance within / between styles (whole corpus):")
print("         " + "".join(f"{s[:7]:>9s}" for s in present))
for a in present:
    line = f"  {a[:7]:7s}"
    for b in present:
        block = D_all[np.ix_(group[a], group[b])]
        m = (block[np.triu_indices(len(group[a]), 1)].mean() if a == b else block.mean())
        line += f"{m:9.3f}"
    print(line)

print(f"\njazz <-> common-practice (mean)            = {D_all[np.ix_(jz, savant)].mean():.3f}")

X, variance_share = classical_mds(D_all)
if X[jz, 0].mean() < X[:, 0].mean():  # orient jazz to the right
    X[:, 0] *= -1
print(f"\n[Mixed]  n = {len(songs)}, MDS top-2 variance share = {variance_share:.0%}")

# Which axis actually separates the two repertoires?  The gap between the group
# means, in units of the axis's own standard deviation, answers it directly.
# It is the second axis, not the first: the first tracks mean chord size and
# runs through both corpora rather than between them.
print("  separation of the two groups along each axis:")
for a in (0, 1):
    gap = abs(X[jz, a].mean() - X[savant, a].mean()) / X[:, a].std()
    r = np.corrcoef(X[:, a], (styles == "jazz").astype(float))[0, 1]
    print(f"    axis {a + 1}: mean gap = {gap:.2f} sd, corr with group = {r:+.3f}")

# What the first axis is made of.  Mean chord size counts notes, root included.
print("  corr(axis, mean chord size):")
for a in (0, 1):
    print(f"    axis {a + 1}: whole corpus {np.corrcoef(X[:, a], size_all)[0, 1]:+.2f}, "
          f"jazz alone {np.corrcoef(X[jz, a], size_all[jz])[0, 1]:+.2f}")

# Is the second axis merely chord size?  Remove chord size (and its square) from
# it and re-measure the group gap.  Most of the separation survives, so the
# tradition difference the axis carries is not a density artefact.
Z = np.column_stack([np.ones(len(X)), size_all, size_all ** 2])
beta, *_ = np.linalg.lstsq(Z, X[:, 1], rcond=None)
resid = X[:, 1] - Z @ beta
raw_gap = abs(X[jz, 1].mean() - X[savant, 1].mean()) / X[:, 1].std()
res_gap = abs(resid[jz].mean() - resid[savant].mean()) / resid.std()
print(f"  second axis vs chord size: chord size explains "
      f"{1 - resid.var() / X[:, 1].var():.0%} of its variance; "
      f"group gap {raw_gap:.2f} -> {res_gap:.2f} sd once chord size is removed")

# Two-colour map: jazz versus a merged common-practice group.  The three
# common-practice eras do not separate under this geometry (see the table
# above), so displaying them as one group makes the figure legible.
fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.scatter(X[jz, 0], X[jz, 1], c=GROUP_COLOUR["jazz"], s=10,
           alpha=JAZZ_ALPHA, linewidths=0, zorder=1)
ax.scatter(X[savant, 0], X[savant, 1], c=GROUP_COLOUR["common-practice"], s=12,
           alpha=CP_ALPHA, linewidths=0, zorder=2)
# Legend markers drawn opaque: at the alpha used for the jazz cloud they would
# be invisible in the key.
ax.legend(
    handles=[
        Line2D([0], [0], marker="o", linestyle="none", markersize=5,
               color=GROUP_COLOUR["common-practice"],
               label=f"common-practice (n={len(savant)})"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=5,
               color=GROUP_COLOUR["jazz"], label=f"jazz (n={len(jz)})"),
    ],
    loc="best", frameon=False, fontsize=8,
)
ax.set_xlabel("MDS dimension 1")
ax.set_ylabel("MDS dimension 2")
fig.tight_layout(pad=FIGURE_PAD)
save_article_figure(fig, OUT, "mixed_corpus_mds")
print("\nsaved mixed_corpus_mds.png and .pdf")

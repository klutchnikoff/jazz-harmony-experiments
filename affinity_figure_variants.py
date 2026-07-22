"""Alternative layouts for the affinity figure, kept for comparison.

The article uses the compact transposed layout (variant 3), produced by
modal_affinities.py.  This script renders the alternatives that were weighed
against it, into tmp/ rather than fig/: they are not article figures.

The transposed affinity figure: chord kinds as rows, modes as columns, two
panels (W_D | uniform).  Twenty kinds do not fit as columns at the article's
cell size, but they fit comfortably as rows.

Renders two variants so they can be compared:
  vocab  the 20 kinds of the diagnostic vocabulary, grouped by family
         (what Table 2 lists and what the 14-vs-6 count is computed over)
  top20  the 20 most frequent kinds of the corpus, by token count

Usage:  python py-code/_explore_transposed.py <output-directory>
"""
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import article_setup  # noqa: F401
from article_setup import PACKAGE_ROOT
from diagnostic_vocabulary import VOCABULARY, FAMILY_ORDER
from figure_style import HEATMAP_CMAP
from leadsheetanalyser.constants import W_PYTHAGOREAN, PYTHAGOREAN_MODE_NAMES

from article_setup import ARTICLE_ROOT
OUT = sys.argv[1] if len(sys.argv) > 1 else str(ARTICLE_ROOT / "tmp")
(ARTICLE_ROOT / "tmp").mkdir(exist_ok=True)

CELL = 0.30          # inches; the article's 0.4125 is too tall for 20 rows
LEFT_LABELS = 1.05
BOTTOM = 0.55
TOP = 0.45
GAP = 0.28
CBAR_PAD, CBAR_W = 0.14, 0.10

WD = np.asarray(W_PYTHAGOREAN, float)
SUPPORT = (WD > 0).astype(float)
W_UNIFORM = SUPPORT / SUPPORT.sum(axis=1, keepdims=True)
BRIGHT = ["Lydian", "Ionian", "Mixolydian", "Dorian", "Aeolian", "Phrygian", "Locrian"]
ORDER = [PYTHAGOREAN_MODE_NAMES.index(m) for m in BRIGHT]
SHORT = ["Lyd", "Ion", "Mix", "Dor", "Aeo", "Phr", "Loc"]

PANELS = ((WD, r"theory-guided $W_D$"), (W_UNIFORM, "uniform on the same supports"))


def maxima(row, tol=1e-9):
    return np.flatnonzero(row >= row.max() - tol)


def draw(kinds, labels, breaks, filename, caption, cell=CELL, values=True):
    """kinds: list of 11-vectors, one per row; breaks: row indices to rule above."""
    nrows, ncols = len(kinds), len(BRIGHT)
    mw, mh = ncols * cell, nrows * cell
    group = 2 * mw + GAP + CBAR_PAD + CBAR_W
    fw = group + LEFT_LABELS
    fh = BOTTOM + mh + TOP
    fig = plt.figure(figsize=(fw, fh))
    left = LEFT_LABELS / fw

    axes = []
    for p in range(2):
        ax = fig.add_axes([left + p * (mw + GAP) / fw, BOTTOM / fh, mw / fw, mh / fh])
        axes.append(ax)
    cax = fig.add_axes([left + (2 * mw + GAP + CBAR_PAD) / fw, BOTTOM / fh,
                        CBAR_W / fw, mh / fh])

    for ax, (W, title) in zip(axes, PANELS):
        P = np.asarray([(W @ np.asarray(k, float))[ORDER] for k in kinds])
        image = ax.imshow(P, cmap=HEATMAP_CMAP, vmin=0, vmax=1.0, aspect="equal")
        ax.set_title(title, pad=5, fontsize=9)
        ax.set_xticks(range(ncols), labels=SHORT, rotation=45, ha="right", fontsize=7)
        if ax is axes[0]:
            ax.set_yticks(range(nrows), labels=labels, fontsize=7)
        else:
            ax.set_yticks([])
        for i in range(nrows):
            for j in maxima(P[i]):
                ax.add_patch(matplotlib.patches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red",
                    linewidth=1.0, zorder=5))
        if values:
            for i in range(nrows):
                for j in range(ncols):
                    v = P[i, j]
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.4,
                            color="white" if v >= 0.5 else "black")
        for b in breaks:
            ax.axhline(b - 0.5, color="0.25", linewidth=0.9, zorder=6)

    fig.colorbar(image, cax=cax, label=r"$[\Phi_{W}(k)]_j$")
    path = f"{OUT}/{filename}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}   ({caption})")

    for W, title in PANELS:
        rows = np.asarray([W @ np.asarray(k, float) for k in kinds])
        unique = sum(1 for r in rows if len(maxima(r)) == 1)
        print(f"    {title:34s} unique maximum: {unique}/{nrows}")


# --- variant 1: the diagnostic vocabulary, grouped by family ---------------
by_family = sorted(VOCABULARY, key=lambda t: (FAMILY_ORDER.index(t[2]),))
kinds = [k for _, k, _ in by_family]
labels = [n.replace("C", "", 1) or "maj" for n, _, _ in by_family]
breaks, seen = [], None
for i, (_, _, fam) in enumerate(by_family):
    if fam != seen and i:
        breaks.append(i)
    seen = fam
draw(kinds, labels, breaks, "affinities_vocab_transposed",
     "diagnostic vocabulary, grouped by intervallic family")

# --- variant 2: the 20 most frequent kinds --------------------------------
df = pd.read_pickle(PACKAGE_ROOT / "data" / "music_realbook.pkl")
counts = Counter()
for prog in df["chord_progression"]:
    for c in prog:
        if c is None or c[0] is None or any(x is None for x in c[1:12]):
            continue
        counts[tuple(int(x) for x in c[1:12])] += 1
NAME = {tuple(k): n for n, k, _ in VOCABULARY}
NOTE = ["1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"]
top20 = counts.most_common(20)
tk = [k for k, _ in top20]
tl = []
for k, n in top20:
    name = NAME.get(k)
    tl.append((name.replace("C", "", 1) or "maj") if name
              else "[" + " ".join(NOTE[j + 1] for j in range(11) if k[j]) + "]")
draw(tk, tl, [], "affinities_top20_transposed",
     "20 most frequent kinds, unnamed ones shown by interval content")

# --- variant 3: compact, no cell values -----------------------------------
draw(kinds, labels, breaks, "affinities_vocab_compact",
     "diagnostic vocabulary, compact (no cell values)", cell=0.22, values=False)

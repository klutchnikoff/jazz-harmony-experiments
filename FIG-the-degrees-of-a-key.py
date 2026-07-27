"""Figure for Section 5.2, "The degrees of a key".

Two panels in the grammar of Figure 1, so that nothing has to be relearned:
modes down the rows, one column per degree, every column summing to one, the
largest share boxed in red or in orange when it is attained more than once.

  left    C major.  The seven diatonic degrees, then a rule, then the eight
          chromatic ones the section discusses: the five borrowed from the
          parallel minor, the flattened supertonic, and the two blues sevenths.
  right   A minor, read from its own tonic.  Possible only since the loader
          stopped folding a minor key onto its relative major, which would have
          read this panel from C and put Aeolian beyond reach.

Sevenths throughout, and the diatonic ones are the diatonic sevenths: the
dominant is V7 and not Vmaj7, the leading-note chord half-diminished.  Triads
were tried and read their key's mode 3 times in 7 against 5 for the sevenths,
a triad leaving too many modes containing it.

Run:  LSA_LOCAL=1 .venv/bin/python FIG-the-degrees-of-a-key.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

from chord_scale import SYSTEM, MODES
from figure_style import HEATMAP_CMAP, save_article_figure

ORDER = 0.15
OUT = Path(__file__).resolve().parents[1] / "TeX" / "fig"
SOLE, TIED = "#c0392b", "#e08214"

MAJ7, MI7, DOM, HALF = (4, 7, 11), (3, 7, 10), (4, 7, 10), (3, 6, 10)
FLAT = "♭"

# (label, root above the tonic, kind, the mode harmony predicts)
#
# The prediction owes nothing to the system: a major key's own mode is Ionian
# and a minor key's Aeolian, the parallel minor a major key borrows from has
# Aeolian for its mode, the flattened supertonic is the degree that defines
# Phrygian, a tonic carrying a flattened seventh is Mixolydian, and IV7 brings
# the minor third with the major sixth, which is Dorian.  Unlike the pairings of
# Figure 1 these were written down after the readings were computed, so they
# corroborate rather than test.
I_, A_, M_, D_, P_ = "Ionian", "Aeolian", "Mixolydian", "Dorian", "Phrygian"
C_MAJOR = [("I", 0, MAJ7, I_), ("ii", 2, MI7, I_), ("iii", 4, MI7, I_),
           ("IV", 5, MAJ7, I_), ("V", 7, DOM, I_), ("vi", 9, MI7, I_),
           ("viiø", 11, HALF, I_)]
C_CHROMATIC = [("iiø", 2, HALF, A_), (f"{FLAT}III", 3, MAJ7, A_),
               ("iv", 5, MI7, A_), (f"{FLAT}VI", 8, MAJ7, A_),
               (f"{FLAT}VII", 10, DOM, A_), (f"{FLAT}II7", 1, DOM, P_),
               ("I7", 0, DOM, M_), ("IV7", 5, DOM, D_)]
A_MINOR = [("i", 0, MI7, A_), ("iiø", 2, HALF, A_), ("III", 3, MAJ7, A_),
           ("iv", 5, MI7, A_), ("v", 7, MI7, A_), ("VI", 8, MAJ7, A_),
           ("VII", 10, DOM, A_)]
# A prediction of None is a degree whose mode harmony would name is not among the
# nine: the raised leading note of a minor key belongs to the harmonic minor,
# which Section 3 does not carry.  Those columns carry no dot, which is the
# honest way to draw an expectation the system cannot hold.
DIM7 = (3, 6, 9)
A_ALTERED = [("V7", 7, DOM, None), ("vii°7", 11, DIM7, None),
             ("ii", 2, MI7, D_), (f"{FLAT}II", 1, MAJ7, P_),
             ("IV7", 5, DOM, D_), ("i6", 0, (3, 7, 9), D_),
             ("I", 0, MAJ7, I_)]

CELL, LEFT, BOTTOM, TOP, GAP = 0.205, 1.02, 0.78, 0.30, 0.40


def reading(offset, kind):
    """The degree at `offset` above the tonic, read as a distribution."""
    content = {(offset + i) % 12 for i in (0,) + tuple(kind)} | {0}
    intervals = [i - 1 for i in range(1, 12) if i in content]
    m = np.mean(SYSTEM[:, intervals] ** ORDER, axis=1) ** (1 / ORDER)
    return m / m.sum()


def panel(ax, degrees, norm, rows=True):
    P = np.array([reading(o, k) for _, o, k, _w in degrees])
    ax.imshow(P.T, cmap=HEATMAP_CMAP, aspect="equal", norm=norm,
              interpolation="nearest")
    for column, col in enumerate(P):
        hit = np.flatnonzero(col >= col.max() - 1e-12)
        edge = TIED if len(hit) > 1 else SOLE
        for row in hit:
            ax.add_patch(Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False,
                                   edgecolor=edge, lw=1.0, zorder=3))
    for column, (_, _o, _k, want) in enumerate(degrees):
        if want is not None:
            ax.plot(column, MODES.index(want), "o", ms=2.8, color=SOLE,
                    zorder=4)
    ax.set_xticks(range(len(degrees)))
    ax.set_xticklabels([d[0] for d in degrees], rotation=90, fontsize=7.5)
    ax.set_yticks(range(9))
    ax.set_yticklabels(MODES if rows else [], fontsize=7.5)
    ax.tick_params(length=0, pad=2)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    return P


def draw(groups, norm, stem):
    """One figure, one panel per (title, degrees) pair, sharing a colour bar."""
    widths = [len(d) * CELL for _, d in groups]
    height_matrix = 9 * CELL
    width = LEFT + sum(widths) + GAP * (len(groups) - 1) + 0.42
    height = BOTTOM + height_matrix + TOP
    fig = plt.figure(figsize=(width, height))

    x, first = LEFT, None
    for (title, degrees), w in zip(groups, widths):
        ax = fig.add_axes([x / width, BOTTOM / height,
                           w / width, height_matrix / height])
        panel(ax, degrees, norm, rows=first is None)
        ax.set_title(title, fontsize=8.5, pad=4)
        first = first or ax
        x += w + GAP

    cax = fig.add_axes([(x - GAP + 0.10) / width, BOTTOM / height,
                        0.08 / width, height_matrix / height])
    fig.colorbar(first.images[0], cax=cax,
                 ticks=[0, 0.1, 0.25, 0.5, 0.75]).ax.tick_params(length=2,
                                                                 labelsize=7)
    OUT.mkdir(parents=True, exist_ok=True)
    save_article_figure(fig, OUT, stem)
    print(f"written to {OUT}/{stem}.pdf")


def main():
    everything = np.array([reading(o, k) for _, o, k, _w
                           in C_MAJOR + C_CHROMATIC + A_MINOR + A_ALTERED])
    norm = PowerNorm(gamma=0.55, vmin=0, vmax=everything.max())

    draw([("C major", C_MAJOR), ("A minor", A_MINOR)], norm, "degrees-of-a-key")
    draw([("chromatic in C major", C_CHROMATIC),
          ("altered in A minor", A_ALTERED)], norm, "borrowed-degrees")


if __name__ == "__main__":
    main()

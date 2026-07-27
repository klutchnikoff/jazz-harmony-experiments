"""Figure for Section 4.5, "The order used below".

Two panels, stacked, sharing one story.

  above   Concentration and separation over K_0 against the order p, with the
          floor the linear reading sets, the crossing at which the falling
          separation returns to it, and the order chosen just above.  The point
          of the panel is the shape: concentration climbs throughout, separation
          does not, which is why a floor is needed at all.

  below   The 32 kinds read at p = 0.15, one column each, nine modes down the
          rows.  Every column sums to one, so a column is a reading.  The mode
          carrying the largest share is boxed in red, and four kinds carry two
          boxes: their maximum is attained twice.

The kinds run by family and, within a family, by the symbols they account for in
the jazz corpus, as in Table 1.  They are labelled without their root, the table
writing them above C.

Run:  LSA_LOCAL=1 .venv/bin/python FIG-the-order-used-below.py
"""
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

from chord_scale import SYSTEM, MODES, PAIRING
from figure_style import HEATMAP_CMAP, save_article_figure
from vocabulary import build, corpus_counts, family, name, FAMILIES

ORDER = 0.15
OUT = Path(__file__).resolve().parents[1] / "TeX" / "fig"

WIDTH = 6.0
CELL = 0.148
LEFT = 0.92           # room for the mode names
CURVE_HEIGHT = 1.55
GAP = 0.86            # room for the kind symbols under the map
BOTTOM = 0.60


def readings(vocab, p):
    rows = []
    for k in vocab:
        intervals = [i for i in range(11) if k[i]]
        m = np.mean(SYSTEM[:, intervals] ** p, axis=1) ** (1 / p)
        rows.append(m / m.sum())
    return np.array(rows)


def separation(P):
    return min(np.abs(P[a] - P[b]).sum()
               for a, b in itertools.combinations(range(len(P)), 2))


def main():
    tokens = corpus_counts()[0]["jazz"]
    # grouped by the mode each kind reads, so the boxed cells descend a
    # staircase and the classes of the reading are visible as blocks; the
    # families of Table 1 stay legible in the symbols themselves
    all_kinds = build()[0]
    top_of = {k: int(np.argmax(readings([k], ORDER)[0])) for k in all_kinds}
    vocab = sorted(all_kinds, key=lambda k: (top_of[k], -tokens[k]))
    edges = np.cumsum([sum(1 for k in vocab if top_of[k] == j)
                       for j in range(9)])
    edges = [e for e in edges[:-1] if 0 < e < len(vocab)]

    grid = np.geomspace(0.02, 1.0, 120)
    concentration, gaps = [], []
    for p in grid:
        P = readings(vocab, p)
        concentration.append(np.median(P.max(axis=1)))
        gaps.append(separation(P))
    floor = gaps[-1]
    # by bisection, not off the plotting grid: the two must agree to the third
    # decimal with the value Section 4.5 states
    lo, hi = 0.10, 0.20
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = ((mid, hi) if separation(readings(vocab, mid)) < floor
                  else (lo, mid))
    crossing = hi

    matrix_width, matrix_height = len(vocab) * CELL, 9 * CELL
    height = BOTTOM + GAP + matrix_height + CURVE_HEIGHT + 0.34
    fig = plt.figure(figsize=(WIDTH, height))

    def box(bottom, h):
        return [LEFT / WIDTH, bottom / height, matrix_width / WIDTH, h / height]

    ax = fig.add_axes(box(BOTTOM + GAP + matrix_height + 0.34, CURVE_HEIGHT))
    ax.plot(grid, concentration, color="#1f4e79", lw=1.4,
            label="concentration")
    ax.plot(grid, gaps, color="#b03a2e", lw=1.4, label="separation")
    ax.axvspan(0.02, crossing, color="0.92", zorder=0, lw=0)
    ax.axhline(floor, color="0.45", lw=0.7, ls=(0, (4, 3)))
    ax.plot([crossing], [floor], "o", ms=5, mfc="white", mec="0.25", mew=1.1,
            zorder=4)
    ax.plot([ORDER], [np.interp(ORDER, grid, concentration)], "o",
            ms=4, color="#1f4e79")
    ax.plot([ORDER], [np.interp(ORDER, grid, gaps)], "o", ms=4,
            color="#b03a2e")
    ax.set_xscale("log")
    ax.set_xlim(1.0, 0.02)
    ax.set_ylim(0, 0.62)
    ax.set_xticks([1.0, 0.5, 0.2, ORDER, 0.05, 0.02])
    ax.set_xticklabels(["1", "0.5", "0.2", "0.15", "0.05", "0.02"])
    ax.set_xlabel("order $p$", labelpad=1)
    ax.tick_params(length=2, pad=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.annotate("separation at $p=1$", (0.92, floor - 0.012), fontsize=7,
                color="0.35", va="top")
    ax.annotate(f"$p={crossing:.3f}$", (crossing * 0.94, floor + 0.035),
                fontsize=7, color="0.25", ha="right", va="bottom")
    ax.annotate("orders ruled out", (0.028, 0.35), fontsize=7, color="0.45",
                ha="left", va="center")
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              handlelength=1.4, borderaxespad=0.2)

    P = readings(vocab, ORDER)
    hm = fig.add_axes(box(BOTTOM + GAP, matrix_height))
    # a column is a distribution over nine modes, so most cells sit well under
    # the 0.94 of the darkest: a linear scale to 1 would wash the map out
    norm = PowerNorm(gamma=0.55, vmin=0, vmax=P.max())
    im = hm.imshow(P.T, cmap=HEATMAP_CMAP, aspect="equal", norm=norm,
                   interpolation="nearest")
    # every cell attaining the maximum, not argmax: four kinds have no single
    # top mode, Dorian and Phrygian carrying the same weights permuted over the
    # intervals of a minor seventh, and a box on one of them would be arbitrary
    for column, col in enumerate(P):
        for row in np.flatnonzero(col >= col.max() - 1e-12):
            hm.add_patch(Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False,
                                   edgecolor="#c0392b", lw=1.0, zorder=3))
    # the five families are separated but not named: the kind symbols below say
    # which is which, and labels over the three narrow blocks collide
    for column, k in enumerate(vocab):
        want = PAIRING.get(name(k))
        if want is not None:
            hm.plot(column, MODES.index(want), "o", ms=2.6,
                    color="#c0392b", zorder=4)
    for e in edges:
        hm.axvline(e - 0.5, color="0.35", lw=0.8)
    hm.set_yticks(range(9))
    hm.set_yticklabels(MODES, fontsize=7.5)
    hm.set_xticks(range(len(vocab)))
    hm.set_xticklabels([name(k) for k in vocab], rotation=90, fontsize=6.5)
    hm.tick_params(length=0, pad=2)
    for side in hm.spines:
        hm.spines[side].set_visible(False)

    cax = fig.add_axes([(LEFT + matrix_width + 0.11) / WIDTH,
                        (BOTTOM + GAP) / height, 0.08 / WIDTH,
                        matrix_height / height])
    fig.colorbar(im, cax=cax, ticks=[0, 0.1, 0.25, 0.5, 0.9]).ax.tick_params(
        length=2, labelsize=7)

    OUT.mkdir(parents=True, exist_ok=True)
    save_article_figure(fig, OUT, "order-used-below")
    print(f"crossing {crossing:.3f}   floor {floor:.4f}   "
          f"largest cell {P.max():.3f}")
    print(f"written to {OUT}/order-used-below.pdf")


if __name__ == "__main__":
    main()

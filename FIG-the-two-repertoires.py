"""Figure for Section 6.2, "The two repertoires".

Four mean representations, as bars rather than as the heat maps of Figures 1 to
3.  Those had thirty-two columns and a heat map was the only way to show them at
once; here there are four profiles, and colour compares magnitudes badly where a
bar height compares them directly.

  left    major-key works, jazz against common practice
  right   minor-key works, the same

The nine modes run along each axis in the order of Section 3.2, brightest first,
so that the reversal the subsection reports is a leftward shift of the jazz bars
in both panels.  A mode whose difference clears the overall family-wise 5 % of
the max-T permutation tests and their Bonferroni correction carries a mark; the
modes that do not are faded.  Four of the nine are marked on the left, five on
the right.

The representations and max-T marks use the same tested functions as
ART-the-two-repertoires.py, so the figure cannot maintain a parallel
implementation of the analysis.

Run:  LSA_LOCAL=1 .venv/bin/python FIG-the-two-repertoires.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from article_analysis import (
    TonicModalReader,
    grouped_work_profiles,
    load_annotated_modes,
    profile_permutation_test,
    profile_stream,
)
from article_setup import cache_directory
from chord_scale import MODES, SYSTEM
from corpus import key_exact, load_corpus
from figure_style import save_article_figure

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[0] / "TeX" / "fig"
JAZZ, CP = "#1f4e79", "#a3c4dc"
FADED = 0.38          # modes whose difference does not clear the test
PERMUTATIONS = 20_000
KEY_TYPE_TESTS = 2
ORDER = 0.15
READER = TonicModalReader(SYSTEM, ORDER)


def profiles():
    """Mean representation of each of the four groups, and which modes differ."""
    songs, _titles, ids, _styles, n_jazz = load_corpus()
    keep = key_exact(ids)
    modes = load_annotated_modes(cache_directory())
    works = grouped_work_profiles(songs, ids, keep, n_jazz, modes, READER)

    means, marked = {}, {}
    for m in ("major", "minor"):
        A = works[("J", m)]
        B = works[("C", m)]
        means[m] = (A.mean(0), B.mean(0), len(A), len(B))
        test = profile_permutation_test(
            A,
            B,
            PERMUTATIONS,
            profile_stream(m),
            KEY_TYPE_TESTS,
        )
        marked[m] = list(test.adjusted_coordinate_p <= 0.05)
    return means, marked


def main():
    means, marked = profiles()

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5), sharey=True)
    x = np.arange(9)
    for ax, m in zip(axes, ("major", "minor")):
        jz, cp, n_jz, n_cp = means[m]
        left = ax.bar(x - 0.21, jz, 0.40, color=JAZZ, label=f"jazz ({n_jz})")
        right = ax.bar(x + 0.21, cp, 0.40, color=CP,
                       label=f"common practice ({n_cp})")
        # the mark survives a monochrome print, the fading reads on a screen
        for j, star in enumerate(marked[m]):
            if star:
                ax.plot(j, max(jz[j], cp[j]) + 0.012, marker=(6, 2, 0), ms=4,
                        color="0.35", lw=0.8)
            else:
                left[j].set_alpha(FADED)
                right[j].set_alpha(FADED)
        ax.set_title(f"{m}-key works", fontsize=8.5, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(MODES, rotation=90, fontsize=7.5)
        ax.tick_params(length=2, pad=2, labelsize=7.5)
        ax.set_ylim(0, 0.37)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right",
                  handlelength=1.1, borderaxespad=0.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(r"coordinate of $\overline{\Phi}_p$", fontsize=8)

    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.30, wspace=0.08)
    OUT.mkdir(parents=True, exist_ok=True)
    save_article_figure(fig, OUT, "two-repertoires")
    print(f"written to {OUT}/two-repertoires.pdf")


if __name__ == "__main__":
    main()

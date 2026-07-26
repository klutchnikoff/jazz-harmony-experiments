"""Article data for Section 4.2, "The chord drawn from the mode".

The subsection claims the reading is sharp, and the figures behind that claim are
how many of the nine modes contain each kind: a mode missing one interval of the
chord is excluded outright by this likelihood.

Run:  LSA_LOCAL=1 .venv/bin/python ART-the-chord-drawn-from-the-mode.py
"""
import numpy as np

from article_data import export
from chord_scale import SYSTEM
from vocabulary import build, name


def main():
    vocab = build()[0]
    containing = []
    for k in vocab:
        intervals = [i for i in range(11) if k[i]]
        containing.append(sum(all(SYSTEM[j, i] > 0 for i in intervals)
                              for j in range(SYSTEM.shape[0])))
    containing = np.array(containing)

    print(f"\nmodes containing a kind, over the {len(vocab)} of the vocabulary")
    for n in sorted(set(containing)):
        print(f"   {n} of nine: {int((containing == n).sum()):2d} kinds")
    print(f"   mean {containing.mean():.2f}, median {int(np.median(containing))}")
    orphans = [name(k) for k, c in zip(vocab, containing) if c == 0]
    print(f"   contained in no mode: {len(orphans)} {orphans}")

    export("the-chord-drawn-from-the-mode", {
        "kinds_in_one_mode": f"{int((containing == 1).sum())} lie in a single mode",
        "kinds_in_at_most_two": f"{int((containing <= 2).sum())} in no more than two",
    })


if __name__ == "__main__":
    main()

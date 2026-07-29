"""Article data for Section 4.2, "The two extremes".

The subsection sets the two extreme readings against each other on the same two
statistics, and every figure it states is produced here.

  Phi_1  normalized [W_0 k]_j, the arithmetic end.
  Phi_0  normalized (prod_{i in k} w^j_i)^{1/|k|}, the geometric end.

Phi_0 is the normalized geometric mean, not the normalized product: the |k|th
root is taken in Section 4.1 so that both readings compare means of the same |k|
weights.  The distinction is not cosmetic.  On the raw product the median largest
coordinate is 0.70 and the 32 kinds give 21 distinct readings; on the geometric
mean they are 0.56 and 25, because the root separates kinds that only differed in
length.  The article uses the geometric mean throughout, this being the p = 0
member of the family of Section 4.3.

Run:  LSA_LOCAL=1 .venv/bin/python ART-what-the-two-readings-cost.py
"""
import numpy as np

from article_data import export
from chord_scale import SYSTEM, MODES, PAIRING
from vocabulary import build, name


def readings(vocab):
    """The two posteriors of Section 4.1, one row per kind."""
    phi1, phi0 = [], []
    for k in vocab:
        intervals = [i for i in range(11) if k[i]]
        linear = SYSTEM[:, intervals].sum(axis=1)
        geometric = np.prod(SYSTEM[:, intervals], axis=1) ** (1 / len(intervals))
        phi1.append(linear / linear.sum())
        phi0.append(geometric / geometric.sum())
    return np.array(phi1), np.array(phi0)


def distinct(P):
    return len({tuple(np.round(row, 9)) for row in P})


def classes(P, vocab):
    """The kinds a reading fails to tell apart, grouped."""
    groups = {}
    for k, row in zip(vocab, P):
        groups.setdefault(tuple(np.round(row, 9)), []).append(name(k))
    return [(key, ks) for key, ks in groups.items() if len(ks) > 1]


def main():
    vocab = sorted(build()[0], key=name)
    phi1, phi0 = readings(vocab)
    top1, top0 = phi1.max(axis=1), phi0.max(axis=1)

    print(f"\n{'':22s} {'max coord':>10s} {'median':>8s} {'distinct':>9s}")
    for label, P, top in (("Phi_1  arithmetic", phi1, top1),
                          ("Phi_0  geometric", phi0, top0)):
        print(f"{label:22s} {top.max():10.4f} {np.median(top):8.4f} "
              f"{distinct(P):9d}")
    print(f"{'uniform':22s} {1/9:10.4f}")

    collapsed = classes(phi0, vocab)
    print(f"\n{len(collapsed)} classes collapse under Phi_0, "
          f"{sum(len(ks) - 1 for _, ks in collapsed)} distinctions lost")
    for key, ks in sorted(collapsed, key=lambda t: -len(t[1])):
        spread = ", ".join(f"{MODES[j]} {key[j]:.3f}"
                           for j in range(9) if key[j] > 1e-12)
        print(f"   {', '.join(sorted(ks)):36s} -> {spread}")

    # Section 4.2 says each collapsed class gathers kinds a soloist plays one
    # scale over, and that the scale carries the largest coordinate.  Checked
    # against the reference pairings: no class holds two kinds pointing at
    # different scales, and the scale they point at is the class's top mode.
    for key, ks in collapsed:
        scales = {PAIRING[k] for k in ks if k in PAIRING}
        assert len(scales) <= 1, f"{ks} disagree on a scale: {scales}"
        if scales:
            top = MODES[int(np.argmax(key))]
            assert top == scales.pop(), f"{ks} do not read as their scale, {top}"

    # The manuscript states the injectivity of Phi_1 as a formula, with no number
    # for the checker to find, so the claim is verified here instead.
    assert distinct(phi1) == len(vocab), (
        f"Phi_1 is no longer injective: {distinct(phi1)} values for "
        f"{len(vocab)} kinds, yet Section 4.2 says Phi_1(k) = Phi_1(k') "
        "forces k = k'")

    point_mass = int((top0 > 1 - 1e-9).sum())
    words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}
    export("what-the-two-readings-cost", {
        # phrases, not bare integers: 32 and 10 occur throughout the manuscript
        "phi0_distinct": f"only {distinct(phi0)} distinct values",
        "phi0_point_mass": f"the {point_mass} kinds a single mode weighs",
        "collapsed_classes": f"{words[len(collapsed)]} classes collapse",
        # the bound deserves its third decimal, the medians do not
        "phi1_largest": f"{top1.max():.3f}",
        "phi1_median": f"{np.median(top1):.2f}",
        "phi0_median": f"{np.median(top0):.2f}",
        "uniform": f"{1/9:.3f}",
    })


if __name__ == "__main__":
    main()

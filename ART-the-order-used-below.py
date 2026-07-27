"""Article data for Section 4.5, "The order used below".

The order is fixed by a constraint rather than by taste.  Lowering p raises the
concentration of the reading throughout, but the separation of the kinds peaks
near p = 0.3 and then collapses, so there is a floor below which one must not go.

The floor taken is the separation the linear reading already achieves.  It is
crossed at p = 0.1393, and 0.15 is the round order comfortably above it, clearing
the threshold by 13 per cent where 0.14 would clear it by 0.7 per cent.

  concentration   median over K_0 of the largest coordinate of Phi_p.
  separation      smallest l1 distance between two readings of K_0.

Both are printed at the orders the subsection names, and the crossing is found by
bisection.  The medians are exported to two decimals, matching Section 4.2, which
states the same two statistics at p = 1 and p = 0.

Run:  LSA_LOCAL=1 .venv/bin/python ART-the-order-used-below.py
"""
import itertools
from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np
from scipy.spatial.distance import pdist

from article_data import export
from chord_scale import SYSTEM
from vocabulary import build, name

ORDER = 0.15


def profiles(vocab, p):
    rows = []
    for k in vocab:
        intervals = [i for i in range(11) if k[i]]
        m = np.mean(SYSTEM[:, intervals] ** p, axis=1) ** (1 / p)
        rows.append(m / m.sum())
    return np.array(rows)


def measure(vocab, p):
    """Concentration, separation, and the pair that attains the separation."""
    P = profiles(vocab, p)
    pairs = ((np.abs(P[a] - P[b]).sum(), a, b)
             for a, b in itertools.combinations(range(len(vocab)), 2))
    gap, a, b = min(pairs)
    return np.median(P.max(axis=1)), gap, (name(vocab[a]), name(vocab[b]))


def exact_readings(vocab, p, digits=60):
    """The readings again, in `digits`-place decimal arithmetic.

    Section 4.5 rests on a separation being strictly positive, which is a claim
    float64 alone should not carry.  The standard library suffices: a power is
    exp(p log w), and Decimal has both.
    """
    getcontext().prec = digits
    def dec(x):
        f = Fraction(x).limit_denominator(10 ** 6)
        return Decimal(f.numerator) / Decimal(f.denominator)
    W = [[dec(x) for x in row] for row in SYSTEM]
    order, inverse = dec(p), 1 / dec(p)
    rows = []
    for k in vocab:
        intervals = [i for i in range(11) if k[i]]
        m = []
        for j in range(9):
            mean = sum((order * W[j][i].ln()).exp()
                       for i in intervals) / Decimal(len(intervals))
            m.append((inverse * mean.ln()).exp())
        total = sum(m)
        rows.append([x / total for x in m])
    return rows


def main():
    vocab = sorted(build()[0], key=name)
    floor = measure(vocab, 1.0)[1]

    print(f"\n{'p':>6s} {'concentration':>14s} {'separation':>11s}   closest pair")
    for p in (1.0, 0.5, 0.3, 0.2, ORDER, 0.1, 0.05):
        c, gap, pair = measure(vocab, p)
        flag = "" if gap >= floor else "   below the floor"
        print(f"{p:6.2f} {c:14.3f} {gap:11.4f}   {pair[0]}/{pair[1]}{flag}")

    # where separation peaks, and where it falls back to the p = 1 floor
    grid = np.linspace(0.2, 0.5, 61)
    peak = max(grid, key=lambda p: measure(vocab, p)[1])
    lo, hi = 0.10, 0.20
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if measure(vocab, mid)[1] < floor else (lo, mid)
    print(f"\nseparation peaks at p = {peak:.2f}, and returns to the floor "
          f"at p = {hi:.4f}")

    c_one, _, _ = measure(vocab, 1.0)
    c_chosen, gap_chosen, pair_chosen = measure(vocab, ORDER)
    c_zero = np.median(
        (lambda P: P / P.sum(axis=1, keepdims=True))(
            np.array([np.prod(SYSTEM[:, [i for i in range(11) if k[i]]], axis=1)
                      ** (1 / sum(k)) for k in vocab])).max(axis=1))
    assert gap_chosen > floor, "the chosen order does not clear the floor"

    # a positive separation is injectivity, so this settles that the chosen
    # order is none of the exceptions Proposition 4.2 has to allow for
    E = exact_readings(vocab, ORDER)
    exact_gap = min(sum(abs(a - b) for a, b in zip(E[i], E[j]))
                    for i, j in itertools.combinations(range(len(E)), 2))
    assert exact_gap > 0, f"Phi_{ORDER} is not injective on K_0"
    print(f"at 60 decimal places the separation over K_0 is {exact_gap:.6e}")

    # and over all of K minus zero, where the closest pair is far tighter
    every = [b for b in itertools.product((0, 1), repeat=11) if any(b)]
    closest = pdist(profiles(every, ORDER), "cityblock").min()
    assert closest > 1e-9, "a collision among the nonzero kinds"
    print(f"over all {len(every)} nonzero kinds the closest pair is "
          f"{closest:.4e} apart")
    print(f"the pair binding the choice at p = {ORDER} is "
          f"{pair_chosen[0]}/{pair_chosen[1]}, at {gap_chosen:.4f} "
          f"against a floor of {floor:.4f}")

    export("the-order-used-below", {
        "order": f"p={ORDER}",
        # two decimals, as Section 4.2 states the same medians
        "concentration_one": f"{c_one:.2f}",
        "concentration_chosen": f"{c_chosen:.2f}",
        "concentration_zero": f"{c_zero:.2f}",
        "separation_floor": f"{floor:.3f}",
        "separation_peak": f"{measure(vocab, peak)[1]:.3f}",
        "crossing": f"{hi:.3f}",
    })


if __name__ == "__main__":
    main()
